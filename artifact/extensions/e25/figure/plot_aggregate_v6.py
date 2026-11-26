#!/usr/bin/env python3
"""Plot a whole-SHA-bound genuine complete three-block v6 aggregate only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import statistics

MIB = 1 << 20
GIB = 1 << 30
BLOCKS = ('1', '2', '3')
ARMS = ('reference_union_compiled', 'reference_union_native_persistent')
HORIZONS = {'short320': (320, 576), 'long6400': (6400, 6656)}
PINS = {
    'source_manifest_sha256': '3fd8783173e4a850018d860459881aafd1dd49e8d4cda101eb273641e47af086',
    'protocol_sha256': 'e89b371089bd01f71579fdc00ec06004373bb549542658cc223914759f392e18',
    'wrapper_sha256': '072575b15555a83f223e15643b869874f9a5e4f83172c06e0f4f5d032bdae4c9',
    'wrapper_ready_sha256': '88b06c57d7de42c062fbd2ea76cd268999d3bbabe8bbd31130a9ec79dfa5a785',
}
AUDITOR_SHA = 'b50f69169d8149d090913dcedf5a15aea8ed6d397650a276f432e1781b41deb5'
LONG_POINTS = (
    (32, 'first_trained_batch_launch_to_complete_seconds'),
    (256, 'warmup_complete_launch_to_complete_seconds'),
    (1280, 'measured_prefix1024_launch_to_complete_seconds'),
    (3456, 'measured_prefix3200_launch_to_complete_seconds'),
    (6656, 'measured_prefix6400_launch_to_complete_seconds'),
)
SHORT_POINTS = (LONG_POINTS[0], LONG_POINTS[1],
                (576, 'measured_prefix320_launch_to_complete_seconds'))
BASE_METRICS = {
    'launch_to_first_batch_ready_seconds', 'launch_to_actual_wait_seconds',
    'checkpointed_measured_seconds', 'worker_shutdown_seconds',
    'input_hash_seconds_included',
}
COLORS = {ARMS[0]: '#0072B2', ARMS[1]: '#D55E00'}
LABELS = {ARMS[0]: 'Compiled', ARMS[1]: 'File-backed native'}
CAPTION = (
    'Instrumented Friendster lifetimes across three fresh allocation blocks. '
    'Long-application curves join recorded synchronized checkpoints; medians and '
    'min–max whiskers are pointwise descriptive summaries, not one trajectory '
    'or confidence intervals. Unconnected squares/diamonds mark actual waits '
    'of separate short/long applications. Native includes fresh file-backed '
    'promotion from compact CSR. No long-run timed checkpoint exists at 576.'
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha_hex(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-10)


def pin(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * MIB,
         'bounded regular figure/source payload required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(MIB), b''):
            h.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def read_aggregate(path, expected_sha):
    need(sha_hex(expected_sha), 'explicit independently known aggregate whole SHA required')
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular aggregate file required')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 32 * MIB,
             'aggregate32MiB cap')
        raw = stream.read(32 * MIB + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and
         (before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'complete stable aggregate read required')
    need(hashlib.sha256(raw).hexdigest() == expected_sha, 'aggregate whole SHA differs')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate aggregate JSON key')
            result[key] = value
        return result

    def finite_float(token):
        value = float(token)
        need(math.isfinite(value), 'nonfinite aggregate number')
        return value

    data = json.loads(raw, object_pairs_hook=unique, parse_float=finite_float,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite aggregate JSON')))
    return data, dict(file=path.name, bytes=len(raw), sha256=expected_sha)


def describe(value, time_metric=True):
    need(type(value['n']) is int and value['n'] == 3 and set(value['by_block']) == set(BLOCKS),
         'all three original block observations required')
    raw = {b: value['by_block'][b] for b in BLOCKS}
    need(all(finite(v) and v >= 0 and (not time_metric or v <= 3900) for v in raw.values()),
         'finite nonnegative block values required')
    values = list(raw.values())
    expected = dict(minimum=min(values), median=statistics.median(values),
                    maximum=max(values), sample_sd=statistics.stdev(values))
    need(all(close(value[k], v) for k, v in expected.items()),
         'aggregate descriptive statistics contradict original block values')
    return dict(n=3, by_block=raw, **expected)


def memory(value):
    need(set(value['by_block']) == set(BLOCKS), 'all PSS coverage outcomes required')
    raw = {b: value['by_block'][b] for b in BLOCKS}
    need(all(v is None or type(v) is int and v >= 0 for v in raw.values()),
         'PSS bytes must remain integers or explicitly unobserved')
    missing = [b for b in BLOCKS if raw[b] is None]
    observed = [v for v in raw.values() if v is not None]
    need(type(value['observed_blocks']) is int and value['observed_blocks'] == len(observed)
         and len(value['unobserved_blocks']) == len(missing)
         and set(value['unobserved_blocks']) == set(missing)
         and value['complete_three_block_range'] is (len(observed) == 3)
         and value['minimum_of_observed'] == (min(observed) if observed else None)
         and value['maximum_of_observed'] == (max(observed) if observed else None),
         'PSS range/coverage contradicts original missing observations')
    return dict(by_block=raw, observed_blocks=len(observed), unobserved_blocks=missing,
                complete_three_block_range=len(observed) == 3,
                minimum_of_observed=min(observed) if observed else None,
                maximum_of_observed=max(observed) if observed else None)


def prepare(data):
    need(data.get('kind') == 'E25_V6_COMPLETE_THREE_ALLOCATION_AGGREGATE'
         and data.get('status') == 'PASS_COMPLETE_THREE_BLOCKS_24_APPS'
         and data.get('global24_complete') is True
         and type(data.get('jobs')) is int and data['jobs'] == 24
         and type(data.get('blocks')) is int and data['blocks'] == 3
         and data['source_hashes'] == PINS and data['auditor_source_sha256'] == AUDITOR_SHA
         and data['prior_failed_attempts_included_in_measurements'] is False
         and data['actual_new_experiments_cluster_GPU_YT_actions'] == 0,
         'only a genuine complete source-bound three-block24 aggregate is admitted')
    for name, cap in (('block_audit_pins', 32 * MIB), ('raw_archive_pins', 128 * MIB),
                      ('root_collection_pins', 32 * MIB)):
        pins = data[name]
        need(set(pins) == set(BLOCKS) and all(sha_hex(v['sha256']) and
             type(v['bytes']) is int and 0 < v['bytes'] <= cap for v in pins.values())
             and len({v['sha256'] for v in pins.values()}) == 3,
             'three distinct original audit/archive/collection pins required')
    platforms = data['platforms']
    need(set(platforms) == set(BLOCKS) and len({p['pod_uid'] for p in platforms.values()}) == 3,
         'three fresh actual Pod allocations required')
    for b in BLOCKS:
        p = platforms[b]
        need(p['kind'] == 'E25_V6_ACTUAL_BLOCK_PLATFORM_ADMISSION' and p['block'] == int(b)
             and len(p['cpu_affinity']) == 27 and p['cpu_affinity'] == sorted(set(p['cpu_affinity']))
             and all(type(cpu) is int for cpu in p['cpu_affinity'])
             and p['cpu_max'] == '2700000 100000' and p['memory_max'] == '274877906944'
             and 'H100' in p['gpu']['name'], 'actual per-block platform contract required')
    need(len({p['image_digest'] for p in platforms.values()}) == 1,
         'same declared software image across blocks required')
    streams = data['source_bound_consumed_input_streams']
    prefixes = data['source_bound_semantic_prefixes_by_horizon']
    need(set(streams) == set(prefixes) == set(HORIZONS) and all(sha_hex(v) for v in streams.values()),
         'both source-bound horizon input identity maps required')
    for h, checkpoints in (('short320', SHORT_POINTS), ('long6400', LONG_POINTS)):
        expected = {str(n) for n, _ in checkpoints} | {'576'}
        need(set(prefixes[h]) == expected and all(sha_hex(v) for v in prefixes[h].values()),
             'complete source-bound consumed checkpoint-prefix map required')
    need(streams['short320'] == prefixes['short320']['576'] == prefixes['long6400']['576']
         and streams['long6400'] == prefixes['long6400']['6656'],
         'global consumed-input and short/long576 identity differs')
    rows = {}
    for row in data['rows']:
        need(row['horizon'] in HORIZONS and type(row['workers']) is int and row['workers'] in (1, 8)
             and row['arm'] in ARMS, 'exact horizon/worker/arm axes required')
        key = row['horizon'], row['workers'], row['arm']
        need(key not in rows, 'duplicate configuration summary')
        points = SHORT_POINTS if row['horizon'] == 'short320' else LONG_POINTS
        expected = BASE_METRICS | {metric for _, metric in points}
        need(set(row['metrics']) == expected, 'exact recorded horizon metric set required')
        metrics = {k: describe(v) for k, v in row['metrics'].items()}
        for b in BLOCKS:
            ordered = [metrics[k]['by_block'][b] for _, k in points]
            need(all(y > x for x, y in zip([0., *ordered[:-1]], ordered)) and
                 metrics['launch_to_actual_wait_seconds']['by_block'][b] >= ordered[-1],
                 'recorded long/short checkpoint and actual-wait ordering differs')
        rows[key] = dict(metrics=metrics, PSS=memory(row['sampled_application_PSS_bytes']))
    expected = {(h, w, a) for h in HORIZONS for w in (1, 8) for a in ARMS}
    need(len(data['rows']) == 8 and set(rows) == expected, 'all eight unique configurations required')
    ratios = {}
    for row in data['paired_compiled_over_native']:
        key = row['horizon'], row['workers']
        need(key not in ratios and row['numerator'] == ARMS[0] and row['denominator'] == ARMS[1],
             'exact within-block ratio axes required')
        left, right = rows[(*key, ARMS[0])]['metrics'], rows[(*key, ARMS[1])]['metrics']
        need(set(row['metrics']) == set(left) == set(right), 'paired metric names differ')
        ratios[key] = {k: describe(v, False) for k, v in row['metrics'].items()}
        for metric, value in ratios[key].items():
            need(all(right[metric]['by_block'][b] > 0 and close(value['by_block'][b],
                 left[metric]['by_block'][b] / right[metric]['by_block'][b]) for b in BLOCKS),
                 'within-block ratios must not be ratios of across-block medians')
    need(len(data['paired_compiled_over_native']) == 4 and set(ratios) == {(h, w) for h in HORIZONS for w in (1, 8)},
         'all four paired configurations required')
    points = []
    for w in (1, 8):
        for a in ARMS:
            for consumed, metric in LONG_POINTS:
                points.append(dict(workers=w, arm=a, horizon='long6400', kind='long_cuda_checkpoint',
                                   consumed_graphs=consumed, metric=metric,
                                   values=rows['long6400', w, a]['metrics'][metric]))
            for h, consumed, kind in (('short320', 576, 'short_actual_wait'),
                                      ('long6400', 6656, 'long_actual_wait')):
                points.append(dict(workers=w, arm=a, horizon=h, kind=kind,
                                   consumed_graphs=consumed, metric='launch_to_actual_wait_seconds',
                                   values=rows[h, w, a]['metrics']['launch_to_actual_wait_seconds']))
    need(len(points) == 28 and not any(p['kind'] == 'long_cuda_checkpoint' and p['consumed_graphs'] == 576 for p in points),
         'only recorded timed long checkpoints and independent actual-wait endpoints may be plotted')
    return rows, ratios, points


def number(value):
    return format(value, '.3g')


def interval(value):
    return f"{number(value['median'])} [{number(value['minimum'])}, {number(value['maximum'])}]"


def pss_cell(value):
    if not value['complete_three_block_range']:
        return r'\textit{unobserved} (' + str(value['observed_blocks']) + '/3)'
    return number(value['minimum_of_observed'] / GIB) + '--' + number(value['maximum_of_observed'] / GIB)


def export_tables(out, identity, data, rows, ratios, points):
    with (out / 'plot_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('workers', 'arm', 'horizon', 'point_kind', 'consumed_graphs', 'metric',
                         'block1_seconds', 'block2_seconds', 'block3_seconds',
                         'median_seconds', 'minimum_seconds', 'maximum_seconds', 'sample_sd_seconds'))
        for p in points:
            v = p['values']
            writer.writerow((p['workers'], p['arm'], p['horizon'], p['kind'], p['consumed_graphs'], p['metric'],
                             *(v['by_block'][b] for b in BLOCKS), v['median'], v['minimum'], v['maximum'], v['sample_sd']))
    with (out / 'configuration_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('horizon', 'workers', 'arm', 'measured_graphs', 'consumed_graphs',
                         'wait_block1_seconds', 'wait_block2_seconds', 'wait_block3_seconds',
                         'wait_median_seconds', 'wait_minimum_seconds', 'wait_maximum_seconds',
                         'PSS_block1_bytes', 'PSS_block2_bytes', 'PSS_block3_bytes', 'PSS_observed_blocks',
                         'PSS_unobserved_blocks', 'PSS_complete_three_block_range',
                         'PSS_minimum_of_observed_bytes', 'PSS_maximum_of_observed_bytes'))
        for h in HORIZONS:
            for w in (1, 8):
                for a in ARMS:
                    r = rows[h, w, a];v = r['metrics']['launch_to_actual_wait_seconds'];m = r['PSS']
                    writer.writerow((h, w, a, *HORIZONS[h], *(v['by_block'][b] for b in BLOCKS),
                                     v['median'], v['minimum'], v['maximum'], *(m['by_block'][b] for b in BLOCKS),
                                     m['observed_blocks'], ';'.join(m['unobserved_blocks']),
                                     m['complete_three_block_range'], m['minimum_of_observed'], m['maximum_of_observed']))
    lines = [r'\begin{tabular}{rrlllll}', r'\toprule',
             r'Measured & $W$ & Compiled (s) & Native (s) & Paired $C/N$ & PSS C (GiB) & PSS N (GiB) \\', r'\midrule']
    for h in HORIZONS:
        for w in (1, 8):
            c, n = rows[h, w, ARMS[0]], rows[h, w, ARMS[1]]
            lines.append(' & '.join((str(HORIZONS[h][0]), str(w), interval(c['metrics']['launch_to_actual_wait_seconds']),
                         interval(n['metrics']['launch_to_actual_wait_seconds']),
                         interval(ratios[h, w]['launch_to_actual_wait_seconds']), pss_cell(c['PSS']), pss_cell(n['PSS']))) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              '% Time/paired-ratio cells: median [min,max] across three blocks; ratios paired before aggregation.',
              '% PSS: range of three observed per-application all-phase maxima; incomplete coverage is unobserved(k/3).',
              '% PSS is sampled/non-atomic/partial-resident; raw observed values and missing block IDs remain in CSV.']
    (out / 'summary_table.tex').write_text('\n'.join(lines) + '\n')
    table = dict(kind='E25_V6_GENUINE_AGGREGATE_PROGRESS_AND_TERMINATED_LIFETIME_TABLEDATA',
                 input_aggregate=identity, source_hashes=PINS, auditor_source_sha256=AUDITOR_SHA,
                 points=points, configurations=[dict(horizon=h, workers=w, arm=a, **v) for (h, w, a), v in rows.items()],
                 paired_ratios=[dict(horizon=h, workers=w, metrics=v) for (h, w), v in ratios.items()],
                 platforms=data['platforms'], block_audit_pins=data['block_audit_pins'],
                 raw_archive_pins=data['raw_archive_pins'], root_collection_pins=data['root_collection_pins'],
                 source_bound_consumed_input_streams=data['source_bound_consumed_input_streams'],
                 source_bound_semantic_prefixes_by_horizon=data['source_bound_semantic_prefixes_by_horizon'],
                 native_scope=data['native_scope'], input_identity_scope=data['input_identity_scope'],
                 scope='Pointwise long-checkpoint summaries, not one observed trajectory; unconnected actual-wait endpoints belong to separately terminated applications. Min/max descriptive, no CI. No long timed576 checkpoint, ratio of medians, crossover fit, time-to-quality or historical ratio. PSS missing coverage retained.')
    (out / 'TABLEDATA.json').write_text(json.dumps(table, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (out / 'caption.txt').write_text(CAPTION + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--aggregate-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True, help='fresh output directory; no overwrite')
    args = parser.parse_args(argv)
    data, identity = read_aggregate(args.aggregate, args.aggregate_sha256)
    rows, ratios, points = prepare(data)
    need(not os.path.lexists(args.out), 'fresh figure directory required')
    # Every whole-SHA/source/cohort/statistics/coverage gate above precedes rendering imports.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.linewidth': .6, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.72, wspace=.13)
    upper = max(p['values']['maximum'] for p in points)
    for ax, w in zip(axes, (1, 8)):
        for index, a in enumerate(ARMS):
            subset = [p for p in points if p['workers'] == w and p['arm'] == a]
            progress = [p for p in subset if p['kind'] == 'long_cuda_checkpoint']
            xs = [p['consumed_graphs'] for p in progress]
            ys = [p['values']['median'] for p in progress]
            errors = [[p['values']['median'] - p['values']['minimum'] for p in progress],
                      [p['values']['maximum'] - p['values']['median'] for p in progress]]
            ax.errorbar(xs, ys, yerr=errors, color=COLORS[a], linestyle='-' if index == 0 else '--',
                        marker='o', markersize=3.4 if index == 0 else 4.3,
                        markerfacecolor=COLORS[a] if index == 0 else 'white',
                        markeredgewidth=.8, linewidth=1.1, elinewidth=.7, capsize=2, zorder=3 + index)
            for kind, marker in (('short_actual_wait', 's'), ('long_actual_wait', 'D')):
                p = next(p for p in subset if p['kind'] == kind);v = p['values']
                ax.errorbar([p['consumed_graphs']], [v['median']],
                            yerr=[[v['median'] - v['minimum']], [v['maximum'] - v['median']]],
                            color=COLORS[a], linestyle='none', marker=marker,
                            markersize=6.0 if index == 0 else 4.7, markerfacecolor='white',
                            markeredgewidth=1., elinewidth=.8, capsize=2.5, zorder=6 + index)
        ax.set_xscale('log', base=2)
        ax.set_xlim(24, 8800)
        ax.set_xticks((32, 256, 576, 1280, 3456, 6656), ('32', '256', '576', '1,280', '3,456', '6,656'))
        ax.minorticks_off()
        ax.set_ylim(0, upper * 1.09)
        ax.set_title(f'{w} spawn worker' + ('s' if w == 8 else ''), fontsize=8, pad=5)
        ax.yaxis.grid(True, color='#DFE5EA', linewidth=.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Time from process launch (s)')
    fig.supxlabel('Consumed graphs, including warmup (log scale)', fontsize=7.5, y=.035)
    path_handles = [Line2D([], [], color=COLORS[a], linestyle='-' if i == 0 else '--', linewidth=1.2,
                          label=LABELS[a]) for i, a in enumerate(ARMS)]
    fig.legend(handles=path_handles, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.53, 1.015))
    role_handles = [Line2D([], [], linestyle='none', color='#374151', marker=marker,
                          markerfacecolor='white', markersize=size, label=label)
                    for marker, size, label in (('o', 3.5, 'Long: synchronized checkpoint'),
                                               ('s', 5, 'Short: actual wait at 576'),
                                               ('D', 5, 'Long: actual wait at 6,656'))]
    fig.legend(handles=role_handles, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.53, .885),
               columnspacing=1., handletextpad=.4)
    args.out.mkdir(parents=True)
    stem = 'e25_v6_progress_and_terminated_lifetimes'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(args.out / f'{stem}.{suffix}', dpi=360 if suffix == 'png' else None)
    plt.close(fig)
    export_tables(args.out, identity, data, rows, ratios, points)
    manifest = dict(kind='E25_V6_ACTUAL_AGGREGATE_FIGURE_RECEIPT', input_aggregate=identity,
        script=pin(Path(__file__)), source_hashes=PINS, auditor_source_sha256=AUDITOR_SHA,
        render=dict(python=platform.python_version(), matplotlib=matplotlib.__version__,
                    width_inches=7.1, height_inches=2.9, PNG_dpi=360),
        coverage=dict(blocks=3, applications=24, configuration_cells=8, plotted_summary_points=28,
                      long_checkpoint_points=20, independent_actual_wait_points=8,
                      raw_checkpoint_or_wait_values_in_CSV=84, PSS_missing_values_preserved=True),
        semantics=dict(pointwise_median_and_descriptive_minmax=True, minmax_is_not_CI=True,
                       median_curve_is_not_one_execution=True, logarithmic_x=True, linear_y_from_zero=True,
                       short_wait_not_connected_to_long_curve=True, timed_long576_invented=False,
                       actual_wait_distinct_from_final_CUDA=True, paired_ratios_before_aggregation=True,
                       no_time_to_quality_historical_ratio_or_crossover_fit=True),
        outputs={p.name: pin(p) for p in sorted(args.out.iterdir())},
        visual_QA='PENDING_ACTUAL_FIGURE_INSPECTION', manuscript_or_publication_changed=False)
    (args.out / 'FIGURE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(output=str(args.out), input_sha256=identity['sha256'],
                          applications=24, plotted_summary_points=28), indent=2))


if __name__ == '__main__':
    main()
