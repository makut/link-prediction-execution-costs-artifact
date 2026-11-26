#!/usr/bin/env python3
"""Plot only a whole-SHA-bound genuine E25 ENVv4 independent audit; no mock-data mode."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

MIB = 1 << 20
PACKET_SHA = '82f93337ba4b4333cc319b610f6bad1ad24c5f9edf1c79731ae6127eed94c005'
PROTOCOL_SHA = 'ff42b44bf17ff54d9994731543fbb3d73caaabb56c7150f576e208ad6c3f1fc2'
AUDITOR_SHA = '44b11d6a684be138487b5b1f5112ad14cbc4ea5809a08821f5061b28f9091393'
ENV_AUDITOR_SHA = '5c1b51876e591f9798551043064fdc1fa892d4bd6705bc9d84c7a0d99ab107fe'
DISK_AUDITOR_SHA = '1112282c70e706864e6d916b03ced0a13d1e7d5e8aee58cb228e22cc32e62233'
STORAGE_SHA = '27b1cd73151fbe27592bdde318532f38593d877477478e2ddf1b066ef7b3b668'
STORAGE_ADMISSION_SHA = 'd3e1d4feb0c519eaf6a7f21d5f9847df5a3523efc8840cd1a3fb00c6b82da026'
CPU_MASK = [*range(26, 39), *range(137, 151)]
ARMS = ('reference_union_compiled', 'reference_union_native_persistent')
HORIZONS = {'short320': (320, 576, 'measured_prefix320'), 'long6400': (6400, 6656, 'measured_prefix6400')}
STAGES = (
    ('launch_to_trained32', 'Launch to trained 32', '#0072B2'),
    ('trained32_to_warmup256', 'Trained 32 to warmup 256', '#56B4E9'),
    ('warmup256_to_final_checkpoint', 'Warmup 256 to final checkpoint', '#009E73'),
    ('final_checkpoint_to_actual_wait', 'Final checkpoint to actual wait', '#E69F00'),
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * MIB,
         'bounded regular input/output file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def decode(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_float=Decimal,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def number(value):
    need(type(value) in (int, Decimal) and Decimal(value).is_finite(), 'finite JSON number required')
    return Decimal(value)


def elapsed_ns(value):
    """Recover original integer elapsed ns from the independently audited seconds."""
    seconds = number(value)
    need(0 <= seconds <= 14400, 'bounded nonnegative elapsed application seconds required')
    exact = seconds * 1_000_000_000
    nanoseconds = int(exact.to_integral_value())
    # At <4h, JSON round-trip of ns/1e9 is much closer than0.01ns to its unique ns.
    need(abs(exact - nanoseconds) <= Decimal('0.01'), 'audit elapsed value cannot identify a unique integer ns')
    return nanoseconds


def summary_ns(values):
    need(len(values) == 3, 'all three process repetitions required')
    mean = Fraction(sum(values), 3)
    variance = sum((Fraction(v) - mean) ** 2 for v in values) / 2
    with localcontext() as context:
        context.prec = 50
        sd = (Decimal(variance.numerator) / variance.denominator).sqrt() / 1_000_000_000
    return mean, float(sd)


def seconds_fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator * 1_000_000_000}


def load_observations(path, expected_sha):
    identity = pin(path)
    need(len(expected_sha) == 64 and identity['sha256'] == expected_sha, 'independently known whole audit SHA differs')
    data = decode(path)
    need(data.get('kind') == 'INDEPENDENT_ACTUAL_E25_ENV_V4_RAW_NUMERICAL_PSS_BRACKETS_AND_DISK_PEER'
         and data.get('status') == 'PASS_ACTUAL_RAW_E25_ENV_V4_AUDIT'
         and data.get('source_manifest_sha256') == PACKET_SHA and data.get('protocol_sha256') == PROTOCOL_SHA
         and data.get('all_app_waits_zero_normal_worker_close') is True
         and data.get('all_initial_models_exact') is True
         and data.get('original_summary_independently_recomputed') is True
         and data['script']['sha256'] == AUDITOR_SHA
         and data['independent_environment_module']['sha256'] == ENV_AUDITOR_SHA
         and data['independent_disk_module']['sha256'] == DISK_AUDITOR_SHA
         and data.get('new_experiments_graph_model_GPU_cluster_network_YT_actions') == 0,
         'genuine complete independently verified E25 audit schema required; source-only inputs refused')
    environment = data['environment_v4']
    selected = environment['chosen_cpu_affinity']
    need(environment['policy'] == 'fixed27_env_v3' and isinstance(selected, list)
         and len(selected) == 27 and all(type(x) is int and x >= 0 for x in selected)
         and selected == sorted(set(selected)) == CPU_MASK and len(environment['timing_jobs']) == 24
         and environment['actual_qualification_same_source_contract_and_inputs'] is True
         and len(environment['actual_H100_qualification_consumers']) == 4
         and data['pre_outcome_headline_PSS_rule_sha256'] == '47b8a0f24c4ec3f386a8b8ba29f6d52a2c3af54ea46b3f2389f6483914bb0a90'
         and environment['prior_aborted']['archive'] == dict(bytes=1168934,
             sha256='34d1dbc0aba48f3ce72b766a655f48ee42b857edf70a7d19415d12aa90490d39'),
         'complete new ENVv4 mask/qualification/PSS-rule and separate prior-diagnostic gates required')
    storage = data['storage_v4']
    disk = storage['raw_disk_and_latest_inventory']
    need(storage['storage_amendment_sha256'] == STORAGE_SHA
         and storage['actual_admission_sha256'] == STORAGE_ADMISSION_SHA
         and storage['controller_guard_failure'] is None
         and storage['outer_guard_proof_independently_recomputed'] is True
         and disk['runtime_thresholds_independently_checked'] is True
         and disk['raw_statvfs_block_arithmetic_independently_recomputed'] is False
         and type(disk['raw_disk_observations_checked']) is int and disk['raw_disk_observations_checked'] > 0
         and disk['device'] == 66323
         and disk['latest_inventory']['complete_entry_observations'] is (disk['latest_inventory']['vanished_entries'] == 0),
         'genuine independent PVC/raw disk/latch/non-atomic inventory audit required')
    raw = data['raw24_metrics']
    need(isinstance(raw, list) and len(raw) == 24, 'all24 raw observations required')
    expected = {(h, w, a, r) for h in HORIZONS for w in (1, 8) for a in ARMS for r in (1, 2, 3)}
    observations, seen = [], set()
    for row in raw:
        job, metrics = row['job'], row['metrics']
        key = job['horizon'], job['workers'], job['arm'], job['repetition']
        need(key in expected and key not in seen, 'exact unique horizon/worker/arm/repetition grid required')
        seen.add(key)
        h, w, arm, rep = key
        need(job['name'] == f'{h}_w{w}_r{rep}_{arm}', 'exact original job identity required')
        checkpoints = [elapsed_ns(metrics[name]) for name in (
            'first_trained_batch_launch_to_complete_seconds',
            'warmup_complete_launch_to_complete_seconds',
            HORIZONS[h][2] + '_launch_to_complete_seconds', 'launch_to_actual_wait_seconds')]
        timeline = [0, *checkpoints]
        need(all(b > a for a, b in zip(timeline, timeline[1:])), 'strictly ordered launch/checkpoint/wait endpoints required')
        intervals = [b - a for a, b in zip(timeline, timeline[1:])]
        need(sum(intervals) == checkpoints[-1], 'four nonoverlapping intervals must exactly sum to actual wait ns')
        observations.append({'job': {k: job[k] for k in ('name', 'horizon', 'workers', 'arm', 'repetition')},
                             'elapsed_endpoints_ns': timeline, 'intervals_ns': intervals, 'application_total_ns': checkpoints[-1]})
    need(seen == expected, 'complete24-observation coverage required')
    need({x['name'] for x in environment['timing_jobs']} == {x['job']['name'] for x in observations},
         'all24 ENVv4 timing proofs must bind the same new-cohort plotted jobs')
    environmental = {x['name']: x for x in environment['timing_jobs']}
    for run in observations:
        proof = environmental[run['job']['name']]
        need(proof['worker_boundary_pairs_checked'] == run['job']['workers'],
             'all actual worker boundary pairs must have been audited')
        for role in ('environment_start', 'environment_end'):
            guard = proof[role]
            need(guard['masks_resources_PID_TID_cgroup_stability_independently_recomputed'] is True
                 and guard['boundary_receipts_checked'] == 1
                 and 1 <= guard['all_boundary_attempts_checked'] <= 8
                 and 1 <= guard['maximum_boundary_attempts'] <= 8
                 and number(guard['maximum_declared_boundary_elapsed_seconds']) <= Decimal('0.5'),
                 'genuine all-attempt full-bracket ENVv4 application guards required')
    summaries = {(r['horizon'], r['workers'], r['arm']): r for r in data['rows']}
    need(len(data['rows']) == len(summaries) == 8, 'all8 unique independent summary groups required')
    groups = []
    for w in (1, 8):
        for h in HORIZONS:
            for arm in ARMS:
                runs = sorted((r for r in observations if (r['job']['horizon'], r['job']['workers'], r['job']['arm']) == (h, w, arm)),
                              key=lambda r: r['job']['repetition'])
                totals = [r['application_total_ns'] for r in runs]
                mean, sd = summary_ns(totals)
                segments = [Fraction(sum(r['intervals_ns'][i] for r in runs), 3) for i in range(4)]
                need(sum(segments) == mean, 'mean stacked intervals must exactly sum to mean actual application wall')
                recorded = summaries[h, w, arm]['metrics']['launch_to_actual_wait_seconds']
                need(all(abs(float(number(a)) - b / 1e9) <= 1e-9 for a, b in zip(recorded['values'], totals))
                     and len(recorded['values']) == 3
                     and math.isclose(float(number(recorded['mean'])), float(mean) / 1e9, rel_tol=1e-12, abs_tol=1e-9)
                     and math.isclose(float(number(recorded['sample_sd'])), sd, rel_tol=1e-10, abs_tol=1e-9),
                     'independent recorded totals/mean/sampleSD differ from raw clock partition')
                groups.append({'horizon': h, 'workers': w, 'arm': arm, 'runs': runs,
                               'mean_total_ns': mean, 'mean_segments_ns': segments, 'sample_sd_seconds': sd})
    return identity, data['archive'], groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='genuine completed independent E25 audit JSON')
    parser.add_argument('--input-sha256', required=True, help='independently known whole audit JSON SHA256')
    parser.add_argument('--out', type=Path, required=True, help='fresh figure directory; no overwrite')
    args = parser.parse_args(argv)
    source = args.input.resolve()
    identity, raw_archive, groups = load_observations(source, args.input_sha256)
    need(not args.out.exists() and not args.out.is_symlink(), 'fresh output directory required')
    # Import/render only after a genuine complete audit has passed admission.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.2,
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.70, wspace=.12)
    maximum = max(max(max(r['application_total_ns'] / 1e9 for r in g['runs']),
                      float(g['mean_total_ns']) / 1e9 + g['sample_sd_seconds']) for g in groups)
    minimum = min(0., min(float(g['mean_total_ns']) / 1e9 - g['sample_sd_seconds'] for g in groups))
    plotted_runs, plotted_bars, tabledata = [], [], []
    for ax, workers in zip(axes, (1, 8)):
        subset = [g for g in groups if g['workers'] == workers]
        for x, group in zip((0., .8, 2., 2.8), subset):
            bottom = 0.
            for (key, label, color), interval in zip(STAGES, group['mean_segments_ns']):
                height = float(interval) / 1e9
                ax.bar(x, height, bottom=bottom, width=.61, color=color,
                       edgecolor='white', linewidth=.3,
                       hatch='//' if group['arm'] == ARMS[1] else None, zorder=2)
                bottom += height
            total = float(group['mean_total_ns']) / 1e9
            need(math.isclose(bottom, total, rel_tol=1e-14, abs_tol=1e-10), 'rendered stack rounding exceeds tolerance')
            ax.errorbar(x, total, yerr=group['sample_sd_seconds'], fmt='D', color='#172B3B',
                        mfc='white', mew=.8, ms=3.3, linewidth=.9, capsize=2.5, zorder=5)
            for run, dx, marker in zip(group['runs'], (-.13, 0., .13), ('o', '^', 's')):
                ax.scatter([x + dx], [run['application_total_ns'] / 1e9], marker=marker,
                           s=17, facecolors='#172B3B', edgecolors='white', linewidths=.35, zorder=6)
                plotted_runs.append(run['job']['name'])
            plotted_bars.append((group['horizon'], workers, group['arm']))
            tabledata.append({'horizon': group['horizon'], 'workers': workers, 'arm': group['arm'],
                'mean_total_seconds_exact_fraction': seconds_fraction(group['mean_total_ns']),
                'sample_sd_seconds': group['sample_sd_seconds'],
                'mean_intervals_seconds_exact_fractions': {stage[0]: seconds_fraction(value)
                    for stage, value in zip(STAGES, group['mean_segments_ns'])},
                'runs': group['runs']})
        ax.set_title(f'{workers} spawn worker' + ('s' if workers == 8 else ''), fontsize=8, pad=5)
        ax.set_xticks((0., .8, 2., 2.8), ['Compiled\n320', 'Native\n320', 'Compiled\n6,400', 'Native\n6,400'])
        ax.set_xlim(-.5, 3.3)
        ax.set_ylim(minimum * 1.08, maximum * 1.12)
        ax.yaxis.grid(True, color='#DFE5EA', linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
    axes[0].set_ylabel('Instrumented application wall (s)')
    fig.supxlabel('Measured graphs; each application also consumes 256 warmup graphs', fontsize=8, y=.035)
    phase_handles = [Patch(facecolor=color, edgecolor='white', label=label) for _, label, color in STAGES]
    fig.legend(handles=phase_handles, ncol=2, frameon=False, loc='upper center',
               bbox_to_anchor=(.53, 1.005), handlelength=1.2, columnspacing=1.3)
    observation_handles = [Line2D([], [], linestyle='none', marker=m, color='#172B3B', markersize=3.5,
                                 label=f'Repetition {r}') for r, m in zip((1, 2, 3), ('o', '^', 's'))]
    observation_handles.append(Line2D([], [], marker='D', color='#172B3B', markerfacecolor='white',
                                      linewidth=.9, markersize=3.3, label='Mean ± sample SD'))
    fig.legend(handles=observation_handles, ncol=4, frameon=False, loc='upper center',
               bbox_to_anchor=(.53, .835), handlelength=1., columnspacing=1.)
    need(len(set(plotted_runs)) == 24 and len(set(plotted_bars)) == 8, 'all24 observations and8 bars must be shown')
    args.out.mkdir(parents=True)
    stem = 'e25_env_v4_actual_lifecycle_partition'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(args.out / f'{stem}.{suffix}', dpi=360 if suffix == 'png' else None)
    plt.close(fig)
    table = {'kind': 'E25_ENV_V4_GENUINE_AUDIT_DERIVED_NONOVERLAPPING_LIFECYCLE_TABLEDATA',
        'input_audit': dict(path=str(source), **identity), 'raw_archive': raw_archive,
        'stage_order': [s[0] for s in STAGES], 'groups': tabledata,
        'all_run_integer_ns_partitions_sum_exactly_to_app_wall': True,
        'all_group_rational_mean_partitions_sum_exactly_to_mean_app_wall': True,
        'platform': {'storage_amendment_sha256': STORAGE_SHA, 'storage_actual_admission_sha256': STORAGE_ADMISSION_SHA,
            'storage': 'Actual write-proven project PVC; storage placement can affect measured lifetime.',
            'cpu_affinity': CPU_MASK, 'observed_physical_cores': 14, 'fresh_complete_24_only': True},
        'language': 'Instrumented application wall includes full consumed-input hashing, resource/ENV/disk observation and metadata fsync, native promotion/fsync on the admitted PVC, and bounded unconsumed prefetch. Warmup256 to final checkpoint includes checkpoint and phase bookkeeping; final checkpoint to actual wait includes shutdown/reporting/teardown. Prefetch work can overlap training and shutdown; no overlapping worker-service, prefetch or promotion timer is stacked. Common whole-input admission audits, C++ build and post-exit native verification are outside applications. Storage placement can affect timings; no production-throughput, time-to-quality, causal cross-attempt/E23 ratio, cold-cache, exclusive-bandwidth or disk-reservation claim is inferred.'}
    (args.out / 'TABLEDATA.json').write_text(json.dumps(table, indent=2, sort_keys=True) + '\n')
    receipt = {'kind': 'E25_ENV_V4_ACTUAL_AUDIT_LIFECYCLE_FIGURE_RECEIPT', 'input_audit': dict(path=str(source), **identity),
        'script': pin(Path(__file__)), 'source_manifest_sha256': PACKET_SHA, 'protocol_sha256': PROTOCOL_SHA,
        'render': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                   'width_inches': 7.1, 'height_inches': 3.3, 'PNG_dpi': 360},
        'coverage': {'bars': 8, 'individual_total_observations': 24, 'omitted_observations': 0},
        'semantics': {'linear_common_y_scale': True, 'all3totals': True, 'sample_SD_not_CI': True,
            'markers_have_fixed_repetition_offsets_in_categorical_x': True, 'interpolation': False,
            'crossover_fit': False, 'worker_service_times_stacked': False, 'promotion_timer_stacked': False,
            'full_consumed_hash_and_resource_ENV_cost_included': True, 'native_promotion_fsync_included': True,
            'disk_observation_and_metadata_fsync_included': True, 'admitted_PVC_placement_disclosed': True,
            'sampled_nonatomic_inventory_not_instantaneous_quota': True,
            'common_input_audits_and_postexit_native_verification_outside_apps': True},
        'outputs': {p.name: pin(p) for p in sorted(args.out.iterdir())},
        'visual_QA': 'PENDING_ACTUAL_FIGURE_INSPECTION', 'current_PDF_repo_or_candidate_modified': False}
    (args.out / 'FIGURE_MANIFEST.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.out), 'bars': 8, 'individual_observations': 24}))


if __name__ == '__main__':
    main()
