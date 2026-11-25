"""Replay128 unchanged public Friendster sets through three native controls and
two same-host fixed-mode Numba width workers.

Plan by default. The explicit local admission records this coordinator's own
completed idle/lock preflight; it does not invent a historical quality result.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

from . import systems_common as c

NATIVE = c.ROOT / 'work/vldb_corrected_native'
HELPER = c.ROOT / 'work/vldb_literature/external_operator_benchmark.py'
HELPER_SHA = '525e88c904fd03e10fc617853c8d733e88af90487cbbdd9122ced97962df112c'


def make_plan(args):
    parent = args.out / 'replay'
    command = [sys.executable, '-u', str(NATIVE / 'run_parent_replay_v1.py'),
        '--archive', str(c.ROOT / 'data/friendster_exact_int32_reference.json'), '--count', '128',
        '--output-dir', str(parent), '--execute', '--build-dir', str(args.out / 'build'),
        '--native-dir', str(args.native_dir or (args.out / 'native_graph')),
        '--admission-marker', str(args.out / 'admission.json'), '--include-upstream',
        '--external-helper', str(HELPER), '--external-helper-sha256', HELPER_SHA,
        '--conversion-max-native-gib', '28', '--max-assoc-gib', '0.5', '--max-peak-gib', '48',
        '--max-query-neighbors', '20000000', '--warmup', '1', '--repeats', '3',
        '--timeout-seconds', '3600', '--conversion-timeout-seconds', '1800', '--order-seed', '20261125']
    if not args.native_dir:
        command += ['--prepare-native', '--graph-path', str(args.friendster_root)]
    jobs = [{'name': 'native_friendster128', 'argv': command, 'timeout_seconds': 14400, 'gpu': False}]
    for width in (32, 64):
        followup = [sys.executable, '-u', '-m', 'artifact_public.samehost_friendster_worker',
            '--friendster-root', str(args.friendster_root),
            '--native-dir', str(args.native_dir or (args.out / 'native_graph')),
            '--replay-dir', str(parent), '--expected-host', str(args.out / 'host.json'),
            '--width', str(width), '--output', str(args.out / f'numba_friendster{width}.json'), '--execute']
        if getattr(args, 'reference_hardware', False):
            followup.append('--reference-hardware')
        jobs.append({'name': f'numba_friendster{width}', 'argv': followup, 'timeout_seconds': 3600, 'gpu': False})
    return {'schema_version': 1, 'scope': '128 frozen public Friendster cases; matched3native arms plus2same-host Numba widths, no model/quality claim',
            'execute': args.execute, 'count': 128, 'warmup': 1, 'repeats': 3, 'cpu_threads': 1,
            'coordinator_limits': c.coordinator_limits(args),
            'input_identity_checked': False, 'persistent_map_bytes_per_worker': 65608366 * 8,
            'native_graph_directory': str(args.native_dir or (args.out / 'native_graph')),
            'same_host_required': True, 'scientific_native_arms': 3, 'scientific_numba_width_workers': 2,
            'reference_hardware_required': getattr(args, 'reference_hardware', False),
            'jobs': jobs,
            'frozen_sources': c.frozen_sources()}


def validate_times(result, *, raw=False):
    c.require(result.get('status') == 'ok' and result.get('exact_edges') is True
              and result.get('exact_drnl') is True, 'every measured output must be exact')
    for key in ('edge_seconds', 'drnl_seconds') + (('raw_operator_seconds',) if raw else ()):
        values = result.get(key)
        c.require(isinstance(values, list) and len(values) == 3
                  and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values),
                  'all3 finite recorded repeats required')
    if raw:
        c.require(result.get('exact_ordered_rows_cols_and_original_eids') is True
                  and result.get('root_positions') == [0, 1], 'native ordered output/eid/root exactness required')
        hashes = result.get('ordered_output_sha256')
        c.require(isinstance(hashes, list) and len(hashes) == 3 and all(isinstance(v, str)
            and len(v) == 64 and all(char in '0123456789abcdef' for char in v) for v in hashes),
            'all3 native ordered output hashes required')
        c.require(all(a <= b + 1e-12 for a, b in zip(result['raw_operator_seconds'], result['edge_seconds'])),
                  'raw native timer must be nested inside packaged edge timer')


def validate_result(out, exit_data):
    c.require(c.clean_exit(exit_data), 'native parent failed, guarded, timed out or left an owned process')
    folder = Path(out) / 'replay'
    terminal = c.read(folder / 'COMPLETE.json')
    c.require(terminal.get('complete') is True and terminal.get('success') is True
              and terminal.get('scientific_terminal') is True, 'native replay must be scientifically successful')
    arms = terminal['arms']
    expected = {'reference_full_clear', 'persistent_touched_reset', 'upstream_torch_sparse_direct'}
    c.require(len(arms) == 3 and {a['mode'] for a in arms} == expected, 'all3native arms required')
    plan_path = folder / 'replay_plan.json'
    replay_plan = c.read(plan_path)
    c.require(replay_plan['count'] == 128 and terminal.get('replay_plan_sha256') == c.sha(plan_path), 'native selection binding differs')
    from .samehost_friendster_worker import scientific_functions
    scientific_functions().verify_same_selection(replay_plan, NATIVE / 'replay_plans/friendster_128_v1.json')
    selected = [(r['archive_record_index'], r['node_set_sha256']) for r in replay_plan['selected']]
    c.require(len(selected) == 128 and len(set(selected)) == 128, 'all128 distinct frozen cases required')
    for arm in arms:
        c.require(arm.get('eligible') is True and type(arm['exit']['returncode']) is int
                  and arm['exit']['returncode'] == 0 and arm['exit']['timed_out'] is False,
                  'native arm failure cannot count as completed science')
        path = folder / (arm['mode'] + '.json')
        report = c.read(path)
        c.require(c.sha(path) == arm['report_sha256'] and report.get('status') == 'complete'
                  and report.get('record_count') == 128 and report.get('all_executed_outputs_exact') is True,
                  'all128 frozen outputs and report hashes must match')
        c.require(report.get('replay_plan_sha256') == c.sha(plan_path)
                  and [(r['archive_record_index'], r['node_set_sha256']) for r in report['records']] == selected,
                  'native arm must execute exact selected order')
        for row in report['records']:
            validate_times(row, raw=True)
    return terminal


def validate_followup(out, width, exit_data):
    from . import samehost_friendster_worker as samehost
    c.require(c.clean_exit(exit_data), 'same-host worker failed, guarded, timed out or left an owned process')
    folder = Path(out)
    report = c.read(folder / f'numba_friendster{width}.json')
    c.require(report.get('status') == 'complete' and report.get('record_count') == 128
              and report.get('all_executed_outputs_exact') is True and report.get('neighbor_width') == width,
              'both128case same-host workers must complete exactly; skips are not success')
    samehost.verify_host(c.read(folder / 'host.json'), report['host'])
    c.require(report.get('graph') == 'friendster' and report.get('numba_threads') == 1, 'public Friendster one-thread worker required')
    plan = c.read(folder / 'replay/replay_plan.json')
    scientific = samehost.scientific_functions()
    scientific.verify_same_selection(report['replay_plan'], folder / 'replay/replay_plan.json')
    c.require(report.get('replay_plan_sha256') == c.sha(folder / 'replay/replay_plan.json'), 'same-host selection receipt changed')
    physical = report['physical_arrays']
    c.require(physical['offset_dtype'] == 'int64' and physical['neighbor_dtype'] == f'int{width}'
              and physical['offset_count'] == plan['graph']['nodes'] and physical['neighbor_count'] == plan['graph']['stored_entries']
              and physical['full_graph_copy_or_conversion_performed'] is False, 'physical arrays must match unchanged original/conversion')
    selected = plan['selected']
    c.require(len(report['records']) == 128, '128actual records required')
    for actual, frozen in zip(report['records'], selected):
        c.require(all(actual.get(key) == frozen[key] for key in
            ('archive_record_index', 'node_set_sha256', 'canonical_edges_sha256', 'drnl_sha256', 'sum_out_degrees', 'induced_directed_entries'))
            and actual.get('exact_archived_oracles') is True and set(actual['methods']) == set(samehost.METHODS),
            'same-host exact selection/oracle/method identity differs')
        for result in actual['methods'].values():
            validate_times(result)
    return {'width': width, 'report_sha256': c.sha(folder / f'numba_friendster{width}.json'),
            'record_count': 128, 'all_executed_outputs_exact': True, 'host': report['host']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    c.common_arguments(parser, memory=64, output=32, hours=6)
    parser.add_argument('--friendster-root', type=Path, required=True, help='public full CSR')
    parser.add_argument('--native-dir', type=Path, help='Optional existing hash-bound int64 conversion; never overwritten')
    parser.add_argument('--reference-hardware', action='store_true', help='Require the published8480+ CPU model; default documents any one Linux host')
    args = parser.parse_args(argv)
    args.friendster_root = args.friendster_root.resolve()
    if args.native_dir:
        args.native_dir = args.native_dir.resolve()
    inputs = (args.friendster_root,) + ((args.native_dir,) if args.native_dir else ())
    c.validate_arguments(args, inputs)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    with c.Execution(args, plan, inputs=inputs, min_disk_gib=4 if args.native_dir else 32) as execution:
        from . import samehost_friendster_worker as samehost
        samehost.verify_host(c.read(args.out / 'host.json'), c.host_identity(), args.reference_hardware)
        c.save(args.out / 'input_identity.json', c.verify_inputs(args.friendster_root, 'metadata/friendster_full_expected.json'))
        c.save(args.out / 'admission.json', {'kind': 'public_exclusive_timing_admission_preflight',
            'controller_finished': True, 'complete': True, 'success': True,
            'coordination_lock_held': True, 'no_prior_quality_or_pipeline_result_is_attested': True,
            'scope': 'Current public coordinator preflight completed; user schedules this CPU benchmark in an idle timing window.',
            'completed_unix': time.time()})
        terminal = validate_result(args.out, execution.run(plan['jobs'][0]))
        samehost.verify_host(c.read(args.out / 'host.json'), c.host_identity(), args.reference_hardware)
        followups = [validate_followup(args.out, width, execution.run(job))
                     for width, job in zip((32, 64), plan['jobs'][1:])]
        samehost.verify_host(c.read(args.out / 'host.json'), c.host_identity(), args.reference_hardware)
        c.save(args.out / 'scientific_summary.json', {'status': 'complete', 'count': 128,
            'native_arms': terminal['arms'], 'samehost_numba_workers': followups, 'scope': plan['scope']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
