"""Run the48 frozen public Citation2 PENCIL systems jobs, without training/HPO.

Plan by default. Local setup regenerates the exact validation subset and builds
the C++ control. No historical pod, storage service, wheelhouse or queue is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from . import systems_common as c
from .pencil_validation import ARMS, REPLAY_ARMS, validate_job, crossjob_scores

BRIDGE = c.ROOT / 'work/vldb_pencil_bridge'


def make_plan(args):
    out = args.out
    script = BRIDGE / 'pencil_bridge.py'
    subset = args.validation_root or (out / 'validation')
    build = out / 'cpp_build'
    real = ['--graph-path', str(args.citation2_root / 'train_csr'),
        '--features', str(args.citation2_root / 'node_feat.npy'),
        '--valid-positive', str(subset / 'valid_edge.npy'), '--valid-negative', str(subset / 'valid_edge_neg.npy'),
        '--strict-manifest', str(args.citation2_root / 'manifest.json'),
        '--quality-plan', str(c.ROOT / 'metadata/protocols/citation2_quality_original_plan.json')]
    cpp = ['--cpp-module', str(BRIDGE / 'corrected_native/native_adapter_v1.py'),
           '--cpp-library', str(build / 'native_assoc_v1.so'), '--cpp-build-receipt', str(build / 'build_receipt_v1.json')]
    jobs = []
    def add(name, kind, tail, *, gpu=False, **metadata):
        output = out / (name + '.json')
        jobs.append({'name': name, 'kind': kind, 'argv': [sys.executable, '-u', str(script), *tail, '--output', str(output)],
                     'output': str(output), 'gpu': gpu, 'timeout_seconds': 1800, **metadata})
    tiny, trace = out / 'tiny_trace', out / 'real_trace'
    add('tiny_capture', 'capture', ['capture', '--fixture', '--execute', '--trace-dir', str(tiny)])
    add('tiny_audit', 'audit', ['audit', '--fixture', '--execute', '--trace-dir', str(tiny), '--arms', ','.join(ARMS), *cpp,
                             '--forward', '--tiny-model', '--training-wrapper', '--batch-size', '128'])
    add('real_preflight', 'preflight', ['preflight', *real])
    add('real_capture', 'capture', ['capture', '--execute', *real, '--trace-dir', str(trace)])
    for precision in ('fp32', 'bf16'):
        for batch in (128, 512):
            for workers in (0, 1):
                add(f'audit_{precision}_b{batch}_w{workers}', 'audit',
                    ['audit', '--execute', *real, '--trace-dir', str(trace), '--arms', ','.join(ARMS), *cpp,
                     '--forward', '--training-wrapper', '--device', 'cuda', '--precision', precision,
                     '--batch-size', str(batch), '--num-workers', str(workers), '--max-logit-abs', '0'],
                    gpu=True, precision=precision, batch=batch, workers=workers)
    for mode in ('online', 'replay'):
        configurations = [(b, w, ARMS) for b in (128, 512) for w in (0, 1)] if mode == 'online' else [(128, 0, REPLAY_ARMS)]
        for batch, workers, arms in configurations:
            for arm in arms:
                add(f'bench_{mode}_bf16_b{batch}_w{workers}_{arm}', 'benchmark',
                    ['benchmark', '--execute', *real, '--trace-dir', str(trace), *cpp,
                     '--audit-receipt', str(out / f'audit_bf16_b{batch}_w{workers}.json'), '--arm', arm,
                     '--pipeline-mode', mode, '--forward', '--device', 'cuda', '--precision', 'bf16',
                     '--batch-size', str(batch), '--num-workers', str(workers), '--max-logit-abs', '0',
                     '--warmup', '1', '--repeats', '3'],
                    gpu=True, mode=mode, precision='bf16', batch=batch, workers=workers, arm=arm)
    c.require(len(jobs) == 48 and sum(j['kind'] == 'benchmark' for j in jobs) == 36, 'exact48job matrix required')
    setup = []
    if not args.validation_root:
        setup.append({'name': 'prepare_public_validation', 'gpu': False, 'timeout_seconds': 1800,
          'argv': [sys.executable, '-u', '-m', 'artifact_public.prepare_pencil_validation',
                   '--citation2-root', str(args.citation2_root), '--out', str(subset), '--execute']})
    setup.append({'name': 'build_public_cpp', 'gpu': False, 'timeout_seconds': 180,
                  'argv': [sys.executable, str(BRIDGE / 'corrected_native/build_native_v1.py'), '--output-dir', str(build)]})
    return {'schema_version': 1, 'scope': '48 public Citation2 native-consumer/model systems jobs; random frozen model, no training/HPO/MRR',
            'execute': args.execute, 'input_identity_checked': False,
            'coordinator_limits': c.coordinator_limits(args),
            'validation_root': str(subset), 'setup_jobs': setup, 'jobs': jobs,
            'scientific_job_count': 48, 'timing_jobs': 36, 'semantic_audits': 9,
            'real_cohort': '512 fixed validation positives plus original candidate0 per1000-negative group, seed20261124; extractionseed20261126',
            'model': 'unmodified pinned BERTLP3x512/8heads/FFN2048/features128early, random frozen seed42',
            'frozen_sources': c.frozen_sources()}


def verify_validation(root):
    contract = c.read(BRIDGE / 'input_contract.json')['payloads']
    for name, key in (('valid_edge.npy', 'valid_positive'), ('valid_edge_neg.npy', 'valid_negative')):
        path = Path(root) / name
        c.require(path.is_file() and not path.is_symlink() and c.sha(path) == contract[key]['sha256'],
                  'regenerated validation payload differs from the exact original512-source subset')


def verify_public_inputs(root):
    from .verify_payloads import verify
    expected = c.read(c.ROOT / 'metadata/citation2_prepared_expected.json')
    names = {'train_csr/edge_starts.bin', 'train_csr/edge_ends.bin', 'node_feat.npy', 'valid_edge.npy', 'valid_edge_neg.npy'}
    entries = [entry for entry in expected['files'] if entry['path'] in names]
    c.require(len(entries) == 5, 'five public graph/feature/validation inputs required')
    return verify(root, {'files': entries, 'scope': 'only graph/features/validation; test payloads are not opened'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    c.common_arguments(parser)
    parser.add_argument('--citation2-root', type=Path, required=True)
    parser.add_argument('--validation-root', type=Path, help='Optional existing exact512-source subset; default regenerates it without fitting')
    args = parser.parse_args(argv)
    args.citation2_root = args.citation2_root.resolve()
    if args.validation_root:
        args.validation_root = args.validation_root.resolve()
    inputs = (args.citation2_root,) + ((args.validation_root,) if args.validation_root else ())
    c.validate_arguments(args, inputs)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    with c.Execution(args, plan, inputs=inputs, gpu=True) as execution:
        c.save(args.out / 'input_identity.json', verify_public_inputs(args.citation2_root))
        for job in plan['setup_jobs']:
            c.require(c.clean_exit(execution.run(job)), 'local subset or compiler setup failed')
        verify_validation(plan['validation_root'])
        for job in plan['jobs']:
            validate_job(job, execution.run(job))
        scores = crossjob_scores(plan['jobs'])
        c.save(args.out / 'scientific_summary.json', {'status': 'complete', 'scientific_jobs': 48,
             'setup_jobs': len(plan['setup_jobs']), 'crossjob_model_scores': scores, 'scope': plan['scope']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
