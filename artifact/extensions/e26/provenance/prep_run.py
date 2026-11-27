#!/usr/bin/env python3
"""Actual CPU-only preparation and owned-child wait, one invocation per dataset."""
from pathlib import Path
import hashlib, importlib.util, json, os, socket, subprocess, sys, time

BASE = Path('/home/jovyan/seal_native_lifecycle_20261126/closure_controls_20261127')
PACKET_SHA = '0bb01dfd0524340e5f3f9f08ab1e0f2fc604e47a726db498f59ee1b376149bb6'


def main():
    dataset = sys.argv[1]; assert dataset in ('friendster', 'citation2')
    packet = BASE / 'source/packet_v1'
    assert hashlib.sha256((packet / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest() == PACKET_SHA
    sys.path.insert(0, str(packet)); from common import verify_sources, save, pin, require
    verify_sources()
    spec = importlib.util.spec_from_file_location('closure_prep_support', packet / 'support/owned_supervision.py')
    support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support); support.enable_subreaper()
    graph = BASE.parent / 'data/friendster/full_csr' if dataset == 'friendster' else BASE / 'data/citation2/full_csr'
    out = BASE / 'native_prepared' / dataset
    receipt = BASE / 'ops' / ('prepare_' + dataset + '_actual_exit.json')
    assert not out.exists() and not receipt.exists()
    command = [sys.executable, '-B', '-u', str(packet / 'prepare_native.py'), '--dataset', dataset,
               '--graph-root', str(graph), '--out', str(out), '--max-seconds', '2400']
    launch = {}
    def popen(*values, **kwargs):
        began = time.monotonic_ns(); child = subprocess.Popen(*values, **kwargs)
        ident = support.process_identity(child.pid); assert ident is not None
        launch.update(pid=child.pid, start_ticks=ident['start_ticks'], launch_monotonic_ns=began, argv=command)
        save(BASE / 'ops' / ('prepare_' + dataset + '_actual_launch.json'), launch); return child
    result = support.run_owned(command, BASE / 'ops' / ('prepare_' + dataset + '.log'),
                               cwd=packet, timeout=2500, popen=popen)
    result.update(actual_wait_returned=True, wait_returned_monotonic_ns=time.monotonic_ns())
    ready = pin(out / 'READY.json') if result['returncode'] == 0 else None
    value = {'kind': 'CLOSURE_NATIVE_PREPARATION_ACTUAL_OS_WAIT', 'dataset_id': dataset,
             **launch, **result, 'source_manifest_sha256': PACKET_SHA, 'ready_receipt': ready,
             'prep_driver_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'pod_uid': os.environ['POD_UID'], 'hostname': socket.gethostname(),
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'observed_unix': time.time(), 'outside_all_scientific_application_clocks': True,
             'GPU': 0, 'YT_writes': 0}
    save(receipt, value)
    require(result['returncode'] == 0 and not result['timed_out'] and not result['orphan_cleanup']
            and result['owned_tree_quiescent'] and result['owned_pgids_signaled'] == [], 'CPU preparation failed')
    print(json.dumps({'dataset_id': dataset, 'status': 'PREPARATION_WAIT0', 'ready': ready}), flush=True)


if __name__ == '__main__':
    main()
