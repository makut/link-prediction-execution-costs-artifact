#!/usr/bin/env python3
"""Detach the admitted E23 envelope exactly once; root checks live K8s UIDs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import outer_e23 as outer


def main():
    parser = outer.argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    contract = outer.read(outer.HERE / 'CONTRACT.json')
    binding = outer.admission(contract, args.source_manifest_sha256)
    receipt = Path(contract['outer_launch'])
    log = Path(contract['outer_log'])
    outer.require(not os.path.lexists(receipt) and not os.path.lexists(log), 'fresh one-shot outer launcher required')
    command = [contract['python'], '-B', '-u', str(outer.HERE / 'outer_e23.py'),
               '--source-manifest-sha256', args.source_manifest_sha256]
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=outer.HERE, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                     OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1'))
    actual = outer.process(child.pid)
    outer.require(actual is not None, 'detached outer disappeared at launch; preserve log and do not retry')
    outer.save_new(receipt, dict(pid=child.pid, start_ticks=actual['start_ticks'], argv=command,
        entrypoint=str(outer.HERE / 'outer_e23.py'), source_sha256=outer.sha(outer.HERE / 'outer_e23.py'),
        source_manifest_sha256=args.source_manifest_sha256, admission=binding,
        started_unix=time.time(), stdin_dependency=False, detached_session=True,
        log=str(log), output=contract['outer_out'], automatic_retry=False,
        scientific_launch_policy='Only after exact original bootstrap COMPLETE plus actual PID/start quiescence.'))
    print(json.dumps(dict(pid=child.pid, start_ticks=actual['start_ticks'], receipt=str(receipt),
        outer_output=contract['outer_out'], scientific_output=contract['science_out'], log=str(log))))


if __name__ == '__main__':
    main()
