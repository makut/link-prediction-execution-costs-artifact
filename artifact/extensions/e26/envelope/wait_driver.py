#!/usr/bin/env python3
"""Detached parent records genuine OS wait for exactly one admitted envelope."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from run_block import cleanup, source_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--wrapper-sha256', required=True)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    a = parser.parse_args(); argv = a.argv[1:] if a.argv[:1] == ['--'] else a.argv
    assert len(argv) > 4 and Path(argv[0]).resolve() == Path(sys.executable).resolve()
    wrapper = Path(argv[3]); assert wrapper.name == 'run_block.py'
    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == a.wrapper_sha256
    assert not a.receipt.exists() and not a.log.exists()
    packet = Path(argv[argv.index('--packet') + 1])
    source_packet(packet, argv[argv.index('--packet-manifest-sha256') + 1])
    spec = importlib.util.spec_from_file_location('closure_driver_support', packet / 'support/owned_supervision.py')
    support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support); support.enable_subreaper()
    launch = {}; result = {}; failure = None
    def popen(*values, **kwargs):
        started = time.monotonic_ns(); child = subprocess.Popen(*values, **kwargs)
        actual = support.process_identity(child.pid)
        assert actual is not None
        launch.update(pid=child.pid, start_ticks=actual['start_ticks'], launch_monotonic_ns=started)
        return child
    try:
        result = support.run_owned(argv, a.log, cwd=wrapper.parent, timeout=4300, popen=popen)
        result.update(actual_wait_returned=True, wait_returned_monotonic_ns=time.monotonic_ns())
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)[:2000]}
        result.update(actual_wait_returned=False, returncode=None)
    finally:
        quiet = cleanup(support)
    value = {'kind': 'CLOSURE_V1_ENVELOPE_ACTUAL_OS_WAIT', **launch, **result,
             'owned_tree_quiescent_after_cleanup': quiet, 'failure': failure, 'wrapper_sha256': a.wrapper_sha256,
             'driver_pid': os.getpid(), 'driver_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'argv': argv, 'pod_uid': os.environ['POD_UID']}
    with a.receipt.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(a.receipt.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)
    return 0 if result.get('returncode') == 0 and result.get('actual_wait_returned') is True and not result.get('timed_out', True) and not result.get('orphan_cleanup', True) and quiet and failure is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
