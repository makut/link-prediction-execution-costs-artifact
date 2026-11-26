"""Bounded local envelope QA only; no science, graph, GPU, network or YT write."""
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import outer_e25 as outer


class FakeSupport:
    @staticmethod
    def run_owned(argv, log, *, cwd, timeout, popen):
        with log.open('x') as stream:
            child = popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
            code = child.wait(timeout=timeout)
        return dict(pid=child.pid, returncode=code, timed_out=False,
                    orphan_cleanup=False, owned_tree_quiescent=True, owned_pgids_signaled=[])


class FakeClient:
    def __init__(self):
        self.payload, self.calls, self.committed = None, [], False

    def exists(self, path):
        return path == outer.YT_ROOT

    def Transaction(self, timeout):
        self.calls.append(('transaction', timeout))
        return self

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        self.committed = kind is None

    def create(self, kind, path, recursive):
        self.calls.append(('create', kind, path, recursive))

    def write_file(self, path, stream, file_writer):
        self.calls.append(('write', path, file_writer))
        self.payload = stream.read()

    def get(self, path):
        return 1 if path.endswith('/@chunk_count') else len(self.payload)

    def read_file(self, path):
        return io.BytesIO(self.payload)


class WrapperLifecycle(unittest.TestCase):
    def test_bounded_real_fake_child_and_archive_and_fake_transaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, evidence = root / 'packet', root / 'evidence'
            packet.mkdir()
            evidence.mkdir()
            source = packet / 'run_lifecycle.py'
            source.write_text('print("local supervisor fixture only")\n')
            contract = dict(packet=str(packet), packet_manifest_sha256='local-fake-pin', expected_cpu_affinity=list(range(197, 224)))
            launch, result = outer.supervise_child(contract, FakeSupport(), 5, evidence,
                argv=[sys.executable, '-B', str(source)],
                reader=lambda pid: dict(pid=pid, start_ticks=123456, state='R', argv=[str(source)]))
            outer.successful_exit(result)
            self.assertEqual(result['pid'], launch['pid'])
            self.assertEqual(outer.read(evidence / 'child_launch.json')['start_ticks'], 123456)
            self.assertEqual(outer.quiet(launch, reader=lambda pid: None)['observation'], 'absent')
            zombie = lambda pid: dict(start_ticks=123456, state='Z', kernel_exit_code=0)
            self.assertEqual(outer.quiet(launch, reader=zombie)['kernel_exit_code'], 0)
            with self.assertRaisesRegex(RuntimeError, 'exit_code0'):
                outer.quiet(launch, reader=lambda pid: dict(start_ticks=123456, state='Z', kernel_exit_code=256))
            with self.assertRaisesRegex(RuntimeError, 'PID reused'):
                outer.quiet(launch, reader=lambda pid: dict(start_ticks=999, state='R'))
            with self.assertRaisesRegex(RuntimeError, 'actual wait0'):
                outer.successful_exit(dict(result, returncode=1))
            # Model Linux comm containing spaces and ')' and assert field52, not field51.
            proc = root / 'proc'
            (proc / '75').mkdir(parents=True)
            fields = ['0'] * 50
            fields[0], fields[19], fields[49] = 'Z', '537191365', '256'
            (proc / '75/stat').write_text('75 (test name) with parens)) ' + ' '.join(fields))
            (proc / '75/cmdline').write_bytes(b'')
            observed = outer.process(75, proc)
            self.assertEqual(observed['start_ticks'], 537191365)
            self.assertEqual(observed['kernel_exit_code'], 256)
            # Tiny source-independent NPY structural gate, no NumPy import.
            pairs = root / 'pairs.npy'
            header = repr(dict(descr='<i8', fortran_order=False, shape=(128, 3))).encode() + b'\n'
            pairs.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + bytes(128 * 3 * 8))
            pairs.with_name('pairs.npy.json').write_text(json.dumps(dict(graph=dict(path='fixture'), n_positives=64)))
            self.assertEqual(outer.tiny_pair_identity(pairs)['bytes'], pairs.stat().st_size)
            archive = root / 'fake-only.tgz'
            files = {'e25/envelope/' + p.name: p for p in evidence.iterdir()}
            files['e25/results/' + 'safe_long_path_' * 15 + '/report.json'] = evidence / 'child_exit.json'
            receipt = outer.archive_results(archive, files)
            self.assertTrue(receipt['all_whole_member_hashes_verified'])
            client = FakeClient()
            pointer = outer.upload_once(archive, receipt, outer.YT_FINAL, client=client)
            self.assertTrue(pointer['readback_verified'] and client.committed)
            self.assertEqual(len([c for c in client.calls if c[0] == 'create']), 1)
            self.assertEqual(len([c for c in client.calls if c[0] == 'write']), 1)
            with self.assertRaisesRegex(RuntimeError, 'only pinned'):
                outer.upload_once(archive, receipt, outer.YT_FINAL + '-other', client=client)
            with self.assertRaisesRegex(RuntimeError, 'fresh bounded archive'):
                outer.archive_results(archive, files)


if __name__ == '__main__':
    unittest.main()
