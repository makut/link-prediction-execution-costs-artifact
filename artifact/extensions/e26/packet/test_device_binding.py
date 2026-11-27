"""Bounded real-file regression for cross-host preparation/PVC device identities."""
import json
from pathlib import Path
import tempfile
import unittest

from common import pin, save, sha
from native_storage import reuse_descriptor


class CurrentDeviceBinding(unittest.TestCase):
    def test_historical_device_is_recorded_and_current_device_is_enforced(self):
        with tempfile.TemporaryDirectory(prefix='closure-device-fixture-') as temporary:
            root = Path(temporary).resolve()
            (root / 'rowptr.i8').write_bytes(b'\0' * 32)
            (root / 'col.i8').write_bytes(b'\0' * 24)
            files = {name: pin(root / name) for name in ('rowptr.i8', 'col.i8')}
            expected = {'nodes': 3, 'stored_entries': 3, 'rowptr_bytes': 32, 'col_bytes': 24,
                        'rowptr_sha256': files['rowptr.i8']['sha256'],
                        'col_sha256': files['col.i8']['sha256']}
            actual_device = root.stat().st_dev
            stats = {}
            for name in files:
                p = root / name; p.chmod(0o444); st = p.stat()
                stats[name] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                               'inode': st.st_ino, 'device': actual_device + 1000}
            save(root / 'READY.json', {'kind': 'CLOSURE_V1_IMMUTABLE_NATIVE_REUSE',
                                     'expected_native': expected, 'files': files, 'file_stats': stats})
            (root / 'READY.json').chmod(0o444)
            descriptor = reuse_descriptor(root, expected, sha(root / 'READY.json'),
                                          full=True, expected_device=actual_device)
            self.assertEqual(descriptor['bound_expected_device'], actual_device)
            self.assertEqual(set(descriptor['preparation_file_devices'].values()), {actual_device + 1000})
            self.assertEqual(set(descriptor['actual_file_devices'].values()), {actual_device})
            self.assertEqual(descriptor['verified_files'], files)
            self.assertTrue(descriptor['full_integrity_readback_outside_application'])
            with self.assertRaisesRegex(RuntimeError, 'current admitted PVC device'):
                reuse_descriptor(root, expected, sha(root / 'READY.json'),
                                 full=True, expected_device=actual_device + 1)


if __name__ == '__main__':
    unittest.main()
