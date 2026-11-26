#!/usr/bin/env python3
"""Bounded streaming full public graph admission in an owned subprocess."""
from cpu_common_v1 import *
import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--phase', choices=('before', 'after'), required=True); a = p.parse_args(argv)
    require(a.base.resolve() == BASE and sys.platform.startswith('linux'), 'canonical CPU base required')
    L = layout(a.base); output = L['out']/('full_input_'+a.phase+'.json')
    require(not output.exists(), 'fresh full-input admission receipt required')
    value = admit_original(L['full']); save(output, value)
    print(json.dumps({'status': 'complete', 'files': len(value['files']), 'whole_source_sha': True})); return 0


if __name__ == '__main__': raise SystemExit(main())
