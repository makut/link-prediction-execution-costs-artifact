#!/usr/bin/env python3
"""Owned full3 hash/header/stat admission through unchanged generic helper."""
from pathlib import Path
import argparse,sys
sys.dont_write_bytecode=True
from portable_common import *
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=('before','after'),required=True)
    a=p.parse_args(argv);original,L=configure_worker(a.config)
    original.save(L['out']/('full_input_'+a.phase+'.json'),original.admit_original(L['full']))
    return 0
if __name__=='__main__':raise SystemExit(main())
