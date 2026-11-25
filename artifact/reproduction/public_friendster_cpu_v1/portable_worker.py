#!/usr/bin/env python3
"""Invoke the unchanged CPU v2 execute() under explicit generic local paths."""
import argparse,importlib,sys
sys.dont_write_bytecode=True
from pathlib import Path
from portable_common import *
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--width',type=int,choices=(32,64),required=True)
    args=parser.parse_args(argv);require(sys.platform.startswith('linux'),'actual worker requires Linux')
    original,L=configure_worker(args.config)
    worker=importlib.import_module('cpu_worker_v1')
    require(Path(worker.__file__).resolve()==FROZEN/'cpu_worker_v1.py','unchanged worker module required')
    return worker.execute(L['base'],args.width,L['out']/('friendster_neighbors'+str(args.width)+'.json'))
if __name__=='__main__':raise SystemExit(main())
