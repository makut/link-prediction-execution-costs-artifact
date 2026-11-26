#!/usr/bin/env python3
"""Original environment probe with supplied-interpreter package-root admission.

Only allowed base package roots vary. They come from actual sys.prefix and
sys.base_prefix in the before probe and are reused exactly for later probes.
The original35wheel/source pins,versions,plain .pth,origin andthread gates stay.
"""
from pathlib import Path
import argparse,importlib,sys
sys.dont_write_bytecode=True
from portable_common import *
def set_roots(module,before=None):
    # A supplied venv may itself use a plain fallback into a different local
    # prefix. Discover its actual package sites in the fresh before process,
    # then inherit exactly those observed sites, without guessing /opt or /usr.
    observed=[Path(p).resolve()for p in sys.path if p and Path(p).is_dir()and Path(p).resolve().name in('site-packages','dist-packages')]
    roots=tuple(dict.fromkeys([Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),*observed]))if before is None else tuple(dict.fromkeys([Path(before[k])for k in('prefix','base_prefix')]+[Path(p)for p in before['base_sites']]))
    require(all(p.is_absolute()and p.is_dir()for p in roots),'actual existing interpreter roots required')
    module.BASE_SITE_ROOTS=roots
    return roots

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',choices=('base','isolated'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--before',type=Path);p.add_argument('--venv',type=Path)
    a=p.parse_args(argv);verify_portable_sources();load_originals();env=importlib.import_module('cpu_environment_v1')
    require(Path(env.__file__).resolve()==FROZEN/'cpu_environment_v1.py','original environment source required')
    before=read(a.before)if a.before else None
    require(a.probe=='base'or before is not None,'isolated probe needs genuine fresh before receipt')
    set_roots(env,before)
    value=env.probe(a.probe,a.venv)
    if before is not None:require(value['base_sites']==before['base_sites'],'exact original base package site route required')
    env.fresh(a.output)
    with a.output.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
