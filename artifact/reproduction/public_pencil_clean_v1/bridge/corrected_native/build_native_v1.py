#!/usr/bin/env python3
"""Build only this versioned standalone control; never install a package."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def build(output_dir,compiler=None):
    output_dir=Path(output_dir).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    lib=output_dir/('native_assoc_v1.dylib' if sys.platform=='darwin' else 'native_assoc_v1.so')
    receipt=output_dir/'build_receipt_v1.json'
    if lib.exists() or receipt.exists():raise FileExistsError('existing versioned native build; use a fresh directory')
    compiler=compiler or os.environ.get('CXX') or shutil.which('c++')
    if not compiler:raise RuntimeError('C++17 compiler not found')
    compiler=str(Path(compiler).resolve())
    source=HERE/'native_assoc_v1.cpp';partial=lib.with_suffix(lib.suffix+'.building')
    if partial.exists():raise FileExistsError('preserve existing partial build')
    flags=['-std=c++17','-O3','-DNDEBUG','-fPIC','-pthread',
           '-dynamiclib' if sys.platform=='darwin' else '-shared']
    argv=[compiler,*flags,str(source),'-o',str(partial)]
    version=subprocess.run([compiler,'--version'],text=True,capture_output=True,check=True,timeout=15).stdout[:4096]
    result=subprocess.run(argv,text=True,capture_output=True,timeout=120)
    if result.returncode:raise RuntimeError('C++ build failed: '+result.stderr[-8000:])
    partial.replace(lib)
    value={'schema_version':1,'kind':'corrected_native_build_v1','source':str(source),
           'source_sha256':sha(source),'compiler':compiler,'compiler_version':version,
           'argv':argv,'platform':platform.platform(),'library':str(lib),'library_sha256':sha(lib),
           'scope':'Standalone compiled correctness/control library; not an upstream Torch extension or a performance result.'}
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    return lib,value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--compiler')
    a=p.parse_args();lib,meta=build(a.output_dir,a.compiler);print(json.dumps(meta))
