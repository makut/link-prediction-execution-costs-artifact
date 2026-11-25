#!/usr/bin/env python3
"""Bounded environment provenance only; never loads data or modifies packages."""
import argparse,hashlib,importlib,importlib.metadata as md,json,platform,site,subprocess,sys
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def probe(mode):
    import numpy,torch,torch_sparse
    packages={}
    for d in md.distributions():
        if d.metadata.get('Name'):packages.setdefault(d.metadata['Name'].lower().replace('_','-'),d.version)
    root=Path(torch_sparse.__file__).parent
    result={'schema_version':1,'mode':mode,'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,
        'base_prefix':sys.base_prefix,'platform':platform.platform(),'packages':dict(sorted(packages.items())),
        'site_packages':[str(Path(p).resolve())for p in site.getsitepackages()if Path(p).is_dir()],
        'critical_versions':{name:md.version(name)for name in('torch','numpy','scipy','numba','torch-sparse','torch-geometric')},
        'critical_modules':{name:{'origin':str(Path(importlib.import_module(name).__file__).resolve()),
            'source_sha256':digest(Path(importlib.import_module(name).__file__))}
            for name in('torch','numpy','scipy','numba','torch_sparse','torch_geometric')},
        'torch':torch.__version__,'torch_cuda':torch.version.cuda,'numpy':numpy.__version__,
        'torch_sparse_version':md.version('torch-sparse'),'pyg':md.version('torch-geometric'),
        'torch_sparse_binaries':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.iterdir())
            if p.is_file() and p.suffix in('.so','.pyd','.dylib')},
        'saint_wrapper':{'sha256':digest(root/'saint.py'),
            'git_blob':hashlib.sha1(b'blob '+str((root/'saint.py').stat().st_size).encode()+b'\0'+(root/'saint.py').read_bytes()).hexdigest()},
        'pip_version':subprocess.check_output([sys.executable,'-m','pip','--version'],text=True,timeout=30).strip()}
    assert len(json.dumps(result))<2<<20,'bounded package provenance'
    if mode=='isolated':
        assert sys.version_info[:2]==(3,10),'declared Python3.10 target'
        assert torch.__version__=='2.5.0+cu121' and numpy.__version__=='1.24.4','base Torch/CUDA/NumPy changed'
        assert md.version('torch-sparse').split('+')[0]=='0.6.18' and md.version('torch-geometric')=='2.7.0','native target changed'
        assert md.version('transformers')=='4.46.2' and md.version('wandb')=='0.18.7','extra versions changed'
        assert result['saint_wrapper']['git_blob']=='440455853e0309b00376cbe9ad25d7226cfa5e0e','native wrapper source changed'
        assert sys.prefix!=sys.base_prefix,'isolated venv required'
        assert Path(sys.prefix).name=='pencil_env_v3','fresh version2 environment required'
        assert 'include-system-site-packages = true' in(Path(sys.prefix)/'pyvenv.cfg').read_text().lower(),'base stack sharing required'
        route=Path(sys.prefix)/'lib/python3.10/site-packages/pencil_base_site_v3.pth'
        text=route.read_text();paths=text.splitlines()
        assert paths and all(Path(p).is_absolute()and Path(p).is_dir()and '\x00'not in p and
            Path(p).name in('site-packages','dist-packages')and
            (Path(p).is_relative_to('/opt/python')or Path(p).is_relative_to('/usr'))for p in paths),'plain existing base site paths required'
        order=[str(Path(p).resolve())for p in sys.path if p]
        own=str((Path(sys.prefix)/'lib/python3.10/site-packages').resolve())
        assert all(own in order and p in order and order.index(own)<order.index(p)for p in paths),'isolated packages must precede base fallback paths'
        result['base_site_route']={'path':str(route),'sha256':digest(route),'content':text,'base_paths':paths,
            'isolated_site_precedes_all_base_paths':True}
        result['extra_modules']={}
        for name,version in(('transformers','4.46.2'),('wandb','0.18.7'),('rich','14.2.0'),('tqdm','4.67.1')):
            assert md.version(name)==version,'pinned extra dependency changed'
            origin=Path(importlib.import_module(name).__file__).resolve()
            result['extra_modules'][name]={'origin':str(origin),'sha256':digest(origin),'version':version}
        assert torch.cuda.is_available() and torch.cuda.device_count()==1,'exactly one visible CUDA GPU required'
        sys.path.insert(0,str(Path(__file__).resolve().parent/'bridge/vldb_pencil_bridge_v2'))
        from graph_only_transformers_v2 import configure_graph_only_transformers
        result['runtime_compatibility']=configure_graph_only_transformers()
        result['visible_gpu']={'name':torch.cuda.get_device_name(0),'count':torch.cuda.device_count()}
    return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('base','isolated'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    assert not a.output.exists(),'preserve existing probe'
    value=probe(a.mode);a.output.write_text(json.dumps(value,indent=2)+'\n');return 0

if __name__=='__main__':raise SystemExit(main())
