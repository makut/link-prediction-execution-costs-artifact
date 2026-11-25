"""Fresh supplied-stack admission, before/after; no installation or graph load."""
from pathlib import Path
import argparse,csv,importlib,importlib.metadata as md,importlib.util,json,os,subprocess,sys
sys.dont_write_bytecode=True
from contracts import HERE,BRIDGE,require,sha,save

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def cpu_observation(proc=Path('/proc'),cgroup=Path('/sys/fs/cgroup')):
    with (proc/'cpuinfo').open()as stream:text=stream.read((2<<20)+1)
    bounded=text[:2<<20]
    models=sorted({line.split(':',1)[1].strip()for line in bounded.splitlines()if line.startswith('model name')})
    quota=(cgroup/'cpu.max').read_text().strip()if(cgroup/'cpu.max').is_file()else None
    return {'models_from_bounded_cpuinfo':models,'cpuinfo_prefix_truncated':len(text)>2<<20,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'cgroup_cpu_max':quota,
        'scope':'Observed platform/quota; scheduler request27CPU is not dedicated-core ownership.'}
def probe():
    require(sys.platform.startswith('linux')and sys.version_info[:2]==(3,10),'Linux Python 3.10 target required')
    sys.path.insert(0,str(BRIDGE))
    from graph_only_transformers_v2 import configure_graph_only_transformers
    policy=configure_graph_only_transformers()  # Before any BERT/vision import.
    original=load(HERE/'third_party/runtime_probe_original_v3.py','clean_original_runtime_probe_v3')
    value=original.probe('base')
    expected={'torch':'2.5.0+cu121','numpy':'1.24.4','scipy':'1.10.1','numba':'0.58.0',
              'torch-geometric':'2.7.0','transformers':'4.46.2','wandb':'0.18.7','rich':'14.2.0','tqdm':'4.67.1'}
    require(all(value['packages'].get(name)==version for name,version in expected.items()),'supplied numerical/HF target differs')
    require(value['torch_cuda']=='12.1'and value['torch_sparse_version'].split('+')[0]=='0.6.18'
        and value['saint_wrapper']['git_blob']=='440455853e0309b00376cbe9ad25d7226cfa5e0e'and bool(value['torch_sparse_binaries']),'native CUDA/wrapper/binary admission differs')
    import torch
    require(torch.cuda.is_available()and torch.cuda.device_count()==1,'exactly one visible GPU required')
    properties=torch.cuda.get_device_properties(0)
    require('H100'in properties.name and 79*(1<<30)<=properties.total_memory<=82*(1<<30),'oneH10080 target required')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=list(csv.reader(raw.splitlines(),skipinitialspace=True));require(len(rows)==1 and len(rows[0])==3 and rows[0][0].startswith('GPU-'),'one whole-GPU inventory required')
    value['visible_gpu']={'uuid':rows[0][0],'name':properties.name,'count':1,'total_memory_bytes':properties.total_memory}
    value['cpu_observation']=cpu_observation()
    value['runtime_compatibility']=policy;value['extra_modules']={}
    for name in('transformers','wandb','rich','tqdm'):
        origin=Path(importlib.import_module(name).__file__).resolve();value['extra_modules'][name]={'origin':str(origin),'sha256':sha(origin),'version':md.version(name)}
    require(Path(policy['import_utils_origin'])==Path(value['extra_modules']['transformers']['origin']).parent/'utils/import_utils.py'
        and sha(policy['import_utils_origin'])==policy['import_utils_sha256']and policy['policy_source_sha256']==sha(BRIDGE/'graph_only_transformers_v2.py'),'actual pinned HF policy origins differ')
    value['scope']='Fresh supplied image/native/numerical/HF/GPU identity; no old venv name, recovery PID, graph load, BERT forward, package install or source mutation.'
    return value
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    require(not a.output.exists()and not a.output.is_symlink(),'fresh platform receipt required');save(a.output,probe(),exclusive=True);return 0
if __name__=='__main__':raise SystemExit(main())
