"""Fresh documented Linux CUDA environment admission, without installation."""
from pathlib import Path
import argparse,csv,importlib,importlib.metadata as md,importlib.util,os,re,subprocess,sys
from common import HERE,BRIDGE,GIB,require,sha,save
EXPECTED={'torch':'2.5.0+cu121','numpy':'1.24.4','scipy':'1.10.1','numba':'0.58.0','torch-geometric':'2.7.0','transformers':'4.46.2','wandb':'0.18.7','rich':'14.2.0','tqdm':'4.67.1'}
def cpu_observation():
    raw=Path('/proc/cpuinfo').read_text()[:2<<20]
    return {'models_from_bounded_cpuinfo':sorted({s.split(':',1)[1].strip()for s in raw.splitlines()if s.startswith('model name')}),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip()if Path('/sys/fs/cgroup/cpu.max').is_file()else None,'scope':'Observed CPU/quota; no dedicated-core ownership is inferred.'}
def verify(value,gpu_uuid,gpu_name_contains,minimum_gib):
    require(all(value['packages'].get(k)==v for k,v in EXPECTED.items()),'documented exact numerical/HF stack differs')
    require(value['torch_cuda']=='12.1'and value['torch_sparse_version'].split('+')[0]=='0.6.18'and value['saint_wrapper']['sha256']=='f0928fe9c8cba36021a7b6c470c7ce1b99719b876e1c254cba4cf9d2a717ca34'and value['saint_wrapper']['git_blob']=='440455853e0309b00376cbe9ad25d7226cfa5e0e'and value['torch_sparse_binaries'].get('_saint_cpu.so',{}).get('sha256')=='dfcab1a1c1a5695a775dd21c561bb6669da876bf069e36ce0f55fafb507438ce','fixed original CUDA/native wrapper/CPU binary differs')
    gpu=value['visible_gpu'];require(gpu['count']==1 and gpu['uuid']==gpu_uuid and gpu_name_contains in gpu['name']and gpu['total_memory_bytes']>=minimum_gib*GIB,'actual whole visible CUDA GPU does not meet supplied hardware admission')
    return value

def probe(args):
    require(sys.platform.startswith('linux')and sys.version_info[:2]==(3,10),'documented Linux Python3.10 environment required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu_uuid,'literal supplied GPU UUID visibility required')
    sys.path.insert(0,str(BRIDGE));from graph_only_transformers_v2 import configure_graph_only_transformers
    policy=configure_graph_only_transformers()
    spec=importlib.util.spec_from_file_location('public_pencil_base_probe',HERE/'third_party/runtime_probe_original_v3.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);value=module.probe('base')
    import torch
    require(torch.cuda.is_available()and torch.cuda.device_count()==1,'one actual CUDA device required')
    properties=torch.cuda.get_device_properties(0);raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10);require(len(raw)<=64<<10,'bounded GPU inventory required')
    rows=list(csv.reader(raw.splitlines(),skipinitialspace=True));selected=[r for r in rows if len(r)==3 and r[0]==args.gpu_uuid];require(len(selected)==1,'supplied actual whole GPU UUID absent/ambiguous')
    require(selected[0][1]==properties.name,'selected UUID/actual CUDA device name differs')
    value['visible_gpu']={'uuid':args.gpu_uuid,'name':properties.name,'count':1,'total_memory_bytes':properties.total_memory};value['cpu_observation']=cpu_observation();value['runtime_compatibility']=policy;value['extra_modules']={}
    for name in('transformers','wandb','rich','tqdm'):
        origin=Path(importlib.import_module(name).__file__).resolve();value['extra_modules'][name]={'origin':str(origin),'sha256':sha(origin),'version':md.version(name)}
    require(Path(policy['import_utils_origin'])==Path(value['extra_modules']['transformers']['origin']).parent/'utils/import_utils.py'and sha(policy['import_utils_origin'])==policy['import_utils_sha256']and policy['policy_source_sha256']==sha(BRIDGE/'graph_only_transformers_v2.py'),'actual graph-only HF policy origins differ')
    value['scope']='Fresh supplied public LinuxCUDA stack/source/whole-GPU/CPU observations; no installation, private admission, data or model loading.'
    return verify(value,args.gpu_uuid,args.gpu_name_contains,args.minimum_gpu_memory_gib)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--gpu-name-contains',required=True);p.add_argument('--minimum-gpu-memory-gib',type=float,required=True);a=p.parse_args(argv)
    require(not a.output.exists()and not a.output.is_symlink(),'fresh platform receipt required');save(a.output,probe(a),exclusive=True);return 0
if __name__=='__main__':raise SystemExit(main())
