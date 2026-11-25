"""One local synthetic callspy of the actual frozen SamplerHook; no graph/GPU."""
from pathlib import Path
import hashlib,json,math,os,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BRIDGE=ROOT/'work/vldb_pencil_queue_v3/bridge/vldb_pencil_bridge_v2'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    output=HERE/'ACTUAL_SAMPLER_HOOK_CALLSPY.json'
    assert not output.exists(),'one-shot synthetic receipt already exists'
    assert digest(BRIDGE/'pencil_runtime.py')=='ecd2ffdd38fc225a22e39d4a15fc1735de18e78e5045783f4f232a1880c82ab5'
    sys.path.insert(0,str(BRIDGE))
    import numpy as np
    import torch
    from pencil_runtime import SamplerHook
    calls=[]
    def forbidden(*args):
        calls.append('forbidden')
        raise AssertionError('real sampler must not run for frozenS replay')
    hook=SamplerHook(forbidden);frozen=np.array([0,2,7],dtype=np.int64);hook.frozen=frozen
    replay=hook(None)
    assert calls==[] and replay[:2]==(None,None)
    assert torch.equal(replay[2],torch.tensor([0,2,7],dtype=torch.int64))
    assert math.isfinite(hook.last_seconds) and hook.last_seconds>0
    replay_seconds=hook.last_seconds
    def actual(*args):
        calls.append('online')
        return ('first','second',torch.tensor([1,4],dtype=torch.int64))
    online=SamplerHook(actual);value=online(None)
    assert calls==['online'] and torch.equal(value[2],torch.tensor([1,4]))
    assert math.isfinite(online.last_seconds) and online.last_seconds>=0
    record={'kind':'ACTUAL_FROZEN_SAMPLER_HOOK_SYNTHETIC_CALLSPY','status':'pass',
        'runtime_source_sha256':digest(BRIDGE/'pencil_runtime.py'),
        'runner_source_sha256':digest(Path(__file__)),
        'replay_real_calls':0,'online_real_calls':1,'replay_real_raises_if_called':True,
        'frozenS_int64_values_exact':True,'replay_materialization_seconds':replay_seconds,
        'online_seconds':online.last_seconds,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
        'source_modified':False,'scientific_jobs_run':0,'GPU_work':False,
        'scope':'Actual source on a tiny synthetic three-ID frozen set; local CPU Torch version is not native sparse/CUDA/full-graph evidence.'}
    with output.open('x')as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'pass','replay_real_calls':0,'online_real_calls':1,'positive_replay_materialization_timer':replay_seconds>0,'output':str(output)}))
if __name__=='__main__':main()
