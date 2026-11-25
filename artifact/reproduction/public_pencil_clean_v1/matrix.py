"""Independent generic exact44-job clean PENCIL matrix; no results or execution."""
from pathlib import Path
from common import BRIDGE
from science_validation import ARMS
def make_jobs(config):
    out=Path(config['out']);bridge=BRIDGE;tiny=out/'tiny_trace';real=out/'real_trace';cpp=['--cpp-module',str(bridge/'corrected_native/native_adapter_v1.py'),'--cpp-library',str(out/'cpp_build/native_assoc_v1.so'),'--cpp-build-receipt',str(out/'cpp_build/build_receipt_v1.json')]
    inputs=['--graph-path',config['prepared_root']+'/train_csr','--features',config['prepared_root']+'/node_feat.npy','--valid-positive',config['validation_root']+'/valid_edge.npy','--valid-negative',config['validation_root']+'/valid_edge_neg.npy','--strict-manifest',config['prepared_root']+'/manifest.json']
    rows=[]
    def add(name,kind,tail,gpu=False,**meta):
        output=str(out/(name+'.json'));rows.append({'name':name,'kind':kind,'argv':[config['python'],'-u',str(bridge/'pencil_bridge.py'),*tail,'--output',output,'--query-memory','disabled'],'output':output,'gpu':gpu,'timeout_seconds':1800,**meta})
    add('tiny_capture','capture',['capture','--fixture','--execute','--trace-dir',str(tiny)])
    add('tiny_audit','audit',['audit','--fixture','--execute','--trace-dir',str(tiny),'--arms',','.join(ARMS),*cpp,'--forward','--tiny-model','--training-wrapper','--batch-size','128'])
    add('real_preflight','preflight',['preflight',*inputs])
    add('real_capture','capture',['capture','--execute',*inputs,'--trace-dir',str(real)])
    for precision in('fp32','bf16'):
        for batch in(128,512):
            for workers in(0,1):
                add(f'audit_{precision}_b{batch}_w{workers}','audit',['audit','--execute',*inputs,'--trace-dir',str(real),'--arms',','.join(ARMS),*cpp,'--forward','--training-wrapper','--device','cuda','--precision',precision,'--batch-size',str(batch),'--num-workers',str(workers),'--max-logit-abs','0'],True,precision=precision,batch=batch,workers=workers)
    for batch in(128,512):
        for workers in(0,1):
            for arm in ARMS:
                add(f'bench_online_bf16_b{batch}_w{workers}_{arm}','benchmark',['benchmark','--execute',*inputs,'--trace-dir',str(real),*cpp,'--audit-receipt',str(out/f'audit_bf16_b{batch}_w{workers}.json'),'--arm',arm,'--pipeline-mode','online','--forward','--device','cuda','--precision','bf16','--batch-size',str(batch),'--num-workers',str(workers),'--max-logit-abs','0','--warmup','1','--repeats','3'],True,mode='online',precision='bf16',batch=batch,workers=workers,arm=arm)
    return rows
