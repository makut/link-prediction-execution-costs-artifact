"""Original unchanged pure PENCIL queueV2 scientific builders, for protocol audit.
Supplied namespace must define Path,BRIDGE,ARMS,REPLAY_ARMS,layout. No entrypoint.
"""
def real_inputs(base):
    return ['--graph-path',str(base/'data/citation2/prepared/train_csr'),
        '--features',str(base/'data/citation2/prepared/node_feat.npy'),
        '--valid-positive',str(base/'results/quality/subsets/valid_edge.npy'),
        '--valid-negative',str(base/'results/quality/subsets/valid_edge_neg.npy'),
        '--strict-manifest',str(base/'data/citation2/prepared/manifest.json'),
        '--quality-plan',str(base/'results/quality/plan.json')]


def make_jobs(base,python):
    L=layout(base);out=L['out'];tiny=out/'tiny_trace';trace=out/'real_trace';script=BRIDGE/'pencil_bridge.py'
    cpp=['--cpp-module',str(L['cpp_module']),'--cpp-library',str(L['cpp_library']),'--cpp-build-receipt',str(L['cpp_receipt'])]
    rows=[]
    def add(name,kind,tail,*,gpu=False,**metadata):
        output=out/(name+'.json');rows.append({'name':name,'kind':kind,'argv':[str(python),'-u',str(script),*tail,'--output',str(output)],
            'output':str(output),'gpu':gpu,'timeout_seconds':1800,**metadata})
    add('tiny_capture','capture',['capture','--fixture','--execute','--trace-dir',str(tiny)])
    add('tiny_audit','audit',['audit','--fixture','--execute','--trace-dir',str(tiny),'--arms',','.join(ARMS),*cpp,
        '--forward','--tiny-model','--training-wrapper','--batch-size','128'])
    add('real_preflight','preflight',['preflight',*real_inputs(base)])
    add('real_capture','capture',['capture','--execute',*real_inputs(base),'--trace-dir',str(trace)])
    for precision in('fp32','bf16'):
        for batch in(128,512):
            for workers in(0,1):
                name=f'audit_{precision}_b{batch}_w{workers}'
                add(name,'audit',['audit','--execute',*real_inputs(base),'--trace-dir',str(trace),'--arms',','.join(ARMS),*cpp,
                    '--forward','--training-wrapper','--device','cuda','--precision',precision,'--batch-size',str(batch),
                    '--num-workers',str(workers),'--max-logit-abs','0'],gpu=True,precision=precision,batch=batch,workers=workers)
    for mode in('online','replay'):
        combinations=[(b,w,ARMS)for b in(128,512)for w in(0,1)]if mode=='online'else[(128,0,REPLAY_ARMS)]
        for batch,workers,arms in combinations:
            for arm in arms:
                name=f'bench_{mode}_bf16_b{batch}_w{workers}_{arm}'
                add(name,'benchmark',['benchmark','--execute',*real_inputs(base),'--trace-dir',str(trace),*cpp,
                    '--audit-receipt',str(out/f'audit_bf16_b{batch}_w{workers}.json'),'--arm',arm,'--pipeline-mode',mode,
                    '--forward','--device','cuda','--precision','bf16','--batch-size',str(batch),'--num-workers',str(workers),
                    '--max-logit-abs','0','--warmup','1','--repeats','3'],gpu=True,mode=mode,precision='bf16',batch=batch,workers=workers,arm=arm)
    return rows


