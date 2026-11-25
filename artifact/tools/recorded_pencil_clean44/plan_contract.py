"""Independent 44-job construction and strict full controller CLI/default decoder."""
from pathlib import Path
import copy,math
from archive_contract import BASE,REMOTE_CODE,REMOTE_BRIDGE,REMOTE_OUT,YT,require,read,identity,sha
ARMS=('torch_sparse_public','torch_sparse_direct','compiled_scan_int64','compiled_pairwise_int64','compiled_scan_int32','compiled_pairwise_int32','cpp_full_clear','cpp_touched_reset')
HERE=Path(__file__).resolve().parent
GOLDEN=HERE.parent/'vldb_pencil_clean_controller_v1'
def verify_golden(contract):
    require(sha(GOLDEN/'SOURCE_MANIFEST.json')==contract['source_manifest_sha256'],'local golden controller manifest differs')
    rows=read(GOLDEN/'SOURCE_MANIFEST.json')['files'];require(rows==contract['source_files']and len(rows)==108,'local golden108 source inventory differs')
    for row in rows:require(identity(GOLDEN/row['path'])=={k:row[k]for k in('bytes','sha256')},'local frozen controller file differs: '+row['path'])
    require(sha(GOLDEN.parent/'vldb_pencil_clean_controller_v1_sources.tgz')==contract['source_packet_sha256'],'local golden packet differs')
def controller_configuration(launch,contract):
    argv=launch['argv'];require(isinstance(argv,list)and argv[0]==REMOTE_CODE+'/controller.py','actual launch argv entrypoint differs')
    values={'execute':False,'base':BASE,'out':None,'prepared_root':BASE+'/data/citation2/prepared','validation_root':BASE+'/clean_validation512',
        'python':None,'max_hours':4.0,'memory_guard_gib':96.0,'pod_memory_limit_gib':256.0,'target':YT,'admission':None,'admission_sha256':None,
        'source_manifest_sha256':None,'source_packet':None,'source_packet_sha256':None}
    tail=iter(argv[1:]);seen=set()
    for token in tail:
        require(token.startswith('--'),'unexpected controller positional token');key=token[2:].replace('-','_');require(key in values and key not in seen,'duplicate/unknown controller CLI option');seen.add(key)
        if type(values[key])is bool:values[key]=True
        else:
            value=next(tail);values[key]=float(value)if type(values[key])is float else value
    require(values['execute']is True and values['base']==BASE and values['out']==REMOTE_OUT and values['target']==YT,'actual fixed execution/base/output/target differs')
    require(isinstance(values['python'],str)and values['python'].startswith(BASE+'/')and values['python'].endswith('/bin/python'),'explicit supplied venv lexical interpreter required')
    require(math.isfinite(values['max_hours'])and .25<values['max_hours']<=4 and values['memory_guard_gib']==96 and values['pod_memory_limit_gib']==256,'actual finite execution/96guard/256cgroup limits differ')
    require(values['prepared_root']==BASE+'/data/citation2/prepared'and values['validation_root']==BASE+'/clean_validation512','exact fresh prepared and frozen validation root required')
    require(values['source_manifest_sha256']==contract['source_manifest_sha256']and values['source_packet_sha256']==contract['source_packet_sha256']and values['admission_sha256']==launch['root_admission_sha256'],'actual launch source/root pins differ')
    for key in('source_packet','admission'):require(isinstance(values[key],str)and values[key].startswith(BASE+'/'),'actual supplied source packet/admission path required')
    return values

def make_jobs(config):
    out=Path(config['out']);bridge=Path(REMOTE_BRIDGE);tiny=out/'tiny_trace';real=out/'real_trace';cpp=['--cpp-module',str(bridge/'corrected_native/native_adapter_v1.py'),'--cpp-library',str(out/'cpp_build/native_assoc_v1.so'),'--cpp-build-receipt',str(out/'cpp_build/build_receipt_v1.json')]
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

def validate_plan(plan,launch,contract):
    verify_golden(contract);config=controller_configuration(launch,contract);jobs=make_jobs(config)
    protocol=read(GOLDEN/'candidate/PROTOCOL.json');inherited=protocol['GPU_target'];protocol['GPU_target']='one visible H10080Single per job; active controller27CPU205/256GiB contract'
    setup=[{'name':'platform_before','gpu':True,'timeout_seconds':120,'argv':[config['python'],'-B',REMOTE_CODE+'/platform_probe.py','--output',REMOTE_OUT+'/platform_before.json']},{'name':'build_cpp','gpu':False,'timeout_seconds':180,'argv':[config['python'],'-B',REMOTE_BRIDGE+'/corrected_native/build_native_v1.py','--output-dir',REMOTE_OUT+'/cpp_build']}]
    after=[{'name':'platform_after','gpu':True,'timeout_seconds':120,'argv':[config['python'],'-B',REMOTE_CODE+'/platform_probe.py','--output',REMOTE_OUT+'/platform_after.json']}]
    expected={'status':'ROOT_REVIEWABLE_CONTROLLER_PLAN','execution_authorized':False,'jobs':jobs,'scientific_jobs':44,'protocol':protocol,'source_job_builder_sha256':sha(GOLDEN/'candidate/metadata/ORIGINAL_SCIENTIFIC_JOB_BUILDER.py'),'output_created':False,'inherited_candidate_GPU_target_reference':inherited,'controller_protocol':read(GOLDEN/'PROTOCOL.json'),'setup_jobs':setup,'after_jobs':after}
    require(plan==expected,'complete independent44-job/controller/setup/after protocol differs')
    return config,jobs
