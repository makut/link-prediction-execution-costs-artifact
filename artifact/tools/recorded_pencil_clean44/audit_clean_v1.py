#!/usr/bin/env python3
"""Read-only independent clean-consumer auditor; never executes numerical producer code."""
from pathlib import Path
import argparse,copy,json,sys
from archive_contract import *
from science_core import *
from trace_audit import load_trace
from plan_contract import ARMS,validate_plan
from evidence_checks import phase,platform,cpp_build,public_inputs,strict_manifest
HERE=Path(__file__).resolve().parent

def inventory(payload,jobs,science,contract):
    completed=read(payload/'completed_jobs.json')if(payload/'completed_jobs.json').exists()else{'jobs':[],'count':0}
    count=completed['count'];require(type(count)is int and 0<=count<=44 and count==len(completed['jobs'])==science['completed_jobs'],'completed inventory disagrees with scientific terminal')
    rows=[];raws={}
    for index,job in enumerate(jobs):
        output=payload/(job['name']+'.json');exitpath=payload/(job['name']+'.exit.json');raw=read(output)if output.exists()else None;ex=read(exitpath)if exitpath.exists()else None
        if index<count:
            saved=completed['jobs'][index];require(saved['name']==job['name']and saved['kind']==job['kind']and raw is not None and ex is not None and saved['output']==identity(output)and saved['exit']==ex,'completed inventory must bind whole exact executed-job prefix/output/exit')
            _,guard=phase(payload,job);configuration(raw,job,contract,ARMS);require(raw['configuration']['query_memory']=='disabled','clean mode must be explicitly disabled')
            raws[job['name']]=raw;rows.append({'job':job['name'],'kind':job['kind'],'status':'completed_admitted','guard':guard,'output':identity(output)})
        else:
            require(index==count or(raw is None and ex is None),'scientific evidence exists after first uncompleted job')
            status='not_started'if raw is None and ex is None else'guarded_or_failed'if raw is None or raw.get('status')in('failed','guarded_skip')else'executed_but_not_admitted'
            rows.append({'job':job['name'],'kind':job['kind'],'status':status,'driver_status':raw.get('status')if raw else None,'exit':ex,'driver_error':raw.get('error')if raw else None})
    if science['scientific_success']:require(count==44,'successful science requires all44 predeclared jobs')
    return rows,raws

def cross_scores(rows):
    grouped={};expected={('online',b,w):set(ARMS)for b in(128,512)for w in(0,1)}
    for row in rows:
        key=(row['mode'],row['batch'],row['workers']);bucket=grouped.setdefault(key,{})
        require(row['arm']not in bucket,'duplicate benchmark arm');bucket[row['arm']]=row
    require(set(grouped)==set(expected),'all4 ONLINE shape/worker groups required')
    groups=[]
    for key,bucket in grouped.items():
        require(set(bucket)==expected[key],'all8 frozen native/candidate arms required in each group')
        ref=bucket['torch_sparse_public'];fp=ref['score_fingerprint'];logits_fp(fp,1024)
        require(all(row['model']==ref['model']and len(row['all_sweep_score_fingerprints'])==4 and all(v==fp for v in row['all_sweep_score_fingerprints'])for row in bucket.values()),'all1024 ordered scores differ across arms/warmup/repeats within group')
        groups.append({'mode':'online','precision':'bf16','batch':key[1],'workers':key[2],'arms':list(bucket),'all_4_sweeps_all_arms_scores_exact':True,'score_fingerprint':fp})
        for row in bucket.values():row['ratios_of_median_wall_time']={name:bucket[name]['median_seconds']/row['median_seconds']for name in('torch_sparse_public','torch_sparse_direct','cpp_touched_reset')}
    return groups

def scientific_audit(context,contract,config,jobs,raws):
    payload=context['payload'];observed,environment=platform(payload,contract,context['plan'],config);inputs=public_inputs(payload,contract,config)
    paths={True:payload/'tiny_trace',False:payload/'real_trace'};traces={k:load_trace(v,7 if k else 1024,k)for k,v in paths.items()};semantic=[];audits={};bench=[];real_environment=None
    for planned in jobs:
        job=dict(planned);raw=raws[job['name']];fixture=job['name'].startswith('tiny');trace=traces[fixture];input_binding(raw,trace,contract,fixture)
        if not fixture:strict_manifest(raw,contract,inputs)
        if job['kind']=='preflight':continue
        native_environment(raw,observed,9 if fixture else contract['input_contract']['n'],12 if fixture else contract['input_contract']['m'])
        if not fixture:
            if real_environment is None:real_environment=raw['audit_environment']
            require(raw['audit_environment']==real_environment,'actual native/software/thread origins changed across real clean jobs')
        if job['kind']=='capture':capture_receipt(raw,trace,paths[fixture],7 if fixture else 1024)
        elif job['kind']=='audit':
            job['_trace_path']=paths[fixture];semantic.append({'job':job['name'],**semantic_receipt(raw,job,trace,ARMS)});require(raw['result']['model']['bert_config_sha256']==contract['driver_sources']['bert_config/config.json'],'actual model consumed another offline config');audits[job['name']]=raw
        elif job['kind']=='benchmark':bench.append(benchmark_receipt(raw,job,payload,contract,audits))
    groups=cross_scores(bench);require(len(bench)==32 and len(semantic)==9,'complete32 throughput and9 tiny/real semantic jobs required')
    stored=read(payload/'crossjob_scores.json');require(stored['status']=='pass'and stored['groups']==[{'precision':g['precision'],'batch':g['batch'],'workers':g['workers'],'mode':g['mode'],'arms':g['arms'],'warmup_and_3repeats_all_model_scores_exact':True,'score_fingerprint':g['score_fingerprint']}for g in groups],'controller cross-score receipt must cover exactly4 ONLINE groups')
    for row in bench:row.pop('all_sweep_score_fingerprints');row.pop('model')
    return {'status':'pass','environment':environment,'full_public_input_payloads':19,'frozen_validation_sources':512,'traces':{'tiny':traces[True]['report'],'real':traces[False]['report']},'semantic_audits':semantic,'measured_score_groups':groups,'primary_online_bf16_rows':bench,
        'limitations':['Random frozen model: no trained-checkpoint quality, MRR, ranking, SOTA, training or HPO comparison.',
        'All native graph representations remain resident; no PENCIL memory-saving result.',
        'Per-query memory/smaps disabled; worker memory is unobserved/null. Parent observations lie outside wall; external96GiB guard is sampled, not a true peak.',
        'Mandatory S verification, wrapper/IPC, collation, H2D, forward and score copy remain in wall. Workers overlap; CPU stage sums are not additive wall components.',
        'Raw stages nest sampler and induction; induction includes physical operator plus SparseTensor/value packing.',
        'One excluded full warmup and three measured full sweeps; ratios use medians, without paired-query confidence claims or favorable configuration selection.',
        'One independently observed H100 platform; original instrumented V3 or other-host times cannot establish a causal speed ratio.',
        'GPU high water is cumulative since reset before warmup. Real graph tensors are not exported; local audit checks source/hash/bindings and full native equivalence receipts, not a new full-graph oracle.',
        'Durable tar plus post-readback attribute prove the archived science boundary. Outer final controller PID exit is explicitly UNKNOWN.']}

def build_report(context,contract):
    config,jobs=validate_plan(context['plan'],context['launch'],contract);contract=copy.deepcopy(contract)
    build=context['payload']/'cpp_build/build_receipt_v1.json'
    if build.exists():contract['cpp']=cpp_build(context['payload'],contract)
    rows,raws=inventory(context['payload'],jobs,context['science'],contract);success=context['science']['scientific_success']
    report={'schema_version':1,'kind':'independent_local_pencil_clean_consumer_v1_audit','input_export':str(context['folder']),'status':'scientific_checks_passed'if success else'persisted_scientific_failure_no_performance_analysis','transport_integrity':context['integrity'],'scientific_success':success,'outer_final_pid_exit':'UNKNOWN','scientific_failure':context['science']['failure'],'completed_jobs':len(raws),'planned_jobs':44,'inventory':rows,'independent_scope':'NumPy/stdlib only; no producer parser/validator imports, native replay, Torch, remote access or source mutation.'}
    report['scientific_checks']=scientific_audit(context,contract,config,jobs,raws)if success else{'status':'withheld','reason':'Persisted transport integrity is separate from scientific success; no throughput/speedup analysis for failed, guarded or incomplete science.'}
    return report

def text_report(report):
    lines=['Independent local clean-consumer PENCIL audit','Status: '+report['status'],f"Admitted jobs: {report['completed_jobs']}/44",'Outer final controller PID exit: UNKNOWN']
    checks=report['scientific_checks']
    if checks['status']=='pass':
        lines+=['','ONLINE BF16, per-query smaps disabled;1024 queries;median of3 measured sweeps.','batch workers arm median_seconds queries/second ratio_vs_public']
        for row in checks['primary_online_bf16_rows']:lines.append(f"{row['batch']} {row['workers']} {row['arm']} {row['median_seconds']:.9g} {row['queries_per_second_from_median_seconds']:.9g} {row['ratios_of_median_wall_time']['torch_sparse_public']:.6g}")
        lines+=['',*['- '+v for v in checks['limitations']]]
    else:lines.append(checks['reason'])
    return '\n'.join(lines)+'\n'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-launch',type=Path,required=True);p.add_argument('--root-admission',type=Path,required=True);a=p.parse_args(argv)
    if not(a.export/'export.json').exists():print(json.dumps({'status':'waiting_for_verified_export','scientific_analysis_performed':False,'output_created':False}));return 0
    export=a.export.resolve();output=a.output.resolve();require(not output.exists()and all(not output.is_relative_to(v)and not v.is_relative_to(output)for v in(export,HERE)), 'fresh independent output outside evidence/source trees required')
    contract=read(HERE/'FROZEN_CONTRACT.json');context=verify_export(export,contract,a.root_launch,a.root_admission);report=build_report(context,contract)
    output.mkdir(parents=True);(output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(output/'summary.txt').write_text(text_report(report));print(json.dumps({'status':report['status'],'completed_jobs':report['completed_jobs'],'planned_jobs':44,'output':str(output)}));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except(InvalidEvidence,KeyError,IndexError,TypeError,ValueError,OSError,EOFError,OverflowError,StopIteration)as error:
        print(json.dumps({'status':'invalid_evidence','error_type':type(error).__name__,'message':str(error),'performance_analysis_eligible':False}),file=sys.stderr);raise SystemExit(2)
