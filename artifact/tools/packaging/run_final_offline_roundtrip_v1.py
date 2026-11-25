#!/usr/bin/env python3
"""Run recorded audits/quality CLI from a fresh archive restoration elsewhere.

Local archival checks only; no training, graph/model loading or network calls.
The source archive is verified against its external full-readback receipt.
"""
from pathlib import Path
import argparse, gzip, json, math, os, shutil, subprocess, sys, tarfile
import freeze_local_archive_v1 as archive_tool

MIB=1<<20
def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x')as s:json.dump(value,s,indent=2,sort_keys=True);s.write('\n')
def restore(archive,out):
    require(not out.exists()and not out.is_symlink(),'fresh restored code directory required')
    out.mkdir(parents=True);seen=set()
    with gzip.open(archive,'rb')as uncompressed,tarfile.open(fileobj=uncompressed,mode='r|')as tar:
        for m in tar:
            require(m.name.startswith(archive_tool.PREFIX),'exact code archive prefix required')
            name=archive_tool.safe(m.name[len(archive_tool.PREFIX):])
            require(m.isfile()and not m.issparse()and not m.linkname and name not in seen,'regular unique restoration member required')
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m)as source,target.open('xb')as sink:
                for block in iter(lambda:source.read(MIB),b''):sink.write(block)
            seen.add(name)
        while uncompressed.read(MIB):pass
    rows=archive_tool.verify_snapshot(out);require(seen==set(rows),'whole exact restored snapshot required')
    return rows
def run(archive,archive_receipt,assets,restored,output):
    archive=Path(archive).resolve();archive_receipt=Path(archive_receipt).resolve();assets=Path(assets).resolve()
    restored=Path(restored).resolve();output=Path(output).resolve()
    require(not output.exists()and not output.is_symlink(),'fresh independent offline output required')
    for parent in(restored.parent,output.parent):
        parent.mkdir(parents=True,exist_ok=True)
        require(shutil.disk_usage(parent).free>=archive_tool.RESERVE+768*MIB,
                '1.5GiB reserve plus768MiB bounded code/evidence/output allowance required')
    receipt=read(archive_receipt);require(receipt['status']=='PASS'and archive_tool.pin(archive)==receipt['archive'],'exact final archive/readback receipt required')
    rows=restore(archive,restored);output.mkdir(parents=True)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(restored/'src')}
    commands=[]
    def command(name,args):
        ran=subprocess.run([sys.executable,'-B',*map(str,args)],cwd=output,env=env,capture_output=True,text=True,timeout=120)
        (output/(name+'.stdout.txt')).write_text(ran.stdout);(output/(name+'.stderr.txt')).write_text(ran.stderr)
        commands.append({'name':name,'argv':[sys.executable,'-B',*map(str,args)],'returncode':ran.returncode,
                         'stdout':archive_tool.pin(output/(name+'.stdout.txt')),'stderr':archive_tool.pin(output/(name+'.stderr.txt'))})
        require(ran.returncode==0,'offline command failed: '+name)
        return ran.stdout
    tools=restored/'tools';meta=restored/'metadata/evidence'
    e20=output/'recorded_E20'
    command('E20_materialize',[tools/'public_evidence/materialize_cpu_e20_v1.py','--metadata',meta/'E20/recorded_export',
       '--archive',assets/'lpp_public_friendster_cpu_matched_v2_raw.tgz','--out',e20])
    cpu=output/'CPU_E20_AUDIT.json'
    command('E20_frozen_audit',[tools/'recorded_cpu_v2/audit_public_fr_cpu_v1.py','--export-dir',e20,
       '--root-launch',meta/'E20/ROOT_LAUNCH.json','--source-packet',meta/'E20/public_fr_cpu_v2_sources.tgz',
       '--historical-systems-asset',assets/'lpp_public_systems_evidence_v1.tgz','--wheel-packet',assets/'pt22_native_wheels_v1.tgz','--output',cpu])
    require(read(cpu)==read(meta/'E20/ACTUAL_INDEPENDENT_AUDIT.json'),'relocated fullE20 audit differs')
    equality={'E20_entire_frozen_audit_exact':True}
    for role in('E19','E21'):
        export=output/('recorded_'+role)
        filename='lpp_public_pencil_'+('instrumented48'if role=='E19'else'clean44')+'_raw.tgz'
        command(role+'_materialize',[tools/'public_evidence/materialize_pencil_export_v1.py','--role',role,
           '--metadata',meta/role/'recorded_export','--archive',assets/filename,'--out',export])
        audit=output/(role+'_AUDIT')
        if role=='E19':
            command(role+'_frozen_audit',[tools/'recorded_pencil_e19_v1/audit_recorded_e19.py','--source-root',restored,
               '--export',export,'--root-launch',meta/role/'ROOT_LAUNCH.json','--output',audit])
            original=meta/role/'ACTUAL_DERIVED_AUDIT_V2.json';expected=48
        else:
            command(role+'_frozen_audit',[tools/'recorded_pencil_clean44/audit_clean_v1.py','--export',export,
               '--root-launch',meta/role/'ROOT_LAUNCH.json','--root-admission',meta/role/'ROOT_ADMISSION.json','--output',audit])
            original=meta/role/'ACTUAL_INDEPENDENT_AUDIT.json';expected=44
        actual=read(audit/'audit.json')
        require(actual['status']=='scientific_checks_passed'and actual['planned_jobs']==actual['completed_jobs']==expected,
                'all original recorded jobs must remain scientifically closed')
        require(actual['scientific_checks']==read(original)['scientific_checks'],'relocated frozen scientific checks differ')
        equality[role+'_scientific_checks_exact']=True
    small=output/'small_friendster_RR'
    command('Friendster_small_members',[tools/'public_evidence/extract_friendster_rr_v1.py',
       '--quality-asset',assets/'lpp_public_quality_evidence_v1.tgz','--out',small])
    fr=json.loads(command('Friendster_RR_report',[tools/'public_evidence/reaggregate_quality_v1.py',
       '--quality-root',small,'--dataset','friendster']))
    require(fr==read(meta/'QUALITY/OFFLINE_CLOSURE/FRIENDSTER_RR_REAGGREGATION_v1.json'),'FR original reporting closure differs')
    raw_scores=output/'Friendster_FULL_SCORE_AUDIT.json'
    score_tool=tools/'recorded_friendster_scores_v1'
    command('Friendster_full_float_scores',[score_tool/'verify_friendster_scores.py',
       '--quality-asset',assets/'lpp_public_quality_evidence_v1.tgz','--manifest',restored/'work/vldb_public_fresh_adapters_v1/QUALITY_MANIFEST.json',
       '--output',raw_scores])
    score_manifest=read(score_tool/'SOURCE_MANIFEST.json')
    original_name=next(row['path']for row in score_manifest['files']if row['path'].startswith('ACTUAL_FRIENDSTER_FLOAT_SCORE_RECEIPT_'))
    original_scores=read(score_tool/original_name);reproduced_scores=read(raw_scores)
    # Process high-water memory is an actual observation, not a byte-identity target.
    require({k:v for k,v in reproduced_scores.items()if k!='peak_rss_bytes'}==
            {k:v for k,v in original_scores.items()if k!='peak_rss_bytes'},'FR fullscore numeric/cohort/provenance closure differs')
    require(reproduced_scores['peak_rss_bytes']<reproduced_scores['peak_rss_limit_bytes_exclusive']==256*MIB,
            'FR fullscore process peak must remain below256MiB')
    c2=json.loads(command('Citation2_full_logits_report',[tools/'public_evidence/reaggregate_quality_v1.py',
       '--quality-asset',assets/'lpp_public_quality_evidence_v1.tgz','--manifest',restored/'work/vldb_public_fresh_adapters_v1/QUALITY_MANIFEST.json','--dataset','citation2']))
    original_c2=read(meta/'QUALITY/OFFLINE_CLOSURE/CITATION2_FULL_SCORE_REAGGREGATION_v1.json')
    volatile={'elapsed_seconds','peak_rss_bytes'}
    require({k:v for k,v in c2.items()if k not in volatile}==
            {k:v for k,v in original_c2.items()if k not in volatile},'C2 fullscore numeric/cohort/provenance closure differs')
    require(type(c2['elapsed_seconds'])in(int,float)and math.isfinite(c2['elapsed_seconds'])and c2['elapsed_seconds']>=0
            and type(c2['peak_rss_bytes'])is int and 0<c2['peak_rss_bytes']<=512*MIB,
            'actual C2 elapsed/memory observation must satisfy declared bounds')
    equality.update({'Friendster_original_RR_report_exact':True,'Citation2_full_score_numeric_cohort_and_provenance_exact':True})
    equality['Friendster_full_float_score_numeric_cohort_and_provenance_exact']=True
    catalog=read(restored/'metadata/EXTERNAL_ASSETS.json')['assets']
    systems_archive=assets/catalog['public_systems']['filename']
    require(archive_tool.pin(systems_archive)=={k:catalog['public_systems'][k]for k in('bytes','sha256')},'exact whole systems asset required')
    systems=output/'systems_evidence';systems.mkdir();seen=set()
    with tarfile.open(systems_archive,'r|gz')as tar:
        for m in tar:
            prefix='lpp_public_systems_evidence_v1/'
            require(m.name.startswith(prefix),'exact systems namespace required')
            name=archive_tool.safe(m.name[len(prefix):])
            require(m.isfile()and not m.issparse()and not m.linkname and name not in seen and m.size<128*MIB,
                    'unique bounded regular systems evidence member required')
            target=systems/name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m)as source,target.open('xb')as sink:
                for block in iter(lambda:source.read(MIB),b''):sink.write(block)
            seen.add(name)
    systems_report=json.loads(command('Recorded_systems_tables',[tools/'public_evidence/verify_systems_v1.py','--asset-root',systems]))
    require(systems_report['status']=='PASS'and systems_report['verified_payloads']==391,'all391 exact systems payloads required')
    command('Recorded_three_figures',[tools/'public_evidence/plot_public_figures_v1.py','--asset-root',systems,
       '--quality-asset',assets/'lpp_public_quality_evidence_v1.tgz','--out',output/'figures'])
    command('E17_posthoc_analysis',['-m','artifact_public.reproduce_posthoc_friendster','--out',output/'posthoc_E17','--execute'])
    posthoc=read(output/'posthoc_E17/analysis.json')
    require(posthoc['status']=='verified_public_friendster_projection'and len(posthoc['rows'])==768,'all768 independent E17 rows required')
    dataset_full=read(restored/'metadata/friendster_full_expected.json')
    dataset_fr=read(restored/'metadata/friendster_prepared_expected.json')['files']
    dataset_c2={row['path']:row for row in read(restored/'metadata/citation2_prepared_expected.json')['files']}
    dimensions={'friendster_full':[dataset_full['num_nodes'],dataset_full['stored_entries']],
      'friendster_train':[dataset_fr['train_csr/edge_starts.bin']['bytes']//8,(dataset_fr['train_csr/edge_ends.bin']['bytes']-8)//4],
      'citation2_train':[dataset_c2['train_csr/edge_starts.bin']['bytes']//8,(dataset_c2['train_csr/edge_ends.bin']['bytes']-8)//4]}
    require(dimensions=={'friendster_full':[65608366,3612134270],'friendster_train':[65608366,2889707416],
                        'citation2_train':[2927963,60703502]},'saved platform/dataset projection differs')
    hardware=read(meta/'E15/HARDWARE_CURRENT_PROVENANCE.json')['pods']
    require('6338'in hardware['lpp-cpu-0']['cpu_model']and'8480'in hardware['seal-revision-20261124-0']['cpu_model'],
            'saved admitted platform metadata differs')
    objectmap=read(restored/'PUBLIC_OBJECT_COMMANDS.json')
    require(len(objectmap['objects'])==objectmap['object_count']==13,'all13 current paper objects required')
    object_labels=[row['manuscript_label']for row in objectmap['objects']]
    equality.update({'all391_systems_payloads_verified':True,'all_three_figure_numeric_panels_recomputed':True,
      'all768_E17_rows_independently_recomputed':True,'saved_dataset_and_platform_metadata_projected':True})
    require(archive_tool.verify_snapshot(restored)==rows,'restored code changed during offline verification')
    report={'kind':'ACTUAL_FINAL_ARCHIVE_UNRELATED_PATH_OFFLINE_ROUNDTRIP_V1','status':'PASS',
       'archive':archive_tool.pin(archive),'archive_receipt':archive_tool.pin(archive_receipt),
       'restored_source_manifest':rows['MANIFEST.json'],'restored_code_path':str(restored),'cwd':str(output),
       'commands':commands,'exact_equalities':equality,'saved_dataset_dimensions':dimensions,
       'mapped_current_paper_objects':object_labels,'restored_code_unchanged_after_checks':True,
       'science_jobs_or_training':0,'graphs_or_models_loaded':0,'remote_actions':0,'YT_writes':0}
    write(output/'OFFLINE_ROUNDTRIP.json',report);return report
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('archive','archive-receipt','assets','restored','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv);r=run(a.archive,a.archive_receipt,a.assets,a.restored,a.output)
    print(json.dumps({'status':r['status'],'commands':len(r['commands']),'exact_equalities':r['exact_equalities']}))
if __name__=='__main__':main()
