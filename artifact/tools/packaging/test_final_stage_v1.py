"""Meaningful immutable-stage, offline transport and corruption checks."""
from pathlib import Path
import copy,gzip,hashlib,io,json,os,subprocess,sys,tarfile,tempfile
import pytest
import build_final_stage_v1 as b
import materialize_cpu_e20_v1 as c
import materialize_pencil_export_v1 as p
import extract_friendster_rr_v1 as q
import freeze_local_archive_v1 as f
import review_recorded_public_surface_v1 as surface

AUDIT=b.ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_clean_consumer_v1_actual_independent_audit_v1/audit.json'
PEER=b.ROOT/'work/vldb_clean_actual_numeric_peer_v1/ACTUAL_RAW_NUMERIC_LIFECYCLE_PEER_v1.json'

@pytest.fixture(scope='module')
def definition():return b.plan(AUDIT,PEER)

@pytest.fixture(scope='module')
def staged(definition):
    with tempfile.TemporaryDirectory(prefix='reviewable_v05_final_stage_test_')as directory:
        root=Path(directory)/'candidate';result=b.stage(root,definition)
        yield root,result,Path(directory)

def test_plan_keeps471_and_v04_science_and_separate_assets(definition):
    assert definition['preserved_prior_payloads']==471
    assert definition['gates']['E21_actual_clean44']['accepted']is True
    assert definition['gates']['E19']=='OLD_INSTRUMENTED48_ONLY'
    assert definition['gates']['fresh_full_7arm_or44job_public_adapter_reruns_claimed']is False
    assert definition['payload_bytes']<384<<20
    assert all(not row['included_in_code_snapshot']for row in definition['catalog'].values())
    assert not any(name.endswith(('raw.tgz','pt22_native_wheels_v1.tgz'))for name in definition['files'])

def test_fresh_hardlinked_stage_and_original_manifest_unchanged(staged,definition):
    root,result,temp=staged
    assert result['status']=='ARTIFACT_V0_5_0_LOCAL_RELEASE_PACKAGE'
    assert result['payloads']==len(definition['files'])+len(definition['generated'])+1
    assert b.sha(b.BASE/'MANIFEST.json')==b.BASE_MANIFEST_SHA
    assert b.read(root/'metadata/FINAL_BASELINE_PRESERVATION.json')['all_original471_payload_bytes_rehashed_unchanged']is True
    old=b.read(b.BASE/'MANIFEST.json')
    for name,row in old['files'].items():
        relocated='metadata/lineage/PRIOR_V0_5_CANDIDATE/'+name if name in b.OVERRIDES else name
        assert b.pin(root/relocated)=={k:row[k]for k in('bytes','sha256')}
    assert(root/'src/seal/graph_csr_kernels.py').stat().st_ino==(b.BASE/'src/seal/graph_csr_kernels.py').stat().st_ino
    with pytest.raises(RuntimeError,match='fresh'):b.stage(root,definition)

def test_no_freeze_publication_or_private_binary_surface():
    for name in('data/vk/queries.npy','metadata/evidence/vk_raw/a.json','some/model.pt','some/library.so','../escape'):
        with pytest.raises(RuntimeError):b.safe(name)
    assert b.safe('src/quality/vk_reference_config.json')=='src/quality/vk_reference_config.json'
    assert b.clean_evidence(None,None)['accepted']is False
    with pytest.raises(RuntimeError):b.clean_evidence(AUDIT,None)

def test_wrong_actual_clean_identity_cannot_be_admitted(tmp_path):
    value=b.read(AUDIT);value['planned_jobs']=43;path=tmp_path/'altered_audit.json';path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='exact root-accepted'):b.clean_evidence(path,PEER)

def test_cpu_recorded_asset_materializes_exact_bytes(staged):
    root,result,temp=staged;out=temp/'cpu_recorded'
    report=c.materialize(root/'metadata/evidence/E20/recorded_export',b.ROOT/b.ASSETS['cpu_E20_raw'][0],out)
    assert report['new_observations']==report['numeric_audits']==report['scientific_jobs']==0
    assert c.pin(out/'final_snapshot.tgz')==c.ARCHIVE
    assert c.pin(out/'export.json')==c.pin(root/'metadata/evidence/E20/recorded_export/export.json')
    with pytest.raises(RuntimeError,match='fresh'):c.materialize(root/'metadata/evidence/E20/recorded_export',b.ROOT/b.ASSETS['cpu_E20_raw'][0],out)

@pytest.mark.parametrize('role,asset',[('E19','pencil_E19_instrumented48_raw'),('E21','pencil_E21_clean44_raw')])
def test_pencil_offline_materialization_all_declared_member_bytes(staged,role,asset):
    root,result,temp=staged;out=temp/('recorded_'+role)
    report=p.materialize(role,root/('metadata/evidence/'+role+'/recorded_export'),b.ROOT/b.ASSETS[asset][0],out)
    assert report['new_observations']==report['new_numeric_audits']==report['science_jobs']==0
    declared=p.declared_extraction((out/'extraction_manifest.json').read_bytes())
    assert report['members']==p.ASSETS[role]['members']==len(declared)
    assert{path.relative_to(out/'extracted').as_posix():p.pin(path)for path in(out/'extracted').rglob('*')if path.is_file()}==declared

@pytest.mark.parametrize('kind',['traversal','symlink','duplicate'])
def test_unsafe_synthetic_tar_refused_before_any_output(tmp_path,monkeypatch,kind):
    # Explicit fake bytes exercise the extraction safety gate; the real CLI has
    # no asset-pin override and never accepts these as recorded evidence.
    archive=tmp_path/'fake.tgz';name='../outside'if kind=='traversal'else'one'
    with tarfile.open(archive,'w:gz')as tar:
        entry=tarfile.TarInfo(name);entry.size=1
        if kind=='symlink':entry.type=tarfile.SYMTYPE;entry.linkname='/outside';entry.size=0
        tar.addfile(entry,io.BytesIO(b'x'))
        if kind=='duplicate':tar.addfile(entry,io.BytesIO(b'x'))
    monkeypatch.setitem(p.ASSETS,'E21',{**p.pin(archive),'members':1})
    metadata=tmp_path/'metadata';metadata.mkdir()
    declared={'files':{'one':{'bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}}}
    (metadata/'extraction_manifest.json').write_text(json.dumps(declared));(metadata/'export.json').write_text('{}')
    (metadata/'FILE_INVENTORY.json').write_text(json.dumps({'role':'E21','files':{n:p.pin(metadata/n)for n in('export.json','extraction_manifest.json')}}))
    out=tmp_path/'never-created'
    with pytest.raises(RuntimeError):p.materialize('E21',metadata,archive,out)
    assert not out.exists()and not(tmp_path/'outside').exists()

def test_materialized_clean44_runs_unchanged_frozen_auditor_offline(staged):
    root,result,temp=staged;export=temp/'recorded_E21'
    if not export.exists():p.materialize('E21',root/'metadata/evidence/E21/recorded_export',b.ROOT/b.ASSETS['pencil_E21_clean44_raw'][0],export)
    out=temp/'offline_clean44_audit'
    command=[sys.executable,'-B',str(root/'tools/recorded_pencil_clean44/audit_clean_v1.py'),'--export',str(export),'--output',str(out),
      '--root-launch',str(root/'metadata/evidence/E21/ROOT_LAUNCH.json'),'--root-admission',str(root/'metadata/evidence/E21/ROOT_ADMISSION.json')]
    ran=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert ran.returncode==0,ran.stderr
    reproduced=json.loads((out/'audit.json').read_text());original=b.read(AUDIT)
    assert reproduced['status']=='scientific_checks_passed'and reproduced['completed_jobs']==reproduced['planned_jobs']==44
    assert reproduced['scientific_checks']==original['scientific_checks']

def test_materialized_E19_runs_relocated_frozen_addendum_offline(staged):
    if sys.version_info[:2]!=(3,12):pytest.skip('preserved E19 raw AST contract explicitly requires CPython3.12')
    root,result,temp=staged;export=temp/'recorded_E19'
    if not export.exists():p.materialize('E19',root/'metadata/evidence/E19/recorded_export',b.ROOT/b.ASSETS['pencil_E19_instrumented48_raw'][0],export)
    out=temp/'offline_E19_audit'
    command=[sys.executable,'-B',str(root/'tools/recorded_pencil_e19_v1/audit_recorded_e19.py'),'--source-root',str(root),
      '--export',str(export),'--root-launch',str(root/'metadata/evidence/E19/ROOT_LAUNCH.json'),'--output',str(out)]
    ran=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert ran.returncode==0,ran.stderr
    reproduced=json.loads((out/'audit.json').read_text())
    original=b.read(b.ROOT/'outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_independent_audit_addendum_v2_actual_v1/audit.json')
    assert reproduced['status']=='scientific_checks_passed'and reproduced['completed_jobs']==reproduced['planned_jobs']==48
    assert reproduced['scientific_checks']==original['scientific_checks']

def test_materialized_E20_runs_unchanged_frozen_auditor_offline(staged):
    root,result,temp=staged;export=temp/'cpu_recorded'
    if not export.exists():c.materialize(root/'metadata/evidence/E20/recorded_export',b.ROOT/b.ASSETS['cpu_E20_raw'][0],export)
    out=temp/'offline_cpu_e20_audit.json'
    command=[sys.executable,'-B',str(root/'tools/recorded_cpu_v2/audit_public_fr_cpu_v1.py'),'--export-dir',str(export),
      '--root-launch',str(root/'metadata/evidence/E20/ROOT_LAUNCH.json'),'--source-packet',str(root/'metadata/evidence/E20/public_fr_cpu_v2_sources.tgz'),
      '--historical-systems-asset',str(b.ROOT/b.ASSETS['public_systems'][0]),'--wheel-packet',str(b.ROOT/b.ASSETS['cpu_pt22_wheels'][0]),'--output',str(out)]
    ran=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert ran.returncode==0,ran.stderr
    reproduced=json.loads(out.read_text())
    original=b.read(b.ROOT/'outputs/vldb_public_fr_cpu_audit_20261125/ACTUAL_PUBLIC_FR_CPU_V2_AUDIT_v2_transport_adapter.json')
    assert reproduced==original

def test_original_small_friendster_RR_subset_supports_actual_reporting_CLI(staged):
    root,result,temp=staged;out=temp/'small_original_friendster_quality'
    report=q.extract(b.ROOT/b.ASSETS['public_quality'][0],out)
    assert report['files']==19 and report['bytes']==597978
    assert report['models_or_full_scores_copied']==report['numeric_recomputations']==0
    command=[sys.executable,'-B',str(root/'tools/public_evidence/reaggregate_quality_v1.py'),
             '--quality-root',str(out),'--dataset','friendster']
    ran=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert ran.returncode==0,ran.stderr
    value=json.loads(ran.stdout)
    assert value['status']=='PASS' and len(value['rows'])==6
    assert len(value['aggregates']['paired_difference']['seed_differences'])==3
    with pytest.raises(RuntimeError,match='fresh'):q.extract(b.ROOT/b.ASSETS['public_quality'][0],out)

def test_all13_current_paper_objects_resolve_packaged_commands_and_files(staged):
    root,result,temp=staged;mapping=b.read(root/'PUBLIC_OBJECT_COMMANDS.json')
    labels={'tab:datasets','tab:platforms','fig:native-global-n','tab:posthoc-cost-regimes',
      'fig:posthoc-cost-regimes','tab:native-reset','tab:memory','tab:pipeline','tab:pencil-clean',
      'tab:quality','tab:heuristics','fig:degree-imbalance','tab:citation2-quality'}
    assert len(mapping['objects'])==mapping['object_count']==13
    assert{row['manuscript_label']for row in mapping['objects']}==labels
    for row in mapping['objects']:
        assert row['aggregation'] and row['raw_inputs'] and row['selector_and_scope']
        assert set(row['command_ids'])<=set(mapping['commands'])
        for raw in row['raw_inputs']:
            if'artifact_file'in raw:assert(root/raw['artifact_file']).is_file(),raw
    reset=next(row for row in mapping['objects']if row['manuscript_label']=='tab:native-reset')
    assert reset['evidence_ids']==['E20']and reset['command_ids']==['cpu_E20_recorded_audit']
    assert b.read(root/'metadata/evidence/E19/PUBLIC_RAW_ALLOWLIST_REVIEW.json')['members']==248
    clean=b.read(root/'metadata/evidence/E21/PUBLIC_RAW_ALLOWLIST_REVIEW.json')
    assert clean['members']==221 and clean['credential_pattern_hits']==0

def test_archive_hardlinks_become_regular_payloads_and_full_readback(tmp_path):
    root=tmp_path/'fake_snapshot';root.mkdir();(root/'one.txt').write_bytes(b'literal archival fixture\n')
    os.link(root/'one.txt',root/'two.txt')
    rows={name:f.pin(root/name)for name in('one.txt','two.txt')}
    manifest={'version':'0.5.0','manifest_excludes_itself':True,'file_count':2,'files':rows}
    (root/'MANIFEST.json').write_text(json.dumps(manifest))
    archive=tmp_path/'code.tgz';receipt=tmp_path/'receipt.json';report=f.freeze(root,archive,receipt)
    assert report['members']==3 and report['all_members_regular']is True
    with tarfile.open(archive)as tar:assert all(m.isfile()and not m.linkname for m in tar)
    with pytest.raises(RuntimeError,match='fresh'):f.freeze(root,archive,receipt)
    (root/'unlisted.txt').write_text('extra')
    with pytest.raises(RuntimeError,match='surface'):f.verify_snapshot(root)
