"""Path/corruption tests on genuine immutable archival bytes, never new science."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import pytest
import audit_recorded_e19 as a

def hardlink_copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
def replace_test_link(path,data):
    # Never overwrite an inode shared with frozen author/source evidence.
    path=Path(path);path.unlink();path.write_bytes(data)

@pytest.fixture(scope='module')
def seeds():
    names=('E19_SOURCE_ROOT','E19_EXPORT','E19_EVIDENCE_ROOT','E19_ROOT_LAUNCH')
    if not all(os.environ.get(name)for name in names):pytest.skip('Genuine recorded E19 materializer inputs required; see README')
    return {name:Path(os.environ[name]).resolve()for name in names}

@pytest.fixture
def relocated(tmp_path,seeds):
    contract=a.dependency_contract();root=tmp_path/'unrelated relocated package α with spaces';export=tmp_path/'original recorded export β';output=tmp_path/'fresh archival audit γ';evidence=root/'metadata/evidence/E19'
    for row in contract['source_dependencies']:hardlink_copy(seeds['E19_SOURCE_ROOT']/row['path'],root/row['path'])
    for row in contract['historical_path_aliases']:hardlink_copy(seeds['E19_EVIDENCE_ROOT']/row['evidence_name'],evidence/row['evidence_name'])
    launch=evidence/'ROOT_LAUNCH.json';hardlink_copy(seeds['E19_ROOT_LAUNCH'],launch)
    for source in seeds['E19_EXPORT'].rglob('*'):
        if source.is_file():hardlink_copy(source,export/source.relative_to(seeds['E19_EXPORT']))
    # Relocate the adapter itself too, so its source has no author-folder import.
    tool=root/'tools/recorded_pencil_e19_v1'
    for name in('audit_recorded_e19.py','DEPENDENCIES.json'):hardlink_copy(a.HERE/name,tool/name)
    return {'root':root,'export':export,'output':output,'evidence':evidence,'launch':launch,'tool':tool,'contract':contract}

def invoke(v):
    command=[sys.executable,'-B',str(v['tool']/'audit_recorded_e19.py'),'--artifact-root',str(v['root']),'--export',str(v['export']),'--output',str(v['output']),'--root-launch',str(v['launch'])]
    return subprocess.run(command,capture_output=True,text=True,timeout=60)

def test_dependency_schema_pins_source_and_three_exact_historical_files():
    c=a.dependency_contract();assert len(c['source_dependencies'])==115 and len(c['recorded_export_sidecars'])==13 and len(c['historical_path_aliases'])==3
    assert c['recorded_archive']=={'bytes':9966534,'sha256':'7ab909e987af900f8ad5f9ea9a559e17e650f78891cb0bc99200a0fdb97fa36c'}

def test_exact_path_alias_constructor_does_not_touch_pathlib_or_broadly_rewrite(tmp_path):
    before=Path;old='/historical/author/known.json';target=tmp_path/'genuine.json';target.write_text('{}');mapper=a.ExactPathAliases({old:target},'/historical/author',(tmp_path,))
    assert mapper(old)==target and mapper.hits[old]==1 and Path is before and mapper(tmp_path/'other')==tmp_path/'other'
    with pytest.raises(a.InvalidRelocation):mapper('/historical/author/unlisted.json')
@pytest.mark.parametrize('name',['../escape','/absolute/path','a/../../escape'])
def test_unlisted_relative_escape_refused(tmp_path,name):
    with pytest.raises(a.InvalidRelocation):a.rooted_path(tmp_path,name)

def test_archival_python312_required_before_any_data_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(a.sys,'version_info',(3,10));out=tmp_path/'output'
    with pytest.raises(a.InvalidRelocation,match='Python3.12'):a.main(['--source-root',str(tmp_path/'source'),'--export',str(tmp_path/'export'),'--output',str(out),'--root-launch',str(tmp_path/'launch')])
    assert not out.exists()

def test_real_recorded_e19_roundtrip_arbitrary_path_and_exact_scientific_report(relocated):
    result=invoke(relocated);assert result.returncode==0,result.stderr
    report=json.load((relocated['output']/'audit.json').open());assert report['status']=='scientific_checks_passed'and report['completed_jobs']==48
    assert len(report['scientific_checks']['primary_online_bf16_rows'])==32 and len(report['scientific_checks']['supplemental_replay_bf16_rows'])==4
    receipt=json.load((relocated['output']/'ARCHIVAL_RELOCATION.json').open());assert receipt['all_core_addendum_science_source_bytes_unchanged']is True and receipt['Torch_imported']is False and receipt['new_observations_or_scientific_jobs']==0
    assert all(row['uses']>0 for row in receipt['exact_aliases'])
    if os.environ.get('E19_REFERENCE_AUDIT'):
        reference=Path(os.environ['E19_REFERENCE_AUDIT']);assert a.sha(reference)=='ec3d25768c1c9dd3b444b84ee3420521053472d007c567c383b0abcf612f5ec4'
        assert report['scientific_checks']==json.load(reference.open())['scientific_checks']

@pytest.mark.parametrize('change',['core_source','producer_source','first_failure','stderr','recovery','root_launch','raw_tar','export_sidecar','missing_failure','symlink_failure','dependency_contract','raw_extracted'])
def test_meaningful_corruption_rejected_without_author_fallback(relocated,change):
    v=relocated
    if change=='core_source':target=v['root']/'work/vldb_pencil_final_audit_v3/audit_pencil_v3_v1.py'
    elif change=='producer_source':target=v['root']/'work/vldb_pencil_queue_v3/bridge/vldb_pencil_bridge_v2/pencil_runtime.py'
    elif change=='first_failure':target=v['evidence']/'ORIGINAL_FIRST_FAILED_ATTEMPT.json'
    elif change=='stderr':target=v['evidence']/'ORIGINAL_FIRST_FAILURE_STDERR.txt'
    elif change=='recovery':target=v['evidence']/'RECOVERY_ADMISSION.json'
    elif change=='root_launch':target=v['launch']
    elif change=='raw_tar':target=v['export']/'final_snapshot.tgz'
    elif change=='export_sidecar':target=v['export']/'export.json'
    elif change=='dependency_contract':target=v['tool']/'DEPENDENCIES.json'
    elif change=='raw_extracted':target=v['export']/'extracted/results/pencil_bridge_v3/SCIENTIFIC_TERMINAL.json'
    else:target=v['evidence']/'ORIGINAL_FIRST_FAILED_ATTEMPT.json'
    if change=='missing_failure':target.unlink()
    elif change=='symlink_failure':original=target.read_bytes();target.unlink();other=target.with_name('other.json');other.write_bytes(original);target.symlink_to(other)
    else:replace_test_link(target,target.read_bytes()+b'CORRUPTED')
    result=invoke(v);assert result.returncode==2 and not v['output'].exists(),result.stdout+result.stderr

@pytest.mark.parametrize('where',['source','export','evidence'])
def test_source_export_and_original_anchor_output_overlap_refused(relocated,where):
    v=relocated;v['output']={'source':v['root'],'export':v['export'],'evidence':v['evidence']}[where]/'new_but_unsafe_output';result=invoke(v)
    assert result.returncode==2 and not v['output'].exists()
