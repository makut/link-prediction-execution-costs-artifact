"""Every transport test uses explicitly synthetic lifecycle observations."""
from pathlib import Path
import base64,copy,importlib.util,json,sys,tempfile
import pytest
import normalize_transport_v1 as n

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/vldb_public_fr_cpu_audit_v1'))
import audit_public_fr_cpu_v1 as a
import synthetic_fixtures as f
FAIL=ROOT/'outputs/vldb_public_fr_cpu_audit_20261125/ACTUAL_PUBLIC_FR_CPU_V2_AUDIT_v1.json'

@pytest.fixture(scope='module')
def anchors():return a.load_anchors(ROOT/'work/vldb_public_fr_cpu_v2_sources.tgz',ROOT/'outputs/vldb_public_systems_evidence_20261125/lpp_public_systems_evidence_v1.tgz',ROOT/'work/vldb_cpu_native_wheels_pt22_v1/pt22_native_wheels_v1.tgz')

@pytest.fixture
def original(anchors):
    with tempfile.TemporaryDirectory(prefix='explicit_synthetic_transport_')as temp:
        fx=f.make_fixture(Path(temp)/'original',anchors)
        export=f.read(fx['export_dir']/'export.json')
        for key,name in(('before_view','remote_before.json'),('after_view','remote_after.json')):
            view=export.pop(key)
            observation={'ready':True,'files':{rel:base64.b64encode((fx['export_dir']/'remote_receipts'/rel).read_bytes()).decode()for rel in export['remote_receipts']},
               'owner':view['owner'],'timing_locks':view['timing_locks'],'host':view['hostname'],
               'standalone_library_observed':view['standalone_library_observed'],'actual_cpu_pod_uid':'SYNTHETIC-UID-NOT-OBSERVED',
               'actual_node':'SYNTHETIC-NODE-NOT-OBSERVED','observed_unix':1.}
            f.save(fx['export_dir']/name,observation)
        export['root_launch_sha256']=a.sha(fx['root_launch_file'])
        f.save(fx['export_dir']/'export.json',export)
        fx['output']=Path(temp)/'derived'
        fx['expected']={name:a.sha(fx['export_dir']/name)for name in('export.json','remote_before.json','remote_after.json')}
        yield fx

def run(fx):return n.adapt(fx['export_dir'],fx['root_launch_file'],FAIL,fx['output'],fx['expected'])
def edit(fx,name,fn,rebind=False):
    path=fx['export_dir']/name;value=f.read(path);fn(value);f.save(path,value)
    if rebind:fx['expected'][name]=a.sha(path)

def test_transport_mapping_is_lossless_hardlinked_and_explicitly_synthetic(original):
    before={p.relative_to(original['export_dir']).as_posix():n.file_pin(p)for p in original['export_dir'].rglob('*')if p.is_file()}
    result=run(original);assert result['status']=='LOSSLESS_TRANSPORT_DERIVATIVE_CREATED_NO_AUDIT'
    adapted=f.read(original['output']/'export.json');old=f.read(original['export_dir']/'export.json')
    for key,value in old.items():assert adapted[key]==value
    assert adapted['synthetic_fixture']=='NOT_SCIENTIFIC_EVIDENCE'
    for key,name in(('before_view','remote_before.json'),('after_view','remote_after.json')):
        raw=f.read(original['export_dir']/name);view=adapted[key]
        assert view['hostname']==raw['host']
        for field in('owner','timing_locks','actual_cpu_pod_uid','actual_node','standalone_library_observed'):assert view[field]==raw[field]
        assert view['source_observed_sha256']==n.SOURCE_SHA and view['source_manifest_observed_sha256']==n.MANIFEST_SHA
        assert view['receipt_sha256']=={rel:a.sha(original['export_dir']/'remote_receipts'/rel)for rel in old['remote_receipts']}
    assert n.file_pin(original['output']/'final_snapshot.tgz')==n.file_pin(original['export_dir']/'final_snapshot.tgz')
    assert(original['output']/'final_snapshot.tgz').stat().st_ino==(original['export_dir']/'final_snapshot.tgz').stat().st_ino
    for name in old['remote_receipts']:
        assert(original['output']/'remote_receipts'/name).stat().st_ino==(original['export_dir']/'remote_receipts'/name).stat().st_ino
    after={p.relative_to(original['export_dir']).as_posix():n.file_pin(p)for p in original['export_dir'].rglob('*')if p.is_file()}
    assert before==after
    proof=f.read(original['output']/'TRANSPORT_DERIVATION_MANIFEST.json')
    assert proof['new_observations']==proof['scientific_or_timer_changes']==proof['audits_executed']==0
    assert proof['preserved_first_failure']['sha256']==n.FIRST_FAILURE_SHA
    # The frozen auditor can consume this synthetic derivative through its
    # explicit Python test API only; never actual scientific evidence.
    audit=a.audit_export(original['output'],original['root_launch_file'],original['anchors'],synthetic=True)
    assert audit['status']=='PASS_SYNTHETIC_FIXTURE'and audit['scientific_evidence_eligible']is False
    with pytest.raises(RuntimeError,match='fresh'):run(original)

@pytest.mark.parametrize('case',['export_pin','before_pin','after_pin','decoded_receipt','missing_receipt','source_byte','missing_observation','archive_byte','root_launch_byte','duplicate_json','not_ready','already_embedded'])
def test_corruption_refused_before_derivative_creation(original,case):
    if case.endswith('_pin'):
        name={'export_pin':'export.json','before_pin':'remote_before.json','after_pin':'remote_after.json'}[case];original['expected'][name]='0'*64
    elif case=='decoded_receipt':edit(original,'remote_before.json',lambda d:d['files'].update({n.RECEIPTS['build']:base64.b64encode(b'changed').decode()}),True)
    elif case=='missing_receipt':edit(original,'remote_after.json',lambda d:d['files'].pop(n.RECEIPTS['manifest']),True)
    elif case=='source_byte':
        path=original['export_dir']/'remote_receipts'/n.RECEIPTS['source'];path.write_bytes(path.read_bytes()+b'\n# changed\n')
        edit(original,'export.json',lambda d:d['remote_receipts'].update({n.RECEIPTS['source']:n.file_pin(path)}),True)
    elif case=='missing_observation':edit(original,'remote_before.json',lambda d:d.pop('standalone_library_observed'),True)
    elif case=='archive_byte':
        path=original['export_dir']/'final_snapshot.tgz';path.write_bytes(path.read_bytes()+b'changed')
    elif case=='root_launch_byte':original['root_launch_file'].write_text('{}\n')
    elif case=='duplicate_json':
        path=original['export_dir']/'remote_before.json';path.write_bytes(b'{"ready":true,"ready":false}');original['expected']['remote_before.json']=a.sha(path)
    elif case=='not_ready':edit(original,'remote_after.json',lambda d:d.update(ready=False),True)
    else:edit(original,'export.json',lambda d:d.update(before_view={}),True)
    with pytest.raises((RuntimeError,KeyError,ValueError)):run(original)
    assert not original['output'].exists()

def test_missing_optional_fields_are_not_inferred(original):
    for name in('remote_before.json','remote_after.json'):
        edit(original,name,lambda d:[d.pop(k)for k in('actual_cpu_pod_uid','actual_node')],True)
    run(original);export=f.read(original['output']/'export.json')
    for name in('before_view','after_view'):assert'actual_cpu_pod_uid'not in export[name]and'actual_node'not in export[name]

def test_changed_observed_pid_is_copied_and_frozen_auditor_rejects(original):
    edit(original,'remote_before.json',lambda d:d['owner'].update(pid=99999),True)
    run(original)
    export=f.read(original['output']/'export.json');assert export['before_view']['owner']['pid']==99999
    with pytest.raises(RuntimeError,match='same-owner'):a.audit_export(original['output'],original['root_launch_file'],original['anchors'],synthetic=True)

@pytest.mark.parametrize('target',['sidecar','archive'])
def test_original_change_during_link_is_detected_before_manifest(original,monkeypatch,target):
    real_link=n.os.link;changed=[False]
    watched=original['export_dir']/('remote_before.json'if target=='sidecar'else'final_snapshot.tgz')
    def link(source,destination):
        real_link(source,destination)
        if Path(source)==watched and not changed[0]:
            watched.write_bytes(watched.read_bytes()+b'changed during derivation');changed[0]=True
    monkeypatch.setattr(n.os,'link',link)
    with pytest.raises(RuntimeError,match='changed'):run(original)
    assert not(original['output']/'TRANSPORT_DERIVATION_MANIFEST.json').exists()
    assert not(original['output']/'export.json').exists()
