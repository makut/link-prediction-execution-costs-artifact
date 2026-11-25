"""Corruption tests use FAKE observations; no actual performance evidence."""
from pathlib import Path
import copy, io, json, subprocess, sys, tarfile, tempfile
import pytest
import audit_public_fr_cpu_v1 as a
import synthetic_fixtures as f

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'work/vldb_public_fr_cpu_v2_sources.tgz'
SYSTEMS=ROOT/'outputs/vldb_public_systems_evidence_20261125/lpp_public_systems_evidence_v1.tgz'
WHEELS=ROOT/'work/vldb_cpu_native_wheels_pt22_v1/pt22_native_wheels_v1.tgz'

@pytest.fixture(scope='module')
def anchors():return a.load_anchors(SOURCE,SYSTEMS,WHEELS)

@pytest.fixture
def fx(anchors):
    with tempfile.TemporaryDirectory(prefix='explicit_synthetic_cpu_audit_')as temp:
        yield f.make_fixture(Path(temp)/'fake',anchors)

def result(fx):return a.audit_export(fx['export_dir'],fx['root_launch_file'],fx['anchors'],synthetic=True)
def change(path,fn):
    value=f.read(path);fn(value);f.save(path,value)
def rebundle(fx):
    # Rebind transport hashes deliberately: numeric/source checks must reject
    # adversarial but internally consistent evidence, not just a stale hash.
    p=fx['results']/'native/COMPLETE.json'
    change(p,lambda d:[row.update(report_sha256=a.sha(fx['results']/'native'/(row['mode']+'.json')))for row in d['arms']])
    f.refresh(fx)
def res(fx,name):return fx['results']/name

def sync_outer(fx,export):
    receipts={p.relative_to(fx['export_dir']/'remote_receipts').as_posix():p.read_bytes()for p in(fx['export_dir']/'remote_receipts').rglob('*')if p.is_file()}
    export['remote_receipts']={n:f.identity(raw)for n,raw in receipts.items()}
    for view in(export['before_view'],export['after_view']):view['receipt_sha256']={n:f.digest(raw)for n,raw in receipts.items()}
    f.save(fx['export_dir']/'export.json',export)

def test_explicit_fake_roundtrip_and_independent_raw_medians(fx):
    report=result(fx)
    assert report['status']=='PASS_SYNTHETIC_FIXTURE'and report['scientific_evidence_eligible']is False
    assert report['synthetic_fixture']=='NOT_SCIENTIFIC_EVIDENCE'
    assert report['inventory']['source_files']==39
    for name,expected in fx['expected'].items():
        row=report['raw_recomputed_summaries'][name]
        assert row['records']==128
        assert row['sum_per_case_medians_seconds']['edge_seconds']==expected
        assert row['sum_per_case_medians_seconds']['edge_plus_drnl_seconds']==pytest.approx(expected+25.6)
    assert report['ratios_of_sums_per_case_medians']['native_reference_full_clear_over_native_persistent_touched_reset']['edge_seconds']==2
    assert report['ratios_of_sums_per_case_medians']['native_upstream_torch_sparse_direct_over_native_persistent_touched_reset']['edge_seconds']==4
    assert report['graph_model_or_native_library_loads']==report['remote_actions']==0

def test_production_api_rejects_flag(fx):
    with pytest.raises(RuntimeError,match='synthetic fixture'):a.audit_export(fx['export_dir'],fx['root_launch_file'],fx['anchors'])

@pytest.mark.parametrize('case',['full_sha','full_size','full_stat','train_dimensions','native_sha','native_sentinel','native_size','conversion_budget','literal_fixture','upstream_fixture','cpp_flags','kernel_signature','numba_source','numba_host','wrong_order','wrong_node_hash','missing_case','missing_repeat','zero','negative','bool','raw_larger_than_edge','ordered_eid_hash','false_exact','method_order','plus_arithmetic','isolation_origin','wheel_binary','wrapper_blob','numerical_origin','site_precedence','changed_base','extra_native','cuda','thread','native_parent_order','native_library_binding','native_helper_missing','native_child_timeout','converter_timeout','native_helper_config','parent_timeout','false_science','bad_guard','guard_sampling_failure','empty_guard','only_errors_guard','bad_exit','phase_timeout','missing_timeout','negative_timeout','init_mode','init_pid','init_version','init_nodes','init_policy','upstream_policy','job_denominator'])
def test_rehashed_corrupt_scientific_evidence_is_rejected(fx,case):
    native='native/reference_full_clear.json';numba='friendster_neighbors32.json'
    def alter(name,fn):change(res(fx,name),fn)
    if case=='full_sha':alter('full_input_before.json',lambda d:next(iter(d['files'].values())).update(sha256='0'*64))
    elif case=='full_size':alter('full_input_before.json',lambda d:next(iter(d['files'].values())).update(bytes=1))
    elif case=='full_stat':alter('full_input_after.json',lambda d:next(iter(d['files'].values())).update(inode=999))
    elif case=='train_dimensions':alter('full_input_before.json',lambda d:d.update(stored_entries=2889707416))
    elif case=='native_sha':alter('native/native_conversion_receipt.json',lambda d:d['conversion'].update(col_sha256='0'*64))
    elif case=='native_sentinel':alter('native/native_conversion_receipt.json',lambda d:d['conversion'].update(sentinel_added=False))
    elif case=='native_size':alter(native,lambda d:d['native_setup']['runtime_files'][0].update(bytes=8*a.N))
    elif case=='conversion_budget':alter('native/native_conversion_receipt.json',lambda d:d['estimate'].update(conservative_peak_bytes=1))
    elif case=='literal_fixture':alter(native,lambda d:d['actual_native_admission'].update(literal_cases=3))
    elif case=='upstream_fixture':alter('native/upstream_native_admission.json',lambda d:d['records'][0].update(drnl_sha256='0'*64))
    elif case=='cpp_flags':alter('build_standalone_native.log',lambda d:d['argv'].__setitem__(2,'-O0'))
    elif case=='kernel_signature':alter(numba,lambda d:next(iter(d['compiled_kernels'].values())).clear())
    elif case=='numba_source':alter(numba,lambda d:next(iter(d['source_bindings'].values())).update(sha256='0'*64))
    elif case=='numba_host':alter(numba,lambda d:d['host'].update(boot_id='changed'))
    elif case=='wrong_order':alter(native,lambda d:d['records'].reverse())
    elif case=='wrong_node_hash':alter(numba,lambda d:d['records'][0].update(node_set_sha256='0'*64))
    elif case=='missing_case':alter(native,lambda d:d['records'].pop())
    elif case=='missing_repeat':alter(native,lambda d:d['records'][0]['edge_seconds'].pop())
    elif case in('zero','negative','bool'):alter(native,lambda d:d['records'][0]['edge_seconds'].__setitem__(0,{'zero':0,'negative':-1,'bool':True}[case]))
    elif case=='raw_larger_than_edge':alter(native,lambda d:d['records'][0]['raw_operator_seconds'].__setitem__(0,999.))
    elif case=='ordered_eid_hash':alter(native,lambda d:d['records'][0]['ordered_output_sha256'].__setitem__(2,'0'*64))
    elif case=='false_exact':alter(numba,lambda d:d['records'][0]['methods']['scan'].update(exact_drnl=False))
    elif case=='method_order':alter(numba,lambda d:d['records'][0].update(method_order=['scan','scan']))
    elif case=='plus_arithmetic':alter(numba,lambda d:d['records'][0]['methods']['scan']['edge_plus_drnl_seconds'].__setitem__(0,999.))
    elif case in('isolation_origin','wheel_binary','wrapper_blob','numerical_origin','site_precedence','extra_native','cuda','thread'):
        def mutate(d):
            if case=='isolation_origin':d['native']['torch_sparse']['origin']='/opt/python/foreign/__init__.py'
            elif case=='wheel_binary':next(iter(d['native']['torch_sparse']['binaries'].values()))['sha256']='0'*64
            elif case=='wrapper_blob':d['native']['torch_sparse']['saint_wrapper']['git_blob']='0'*40
            elif case=='numerical_origin':d['numerical_modules']['numpy']['origin']='/tmp/numpy.py'
            elif case=='site_precedence':d['package_site_path_order'].reverse()
            elif case=='extra_native':d['native']['torch_sparse']['binaries']['unapproved.so']={'path':'/tmp/unapproved.so','bytes':1,'sha256':'0'*64}
            elif case=='cuda':d['CUDA_VISIBLE_DEVICES']='0'
            else:d['numba_threads']=8
        for name in('isolated_environment.json','isolated_environment_after.json'):alter(name,mutate)
        # Rebind admission so the independent environment gate is exercised.
        alter('admission.json',lambda d:d.update(environment_probe_sha256=a.sha(res(fx,'isolated_environment.json'))))
    elif case=='changed_base':alter('base_environment_after.json',lambda d:d['packages'].update(numpy='2.0'))
    elif case=='native_parent_order':alter('native/parent_plan.json',lambda d:d['modes'].reverse())
    elif case=='native_library_binding':
        alter('native/runtime_bindings.json',lambda d:d.update({str(f.BUILD/'native_assoc_v1.so'):'0'*64}))
        alter('native/COMPLETE.json',lambda d:d.update(runtime_bindings_sha256=a.sha(res(fx,'native/runtime_bindings.json'))))
    elif case=='native_helper_missing':
        alter('native/runtime_bindings.json',lambda d:d.pop(str(f.SOURCE/'original/external_operator_benchmark.py')))
        alter('native/COMPLETE.json',lambda d:d.update(runtime_bindings_sha256=a.sha(res(fx,'native/runtime_bindings.json'))))
    elif case=='native_child_timeout':
        alter('native/reference_full_clear_exit.json',lambda d:d.update(timeout_seconds=4000))
        alter('native/COMPLETE.json',lambda d:next(row for row in d['arms']if row['mode']=='reference_full_clear')['exit'].update(timeout_seconds=4000))
    elif case=='converter_timeout':alter('native/native_conversion_exit.json',lambda d:d.update(timeout_seconds=2000))
    elif case=='native_helper_config':alter('native/upstream_torch_sparse_direct.json',lambda d:d['configuration'].update(external_helper_sha256='0'*64))
    elif case=='parent_timeout':alter('native/parent_plan.json',lambda d:d['configuration'].update(timeout_seconds=4000))
    elif case=='false_science':alter('SCIENTIFIC_TERMINAL.json',lambda d:d.update(success=False))
    elif case=='bad_guard':
        p=res(fx,'numba32.guard.jsonl');row=json.loads(p.read_text());row['decision']['abort_reason']='PSS budget';p.write_text(json.dumps(row)+'\n')
    elif case=='guard_sampling_failure':res(fx,'numba32.guard.jsonl').write_text(''.join(json.dumps({'sampling_error':'fail'})+'\n'for _ in range(3)))
    elif case=='empty_guard':res(fx,'numba32.guard.jsonl').write_text('')
    elif case=='only_errors_guard':res(fx,'numba32.guard.jsonl').write_text(json.dumps({'sampling_error':'fail'})+'\n')
    elif case=='bad_exit':
        alter('numba32.exit.json',lambda d:d.update(returncode=-9))
        alter('stages.json',lambda d:next(r for r in d['stages']if r['name']=='numba32')['exit'].update(returncode=-9))
    elif case in('phase_timeout','missing_timeout','negative_timeout'):
        def timeout(d):
            if case=='missing_timeout':d.pop('timeout_seconds')
            else:d.update(timeout_seconds=99999 if case=='phase_timeout'else -1)
        alter('numba32.exit.json',timeout)
        alter('stages.json',lambda d:timeout(next(r for r in d['stages']if r['name']=='numba32')['exit']))
    elif case.startswith('init_'):
        field,value={'init_mode':('mode','persistent_touched_reset'),'init_pid':('owner_pid',99999),
          'init_version':('version','other'),'init_nodes':('nodes',a.N-1),'init_policy':('initialization_policy','once per worker')}[case]
        alter(native,lambda d:d['state_initialization'].update({field:value}))
    elif case=='upstream_policy':alter('native/upstream_torch_sparse_direct.json',lambda d:d['state_initialization'].update(policy='persistent'))
    elif case=='job_denominator':alter('scientific_jobs.json',lambda d:d['jobs'][0].update(record_count=127))
    rebundle(fx)
    with pytest.raises((RuntimeError,KeyError,ValueError)):result(fx)

@pytest.mark.parametrize('case',['live','reuse','lock','root_launch','source_observed','library_observed','receipt_changed','false_complete'])
def test_outer_lifecycle_and_genuine_bytes_rejected(fx,case):
    export=f.read(fx['export_dir']/'export.json')
    if case in('live','reuse'):
        owner=export['before_view']['owner'];owner.update(exists=True,state='R'if case=='live'else'Z',start_ticks=fx['launch']['start_ticks']+(case=='reuse'))
    elif case=='lock':export['after_view']['timing_locks']['ADDITIONAL_TIMING_RUNNING.lock']=True
    elif case=='root_launch':fx['root_launch_file'].write_text('{}\n')
    elif case=='source_observed':export['before_view']['source_observed_sha256']='0'*64
    elif case=='library_observed':export['after_view']['standalone_library_observed']['sha256']='0'*64
    elif case=='receipt_changed':export['after_view']['receipt_sha256']['public_fr_cpu_v2_code/SOURCE_MANIFEST.json']='0'*64
    else:
        p=fx['export_dir']/'remote_receipts/results'/f.NAME/'COMPLETE.json';change(p,lambda d:d.update(scientific_complete=False))
        sync_outer(fx,export);export=f.read(fx['export_dir']/'export.json')
    f.save(fx['export_dir']/'export.json',export)
    with pytest.raises(RuntimeError):result(fx)

@pytest.mark.parametrize('case',['source_changed','source_missing','source_extra','duplicate','traversal','symlink','oversized'])
def test_exact_archive_inventory_and_unsafe_members_rejected(fx,case):
    if case.startswith('source_'):
        p=fx['extracted']/'sources/cpu_worker_v1.py'
        if case=='source_changed':p.write_bytes(p.read_bytes()+b'\n# changed\n')
        elif case=='source_missing':p.unlink()
        else:(fx['extracted']/'sources/unexpected.txt').write_text('extra')
        f.refresh(fx)
    else:
        path=fx['export_dir']/'final_snapshot.tgz'
        with tarfile.open(path,'r:gz')as old:rows=[(m,old.extractfile(m).read())for m in old]
        with tarfile.open(path,'w:gz',compresslevel=1)as new:
            for m,raw in rows:new.addfile(m,io.BytesIO(raw))
            member=tarfile.TarInfo(f.NAME+'/bad')
            if case=='duplicate':member.name=rows[0][0].name
            elif case=='traversal':member.name=f.NAME+'/../bad'
            elif case=='symlink':member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd'
            elif case=='oversized':member.size=a.MAX_MEMBER+1
            raw=b'X'*member.size
            new.addfile(member,io.BytesIO(raw))
        export=f.read(fx['export_dir']/'export.json');export['archive'].update(a.identity(path))
        for name in('ARCHIVE_READY.json','COMPLETE.json'):
            p=fx['export_dir']/'remote_receipts/results'/f.NAME/name
            change(p,lambda d:d.update(export['archive'])if name=='ARCHIVE_READY.json'else d.update(local_archive=export['archive']))
        sync_outer(fx,export)
    with pytest.raises(RuntimeError):result(fx)

@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_duplicate_or_nonfinite_json_refused(raw):
    with pytest.raises((RuntimeError,ValueError)):a.decode(raw)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),True,0,-1])
def test_nonfinite_boolean_and_nonpositive_timers_refused(value):
    with pytest.raises(RuntimeError):a.times({'edge_seconds':[.1,.2,value]},'edge_seconds')

def test_cli_rejects_synthetic_and_writes_no_ratios(fx):
    output=fx['export_dir'].parent/'cli_failure.json'
    cmd=[sys.executable,str(Path(a.__file__)),'--export-dir',str(fx['export_dir']),'--root-launch',str(fx['root_launch_file']),'--source-packet',str(SOURCE),'--historical-systems-asset',str(SYSTEMS),'--wheel-packet',str(WHEELS),'--output',str(output)]
    run=subprocess.run(cmd,capture_output=True,text=True)
    assert run.returncode==1
    report=f.read(output)
    assert report['status']=='FAIL_NO_ELIGIBLE_PERFORMANCE_RESULTS'and report['scientific_evidence_eligible']is False
    assert'ratios_of_sums_per_case_medians'not in report
    before=output.read_bytes();run=subprocess.run(cmd,capture_output=True,text=True)
    assert run.returncode!=0 and output.read_bytes()==before
    assert subprocess.run(cmd+['--synthetic'],capture_output=True,text=True).returncode==2
    protected=fx['export_dir']/'protected_new_report.json'
    run=subprocess.run(cmd[:-1]+[str(protected)],capture_output=True,text=True)
    assert run.returncode!=0 and not protected.exists()

def test_independent_selection_refuses_changed_order_and_train_graph(anchors):
    plan=copy.deepcopy(anchors['frozen_plan']);plan['selected'].reverse()
    with pytest.raises(RuntimeError):a.independent_selection(anchors['e1'],plan)
    e1=copy.deepcopy(anchors['e1']);e1['provenance']['graph']['stored_entries']=2889707416
    with pytest.raises(RuntimeError):a.independent_selection(e1,anchors['frozen_plan'])

def test_auditor_imports_only_standard_library():
    tree=__import__('ast').parse(Path(a.__file__).read_text())
    names=[n.module for n in tree.body if isinstance(n,__import__('ast').ImportFrom)]
    names += [alias.name for n in tree.body if isinstance(n,__import__('ast').Import)for alias in n.names]
    assert not any(name.split('.')[0]in('numpy','torch','numba','scipy','yt')for name in names)

def test_decompressed_trailing_data_is_bounded(tmp_path):
    import gzip
    path=tmp_path/'trailing.tgz'
    with gzip.open(path,'wb')as out:
        with tarfile.open(fileobj=out,mode='w|')as tar:
            entry=tarfile.TarInfo('a');entry.size=1;tar.addfile(entry,io.BytesIO(b'x'))
        out.write(b'0'*(2*a.MIB))
    with pytest.raises(RuntimeError,match='decompressed TAR stream'):a.bounded_tar(path,1024,1024,1)

@pytest.mark.parametrize('case',['iso_python','base_prefix','base_site_escape'])
def test_explicit_plain_base_and_isolated_environment_route(fx,case):
    files={p.relative_to(fx['results']).as_posix():p.read_bytes()for p in fx['results'].rglob('*')if p.is_file()}
    before_names=('base_environment_before.json','base_environment_after_install.json','base_environment_after.json')
    iso_names=('isolated_environment.json','isolated_environment_after.json')
    for name in(iso_names if case=='iso_python'else before_names if case=='base_prefix'else before_names+iso_names):
        row=a.decode(files[name])
        if case=='iso_python':row['python']='3.11.0'
        elif case=='base_prefix':row['prefix']='/tmp/another-python'
        else:row['base_sites']=['/tmp/foreign/site-packages']
        files[name]=f.encode(row)
    with pytest.raises(RuntimeError):a.environment(a.Data(files),fx['anchors'])
