import copy,json
from pathlib import Path
import numpy as np
import pytest
from build_native_v1 import build
from fixtures_v1 import data
from native_adapter_v1 import ordered_oracle,pack_local,canonical_edges,drnl,digest
from prepare_replay_v1 import prepare,load_selected,choose,sha
from run_parent_replay_v1 import main as parent,admission
import run_parent_replay_v1 as parent_module
from prepare_native_only_v1 import main as convert

def archive_fixture(tmp_path,count=128):
    ptr,col,_=data();nodes=np.asarray([0,1,4,2,3,8],dtype=np.int64)
    raw=ordered_oracle(ptr,col,nodes);a=pack_local(raw[0],raw[1],len(nodes));rows=[]
    for i in range(count):
        rows.append({'nodes':nodes.tolist(),'pair':nodes[:2].tolist(),'pair_index':i,'workload':'literal_fixture',
            'sampler':'literal_fixture','cap':50,'virtual_graph_nodes':9,'node_set_sha256':digest(nodes),
            'canonical_edges_sha256':digest(canonical_edges(a)),'drnl_sha256':digest(drnl(a)),
            'sum_out_degrees':sum(int(ptr[u+1]-ptr[u])for u in nodes),'induced_directed_entries':int(a.nnz)})
    path=tmp_path/'literal_archive.json';path.write_text(json.dumps({'provenance':{'graph':{'nodes':9,'stored_entries':len(col)}},
        'configuration':{'stock_index_dtype':'int32','fixture':True},'records':rows}))
    native=tmp_path/'native';native.mkdir();ptr.tofile(native/'rowptr.i8');col.tofile(native/'col.i8')
    (native/'manifest.json').write_text(json.dumps({'graph':{'n':9,'m':len(col),'source':'literal_fixture_only'},
        'conversion_seconds':0,'rowptr_sha256':sha(native/'rowptr.i8'),'col_sha256':sha(native/'col.i8')}))
    return path,native

def test_replay_selects_strata_without_changing_original_S(tmp_path):
    archive,_=archive_fixture(tmp_path,768);data=json.loads(archive.read_text())
    for i,row in enumerate(data['records']):row.update(sampler='s'+str(i%3),cap=50 if(i//3)%2==0 else 200,workload='w'+str((i//6)%2))
    archive.write_text(json.dumps(data));plan=prepare(archive,128)
    assert sum(plan['stratum_counts'].values())==128 and len(plan['stratum_counts'])==12
    assert max(plan['stratum_counts'].values())-min(plan['stratum_counts'].values())<=1
    selected=load_selected(plan,archive)
    assert len(selected)==128 and [i for i,_ in selected]==sorted(i for i,_ in selected)
    assert all(r['nodes']==data['records'][i]['nodes']for i,r in selected)
    assert prepare(archive,128)==plan

def test_replay_archive_mutation_and_fingerprint_change_rejected(tmp_path):
    archive,_=archive_fixture(tmp_path);plan=prepare(archive,128)
    bad=copy.deepcopy(plan);bad['selected'][0]['drnl_sha256']='0'*64
    with pytest.raises(ValueError,match='fingerprint'):load_selected(bad,archive)
    archive.write_text(archive.read_text()+' ')
    with pytest.raises(ValueError,match='identity'):load_selected(plan,archive)

def test_admission_requires_terminal_and_absent_pipeline_lock(tmp_path):
    marker=tmp_path/'COMPLETE.json';marker.write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError,match='terminal'):admission(marker)
    marker.write_text(json.dumps({'complete':True}));assert admission(marker)['sha256']==sha(marker)
    (tmp_path/'PIPELINE_SUITE_RUNNING.lock').write_text('fixture')
    with pytest.raises(ValueError,match='overlapping'):admission(marker)

def test_parent_python_interruption_reaps_only_owned_worker(tmp_path,monkeypatch):
    class Child:
        pid=123456789
        calls=0
        def wait(self,timeout):
            self.calls+=1
            if self.calls==1:raise KeyboardInterrupt
            return -15
    child=Child();signals=[]
    monkeypatch.setattr(parent_module.subprocess,'Popen',lambda *args,**kwargs:child)
    monkeypatch.setattr(parent_module.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        parent_module.owned_run(['literal_fixture_only'],tmp_path/'interrupted.log',10)
    assert signals==[(child.pid,parent_module.signal.SIGTERM)] and child.calls==2

def test_actual_parent_two_fresh_native_workers_literal_fixture_only(tmp_path):
    archive,native=archive_fixture(tmp_path);lib,_=build(tmp_path/'build')
    marker=tmp_path/'COMPLETE.json';marker.write_text(json.dumps({'complete':True,'scope':'literal_fixture_only'}))
    out=tmp_path/'parent'
    code=parent(['--archive',str(archive),'--count','128','--output-dir',str(out),'--execute',
        '--native-dir',str(native),'--library',str(lib),'--admission-marker',str(marker),
        '--max-assoc-gib','0.001','--max-peak-gib','2','--warmup','0','--repeats','1','--timeout-seconds','30'])
    assert code==0
    terminal=json.loads((out/'COMPLETE.json').read_text());assert terminal['success']is True
    assert len({v['exit']['pid']for v in terminal['arms']})==2
    for mode,persistent in [('reference_full_clear',0),('persistent_touched_reset',72)]:
        report=json.loads((out/(mode+'.json')).read_text())
        assert report['record_count']==128 and report['all_executed_outputs_exact']is True
        assert report['state_initialization']['persistent_assoc_bytes_per_worker']==persistent
        assert all(r['exact_ordered_rows_cols_and_original_eids']for r in report['records'])
    with pytest.raises(FileExistsError):parent(['--archive',str(archive),'--count','128','--output-dir',str(out)])

def test_converter_reuses_hash_bound_unmodified_external_converter_on_tiny_graph(tmp_path):
    root=Path(__file__).resolve().parents[1]
    external=root/'vldb_literature/external_operator_benchmark.py'
    pinned=json.loads((root/'vldb_literature/external_operator_v1_sources.manifest.json').read_text())
    expected=next(x['sha256']for x in pinned['files']if x['path']=='external_operator_benchmark.py')
    assert sha(external)==expected
    ptr,col,_=data();source=tmp_path/'source';source.mkdir();ptr[:-1].tofile(source/'edge_starts.bin')
    with(source/'edge_ends.bin').open('wb')as f:f.write(len(col).to_bytes(8,'little',signed=True));f.write(col.astype('<i4').tobytes())
    lib,_=build(tmp_path/'build');marker=tmp_path/'COMPLETE.json';marker.write_text(json.dumps({'complete':True}))
    out=tmp_path/'conversion.json';native=tmp_path/'rebuilt_native'
    assert convert(['--external-helper',str(external),'--external-helper-sha256',expected,'--graph-path',str(source),
        '--native-dir',str(native),'--output',str(out),'--execute','--library',str(lib),'--admission-marker',str(marker),
        '--max-native-gib','0.001','--max-peak-gib','2','--max-assoc-gib','0.001'])==0
    report=json.loads(out.read_text());assert report['status']=='complete'
    assert report['runtime_native_content_verification']['status']=='pass'
    assert sha(external)==expected
