from pathlib import Path
import numpy as np
import pytest
import native_adapter_v1 as adapter
from build_native_v1 import build
from native_adapter_v1 import NativeCSR,ordered_oracle,pack_local,canonical_edges,drnl
from fixtures_v1 import data,bfs_labels

@pytest.fixture(scope='session')
def library(tmp_path_factory):
    lib,receipt=build(tmp_path_factory.mktemp('compiled_v1'))
    return lib

@pytest.mark.parametrize('mode',['reference_full_clear','persistent_touched_reset'])
def test_actual_compiled_ordered_outputs_eids_arbitrary_roots_and_independent_drnl(library,mode):
    ptr,col,cases=data()
    with NativeCSR(ptr,col,library,mode,max_assoc_bytes=1024)as native:
        for ids,roots in cases:
            nodes=np.asarray(ids,dtype=np.int64);expected=ordered_oracle(ptr,col,nodes)
            with native.induce(nodes)as result:
                for actual,oracle in zip((result.rows,result.cols,result.edge_ids),expected):np.testing.assert_array_equal(actual,oracle)
                a=pack_local(result.rows,result.cols,len(nodes),roots)
                edges=canonical_edges(a)
                assert not a[roots[0],roots[1]] and not a[roots[1],roots[0]]
                np.testing.assert_array_equal(drnl(a,roots),bfs_labels(edges,len(nodes),roots))
            assert native.live_touched==0
        assert native.initialization['persistent_assoc_bytes_per_worker']==(len(ptr)-1)*8*(mode=='persistent_touched_reset')

def test_successive_disjoint_sets_do_not_leak_old_mapping(library):
    ptr,col,_=data()
    with NativeCSR(ptr,col,library,max_assoc_bytes=1024)as native:
        for ids in ([0,1,2],[3,4],[4,0,8,2,1,3],[3,4],[8,7],[0,2,3]):
            nodes=np.asarray(ids,dtype=np.int64)
            with native.induce(nodes)as out:
                for actual,expected in zip((out.rows,out.cols,out.edge_ids),ordered_oracle(ptr,col,nodes)):np.testing.assert_array_equal(actual,expected)
            assert native.live_touched==0

@pytest.mark.parametrize('after',[1,2,3])
def test_injected_partial_touch_failure_resets_state_and_retry_is_exact(library,after):
    ptr,col,_=data();nodes=np.asarray([3,0,1],dtype=np.int64)
    with NativeCSR(ptr,col,library,max_assoc_bytes=1024)as native:
        with pytest.raises(RuntimeError,match='injected'):native.induce(nodes,fixture_fail_after_touches=after)
        assert native.live_touched==0
        with native.induce(nodes)as out:
            for a,b in zip((out.rows,out.cols,out.edge_ids),ordered_oracle(ptr,col,nodes)):np.testing.assert_array_equal(a,b)

def test_scan_failure_after_all_touches_resets_and_does_not_poison_retry(library):
    ptr,col,_=data();col=col.copy();saved=col[0];col[0]=999;nodes=np.asarray([0,1,2],dtype=np.int64)
    with NativeCSR(ptr,col,library,max_assoc_bytes=1024)as native:
        with pytest.raises(RuntimeError,match='neighbor outside'):native.induce(nodes)
        assert native.live_touched==0;col[0]=saved
        with native.induce(nodes)as out:np.testing.assert_array_equal(out.edge_ids,ordered_oracle(ptr,col,nodes)[2])

@pytest.mark.parametrize('ids',[[0,0],[0,9],[-1,2],[999999,0],[9,0,1],[1,9,0]])
def test_invalid_ids_rejected_before_dirty_state(library,ids):
    ptr,col,_=data()
    with NativeCSR(ptr,col,library,max_assoc_bytes=1024)as native:
        with pytest.raises(RuntimeError):native.induce(np.asarray(ids,dtype=np.int64))
        assert native.live_touched==0

def test_full_N_association_budget_checked_even_for_ephemeral_arm(library):
    ptr,col,_=data()
    for mode in ('reference_full_clear','persistent_touched_reset'):
        with pytest.raises(RuntimeError,match='byte budget'):NativeCSR(ptr,col,library,mode,max_assoc_bytes=71)

def test_graph_width_conversion_is_never_implicit(library):
    ptr,col,_=data()
    with pytest.raises(ValueError,match='already'):NativeCSR(ptr,col.astype(np.int32),library)

def test_root_positions_are_explicit_and_result_arrays_can_be_detached(library):
    ptr,col,_=data();nodes=np.asarray([4,0,2,1,3],dtype=np.int64)
    with NativeCSR(ptr,col,library,max_assoc_bytes=1024)as native:
        with native.induce(nodes)as result:detached=result.copy_arrays()
        np.testing.assert_array_equal(detached[2],ordered_oracle(ptr,col,nodes)[2])
        for roots in ((0,0),(0,5),(True,1)):
            with pytest.raises(ValueError,match='roots|root positions'):pack_local(detached[0],detached[1],len(nodes),roots)

def test_inherited_preinitialized_state_cannot_hide_per_worker_initialization(library,monkeypatch):
    ptr,col,_=data()
    with NativeCSR(ptr,col,library,max_assoc_bytes=1024)as native:
        with monkeypatch.context()as patch:
            patch.setattr(adapter.os,'getpid',lambda:native.owner_pid+1)
            with pytest.raises(RuntimeError,match='after fork'):native.induce(np.asarray([0,1],dtype=np.int64))

def test_closed_state_does_not_expose_freed_native_pointer(library):
    ptr,col,_=data();native=NativeCSR(ptr,col,library,max_assoc_bytes=1024);native.close();native.close()
    with pytest.raises(RuntimeError,match='closed'):_=native.live_touched
