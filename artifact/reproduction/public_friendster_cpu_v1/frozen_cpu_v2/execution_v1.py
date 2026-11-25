"""Exact extracted original functions; only operational module imports adapted."""
from cpu_common_v1 import *
import subprocess,signal,time
def acquire_lock(path,token):
    with path.open('x')as f:f.write(json.dumps(token,sort_keys=True)+'\n')

def release_lock(path,token,quiescent):
    require(quiescent,'never clear timing lock before owned tree quiescence')
    require(path.is_file()and not path.is_symlink()and read(path)==token,'owned timing lock changed')
    path.unlink()

def cleanup_owned(supervisor,timeout=15):
    deadline=time.monotonic()+timeout
    for sig in(signal.SIGTERM,signal.SIGKILL):
        groups=supervisor.owned_groups(os.getpid());supervisor.signal_confirmed(groups,sig)
        until=min(deadline,time.monotonic()+5)
        while time.monotonic()<until and supervisor.owned_groups(os.getpid()):time.sleep(.1)
    return not supervisor.owned_groups(os.getpid())

def guarded_run(supervisor,memory,argv,log,cwd,timeout):
    abort={'reason':None};last=[0.];errors=[0]
    guard=log.with_suffix('.guard.jsonl')
    def clock():return time.monotonic()+(10**9 if abort['reason']else 0)
    def groups(owner):
        found=supervisor.owned_groups(owner)
        if time.monotonic()-last[0]>=2:
            last[0]=time.monotonic()
            try:
                sample=memory.sample_tree(owner);decision=memory.budget_decision(sample,180*GIB,memory.cgroup_memory())
                abort['reason']=abort['reason']or decision['abort_reason'];errors[0]=0
                row={'unix':time.time(),'tree_pss_bytes':sample.get('tree_pss_bytes'),'pss_complete':sample.get('pss_complete'),
                     'membership_stable':sample.get('membership_stable'),'max_process_rss_bytes':sample.get('max_process_rss_bytes'),'decision':decision}
            except Exception as e:
                errors[0]+=1;row={'unix':time.time(),'sampling_error':type(e).__name__+': '+str(e)[:300]}
                if errors[0]>=3:abort['reason']='three consecutive external V3 sampling failures'
            with guard.open('a')as f:f.write(json.dumps(row)+'\n')
            if guard.stat().st_size>8<<20:abort['reason']='bounded guard-log8MiB limit'
        if log.exists()and log.stat().st_size>4<<20:abort['reason']='bounded stdout-log4MiB limit'
        return found
    def popen(*args,**kwargs):
        kwargs['env'].update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
        return subprocess.Popen(*args,**kwargs)
    try:result=supervisor.run_owned(argv,log,cwd=cwd,timeout=timeout,clock=clock,popen=popen,groups_reader=groups)
    except BaseException:
        cleanup_owned(supervisor);raise
    result.update(guard_abort_reason=abort['reason'],guard_policy='unchanged stable_pss_individual_rss_cgroup90_v3 externally sampled at2s;180GiB; CPU-only')
    save(log.with_suffix('.exit.json'),result);return result
