"""Generic Linux owned-child supervision extracted from reviewed V3 source.
No remote storage, scheduler or experiment entrypoint is included.
"""
from pathlib import Path
import os, signal, subprocess, sys, time
REPO = Path(".")
def require(value, message):
    if not value: raise RuntimeError(message)

def enable_subreaper():
    """Orphaned row sessions remain descendants of this private queue process."""
    import ctypes
    require(sys.platform.startswith('linux'),'Linux owned-tree supervision required')
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,'cannot enable private child subreaper')


def process_identity(pid,proc=Path('/proc')):
    try:
        fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),'start_ticks':int(fields[19]),'state':fields[0]}
    except(OSError,ValueError,IndexError):return None


def owned_groups(owner_pid,proc=Path('/proc')):
    """Only observed descendants, including sessions adopted by our subreaper."""
    owner=process_identity(owner_pid,proc);require(owner is not None,'owned supervisor identity unavailable')
    pending=[owner_pid];seen=set();groups={}
    while pending:
        parent=pending.pop()
        if parent in seen:continue
        seen.add(parent)
        try:children=[int(x)for x in(proc/str(parent)/'task'/str(parent)/'children').read_text().split()]
        except(OSError,ValueError):continue
        for pid in children:
            current=process_identity(pid,proc)
            if current is None or current['ppid']!=parent:continue
            pending.append(pid)
            if current['state']!='Z'and current['pgid']>=2 and current['pgid']!=owner['pgid']:
                groups.setdefault(current['pgid'],[]).append(current)
    return groups


def signal_confirmed(groups,sig,*,reader=process_identity,killpg=os.killpg):
    signaled=[]
    for pgid,members in groups.items():
        for saved in members:
            now=reader(saved['pid'])
            if now is not None and now['state']!='Z'and now['start_ticks']==saved['start_ticks']and now['pgid']==pgid:
                try:killpg(pgid,sig);signaled.append(pgid)
                except ProcessLookupError:pass
                break
    return signaled


def run_owned(argv,log,*,cwd=REPO,timeout=7200,clock=time.monotonic,sleep=time.sleep,
              popen=subprocess.Popen,groups_reader=owned_groups,signaler=signal_confirmed):
    require(timeout>0,'no time remains for owned child')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    grace=min(10,timeout/4);soft_deadline=clock()+timeout-grace;hard_deadline=soft_deadline+grace
    with log.open('x')as stream:
        child=popen(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    close_fds=True,start_new_session=True)
        tracked={};timed_out=False;orphan_cleanup=False;signals=[]
        def observe():
            latest=groups_reader(os.getpid())
            for pgid,members in latest.items():
                saved=tracked.setdefault(pgid,[]);known={(m['pid'],m['start_ticks'])for m in saved}
                saved.extend(m for m in members if(m['pid'],m['start_ticks'])not in known)
            return latest
        while child.poll()is None:
            observe()
            if clock()>=soft_deadline:timed_out=True;break
            sleep(min(.5,max(0,soft_deadline-clock())))
        remaining=observe()
        orphan_cleanup=bool(child.poll()is not None and remaining)
        if timed_out or orphan_cleanup:
            signals+=signaler(tracked,signal.SIGTERM)
            cleanup_deadline=min(hard_deadline,clock()+grace)
            while clock()<cleanup_deadline and observe():sleep(min(.2,max(0,cleanup_deadline-clock())))
            observe();signals+=signaler(tracked,signal.SIGKILL)
        code=child.wait(timeout=10)
        quiescence_deadline=clock()+5
        while clock()<quiescence_deadline and groups_reader(os.getpid()):
            sleep(min(.1,max(0,quiescence_deadline-clock())))
        # Reap only our adopted direct children; never discover/signals by name.
        for directory in Path('/proc').iterdir()if Path('/proc').is_dir()else[]:
            if directory.name.isdigit()and int(directory.name)!=child.pid:
                value=process_identity(int(directory.name))
                if value and value['ppid']==os.getpid()and value['state']=='Z':
                    try:os.waitpid(value['pid'],os.WNOHANG)
                    except ChildProcessError:pass
        return {'pid':child.pid,'returncode':code,'timed_out':timed_out,'timeout_seconds':timeout,
                'orphan_cleanup':orphan_cleanup,'owned_pgids_signaled':sorted(set(signals)),
                'owned_tree_quiescent':not groups_reader(os.getpid())}


