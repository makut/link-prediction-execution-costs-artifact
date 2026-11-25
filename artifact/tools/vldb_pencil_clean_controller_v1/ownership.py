"""Outer ownership cleanup survives partial initialization and filesystem errors."""
from pathlib import Path
import fcntl,json,os,signal,sys,time,uuid
from contracts import HERE,read,save
from runner import Runner,load

class Ownership:
    def __init__(self,args,admission,deadline):
        self.args=args;self.admission=admission;self.deadline=deadline
        self.lock_path=args.base/'PENCIL_CLEAN_CONSUMER_EXCLUSIVE_GPU.lock'
        self.lock=None;self.lock_owned=False;self.supervisor=None;self.runner=None;self.handlers={}
        self.out_created=False;self.quiescent=None;self.closed=False
        self.stopped=False;self.stop_deadline=None
        self.token={'nonce':uuid.uuid4().hex,'pid':os.getpid(),'pod_uid':admission['pod_uid'],
            'source_manifest_sha256':args.source_manifest_sha256}
    def acquire(self):
        # Install a Python stop handler BEFORE holding a lease or importing
        # helper modules, so initialization stop also reaches outer finally.
        for sig in(signal.SIGTERM,signal.SIGINT):self.handlers[sig]=signal.signal(sig,self.request_stop)
        self.lock=self.lock_path.open('x');self.lock_owned=True
        self.lock.write(json.dumps(self.token)+'\n');self.lock.flush();fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.supervisor=load(HERE/'third_party/owned_supervision.py','clean_generic_supervision_for_owner')
        memory=load(HERE/'third_party/memory_guard_v3.py','clean_unchanged_memory_for_owner')
        self.runner=Runner(self.args.out,self.deadline,memory_gib=self.args.memory_guard_gib,supervisor=self.supervisor,memory=memory)
        return self.runner
    def request_stop(self,signum=None,frame=None):
        self.stopped=True
        if self.stop_deadline is None:self.stop_deadline=min(self.deadline,time.monotonic()+110)
        if self.runner is not None:return self.runner.request_stop(signum,frame)
        from runner import StopRequested
        raise StopRequested('stop during controller/helper initialization')
    def record_initialization_failure(self,error):
        if self.args.out.is_dir():
            try:save(self.args.out/'CONTROLLER_FAILURE.json',{'type':type(error).__name__,'message':str(error)[:1000],
                'initialization_completed':self.out_created,'partial_local_evidence_retained':True,
                'runner_initialized':self.runner is not None,'stop_requested':self.stopped or bool(self.runner and self.runner.stopped)})
            except BaseException:pass  # Preserve the original error; an unwritable output cannot claim success.
        print(json.dumps({'status':'CONTROLLER_FAILED','type':type(error).__name__,'message':str(error)[:500]}),file=sys.stderr)
    def close(self):
        if self.closed:return self.quiescent
        try:
            if self.runner is not None:self.quiescent=self.runner.cleanup_owned(10)
            elif self.supervisor is not None:self.quiescent=not self.supervisor.owned_groups(os.getpid())
            else:self.quiescent=True  # No phase/child can start before the reviewed helper is loaded.
        except BaseException:self.quiescent=False
        finally:
            for sig,old in self.handlers.items():
                try:signal.signal(sig,old)
                except BaseException:pass
            try:
                if self.lock_owned and self.quiescent is True and self.lock_path.exists():
                    try:
                        if read(self.lock_path,64<<10)==self.token:self.lock_path.unlink()
                    except BaseException:pass  # Never remove a replaced or unreadable ownership marker.
            finally:
                if self.lock is not None:
                    try:self.lock.close()
                    except BaseException:pass
                self.closed=True
        # A nonquiescent or uncertain tree intentionally keeps the O_EXCL path.
        return self.quiescent
