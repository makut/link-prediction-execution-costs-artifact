"""Finite owned-child execution and unchanged memory-guard algorithm."""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,time
from common import HERE,GIB,require,save,clean

class StopRequested(RuntimeError):pass

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Runner:
    def __init__(self,out,deadline,*,memory_gib=96,supervisor=None,memory=None,gpu_visibility=None,clock=time.monotonic,sleep=time.sleep):
        self.gpu_visibility=gpu_visibility;self.out=Path(out);self.deadline=deadline;self.memory_gib=memory_gib;self.clock=clock;self.sleep=sleep
        self.supervisor=supervisor or load(HERE/'third_party/owned_supervision.py','clean_owned_supervision')
        self.memory=memory or load(HERE/'third_party/memory_guard_v3.py','clean_unchanged_memory_guard')
        self.stopped=False;self.stop_deadline=None;self.active=None;self.records=[];self.finalizing=False
    def request_stop(self,signum=None,frame=None):
        first=not self.stopped
        if first:self.stopped=True;self.stop_deadline=min(self.deadline,self.clock()+110)
        if first and self.finalizing and self.active is None:
            return  # Stop is recorded; only bounded local archive/one publisher may finish.
        raise StopRequested('stop signal received; no further scientific phase may start')
    def remaining(self,*,final=False):
        if self.stopped and not final:raise StopRequested('scientific work stopped')
        limit=min(self.deadline,self.stop_deadline)if self.stop_deadline is not None else self.deadline
        return limit-self.clock()
    def check(self,*,final=False):require(self.remaining(final=final)>0,'finite total/stop deadline reached')
    def cleanup_owned(self,seconds=10):
        end=min(self.clock()+seconds,self.deadline,self.stop_deadline if self.stop_deadline is not None else self.deadline)
        for sig in(signal.SIGTERM,signal.SIGKILL):
            self.supervisor.signal_confirmed(self.supervisor.owned_groups(os.getpid()),sig)
            until=min(end,self.clock()+min(4,seconds/2))
            while self.clock()<until and self.supervisor.owned_groups(os.getpid()):self.sleep(min(.1,max(0,until-self.clock())))
        return not self.supervisor.owned_groups(os.getpid())
    def size_and_disk(self):
        import shutil
        total=0
        for path in self.out.rglob('*'):
            require(not path.is_symlink(),'output symlink refused')
            if path.is_file():total+=path.stat().st_size
        require(total<=512<<20,'whole local output/cache budget512MiB exceeded; no truncation')
        require(shutil.disk_usage(self.out).free>=1*GIB,'oneGiB scratch reserve required')
    def run(self,job,*,final=False):
        self.check(final=final);name=job['name'];require(name.replace('_','').isalnum(),'safe unique phase name required')
        if final:raise ValueError('public runner starts no children during local finalization')
        if self.stopped and not final:raise StopRequested('no scientific job after stop')
        limit=min(job.get('timeout_seconds',1800),self.remaining(final=final)-(15 if final else 600))
        require(limit>=5,'phase deadline would consume finalization reserve')
        self.size_and_disk();log=self.out/(name+'.log');guard=self.out/(name+'.guard.jsonl')
        abort=[None];last=[-1e9];errors=[0];child=[None];self.active=name
        def clock():return self.clock()+(1e9 if abort[0] else 0)
        def groups(owner):
            found=self.supervisor.owned_groups(owner)
            if self.clock()-last[0]>=2:
                last[0]=self.clock()
                try:
                    sample=self.memory.sample_tree(owner);decision=self.memory.budget_decision(sample,int(self.memory_gib*GIB),self.memory.cgroup_memory())
                    abort[0]=abort[0]or decision['abort_reason'];errors[0]=0
                    row={'unix':time.time(),'tree_pss_bytes':sample.get('tree_pss_bytes'),'pss_complete':sample.get('pss_complete'),
                        'membership_stable':sample.get('membership_stable'),'max_process_rss_bytes':sample.get('max_process_rss_bytes'),'decision':decision}
                except Exception as error:
                    errors[0]+=1;row={'unix':time.time(),'sampling_error':type(error).__name__+': '+str(error)[:300]}
                    if errors[0]>=3:abort[0]='three consecutive external memory/resource sampling failures'
                with guard.open('a')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if guard.stat().st_size>8<<20:abort[0]='whole guard log exceeds8MiB'
            if log.exists()and log.stat().st_size>4<<20:abort[0]='whole stdout exceeds4MiB'
            try:self.size_and_disk()
            except Exception as error:abort[0]=abort[0]or 'local output/disk resource budget: '+str(error)[:200]
            return found
        def popen(*argv,**kwargs):
            kwargs['env'].update(CUDA_VISIBLE_DEVICES=self.gpu_visibility if job.get('gpu')else'',WANDB_MODE='disabled',
                HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
                VECLIB_MAXIMUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(HERE),
                NUMBA_CACHE_DIR=str(self.out/'cache/numba'/name),HF_HOME=str(self.out/'cache/hf'/name))
            child[0]=subprocess.Popen(*argv,**kwargs);return child[0]
        try:
            exit=self.supervisor.run_owned(job['argv'],log,cwd=HERE,timeout=limit,clock=clock,popen=popen,groups_reader=groups)
        except BaseException as error:
            quiescent=self.cleanup_owned(10);process=child[0]
            code=process.poll()if process is not None else None
            exit={'pid':process.pid if process is not None else None,'returncode':code,'timed_out':False,
                'timeout_seconds':limit,'owned_tree_quiescent':quiescent,'orphan_cleanup':not quiescent,
                'interrupted':True,'error':type(error).__name__+': '+str(error)[:300],
                'guard_abort_reason':abort[0],'exactly_one_visible_gpu':job.get('gpu',False),
                'guard_policy':'unchanged stable_pss_individual_rss_cgroup90_v3 sampled2s','memory_budget_gib':self.memory_gib}
            save(self.out/(name+'.exit.json'),exit,exclusive=True);self.records.append({'name':name,'exit':exit});raise
        finally:self.active=None
        exit.update(guard_abort_reason=abort[0],exactly_one_visible_gpu=job.get('gpu',False),
            guard_policy='unchanged stable_pss_individual_rss_cgroup90_v3 sampled2s',memory_budget_gib=self.memory_gib)
        save(self.out/(name+'.exit.json'),exit,exclusive=True);self.records.append({'name':name,'exit':exit})
        return exit
