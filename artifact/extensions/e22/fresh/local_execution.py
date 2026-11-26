"""Local-only owned process supervision with unchanged sampled-memory policy."""
from pathlib import Path
import fcntl,importlib.util,json,os,shutil,signal,subprocess,time,uuid
from portable_common import *

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

class Execution:
    def __init__(self,args,L,bindings):
        self.args,self.L,self.bindings=args,L,bindings;self.jobs=[];self.token=None;self.lock=None;self.active=None;self.previous={};self.deadline=None
        self.supervisor=load(HERE/'owned_supervision.py','portable_cpu_owned_supervision')
        self.memory=load(FROZEN/'third_party/memory_guard_v3.py','portable_cpu_sampled_memory')
    def check_bindings(self):require(all(pin(p)==value for p,value in self.bindings.items()),'bound portable/frozen/environment source changed')
    def disk(self):
        require(shutil.disk_usage(self.L['root']).free>=int(1.5*GIB),'live1.5GiB scratch reserve required')
        size=0
        for p in self.L['out'].rglob('*'):
            require(not p.is_symlink(),'symlink result refused')
            if p.is_file():size+=p.stat().st_size
        require(size<=96*MIB,'whole raw result96MiB limit exceeded')
    def __enter__(self):
        require(sys.platform.startswith('linux'),'execution requires Linux; default plan is platform-independent')
        require(not self.L['root'].exists(),'fresh experiment root required')
        self.L['root'].parent.mkdir(parents=True,exist_ok=True)
        require(shutil.disk_usage(self.L['root'].parent).free>=8*GIB,'8GiB free conversion/output disk admission required')
        coord=self.args.coordination_dir;coord.mkdir(parents=True,exist_ok=True)
        require(not (coord/'PUBLIC_SYSTEMS_RUNNING.lock').is_symlink(),'shared flock must not be a symlink')
        self.lock=(coord/'PUBLIC_SYSTEMS_RUNNING.lock').open('a+')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.active=coord/'PUBLIC_SYSTEMS_ACTIVE.json'
            require(not os.path.lexists(self.active),'shared active/unknown predecessor retained; no stale-marker removal')
            self.supervisor.enable_subreaper();identity=self.supervisor.process_identity(os.getpid());require(identity,'actual controller PID/start required')
            self.token={'role':'citation2_cpu7','pid':os.getpid(),'start_ticks':identity['start_ticks'],'nonce':uuid.uuid4().hex,
              'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json')}
            with self.active.open('x')as stream:json.dump(self.token,stream,sort_keys=True);stream.write('\n')
            self.L['out'].mkdir(parents=True);self.L['base'].mkdir();self.L['cache'].mkdir()
            self.deadline=time.monotonic()+3600*self.args.max_hours
            for signum in(signal.SIGTERM,signal.SIGINT):
                self.previous[signum]=signal.getsignal(signum)
                signal.signal(signum,lambda sig,frame:(_ for _ in()).throw(KeyboardInterrupt('local controller stop '+str(sig))))
            return self
        except BaseException:
            # No child exists before __enter__ returns. Remove only our exact
            # token if it was created, never an inherited active marker.
            if self.token and self.active.is_file()and read(self.active)==self.token:self.active.unlink()
            for sig,handler in self.previous.items():signal.signal(sig,handler)
            self.lock.close();raise
    def run(self,name,argv,timeout):
        self.check_bindings();self.disk();require(read(self.active)==self.token,'shared active ownership changed')
        remaining=self.deadline-time.monotonic()-120;require(remaining>=30,'finite6hour deadline before phase')
        timeout=min(timeout,remaining);log=self.L['out']/(name+'.log');guard=self.L['out']/(name+'.guard.jsonl');abort=[None];last=[0.];errors=[0]
        def clock():return time.monotonic()+(10**9 if abort[0]else 0)
        def groups(owner):
            current=self.supervisor.owned_groups(owner)
            if time.monotonic()-last[0]>=2:
                last[0]=time.monotonic()
                try:
                    sample=self.memory.sample_tree(owner);decision=self.memory.budget_decision(sample,int(self.args.memory_budget_gib*GIB),self.memory.cgroup_memory())
                    abort[0]=abort[0]or decision['abort_reason'];errors[0]=0
                    row={'unix':time.time(),'tree_pss_bytes':sample.get('tree_pss_bytes'),'pss_complete':sample.get('pss_complete'),
                      'membership_stable':sample.get('membership_stable'),'max_process_rss_bytes':sample.get('max_process_rss_bytes'),'decision':decision}
                    self.disk()
                except Exception as exc:
                    errors[0]+=1;row={'unix':time.time(),'sampling_error':type(exc).__name__+': '+str(exc)[:300]}
                    if errors[0]>=3:abort[0]='three consecutive external V3 sampling failures'
                with guard.open('a')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if guard.stat().st_size>8*MIB:abort[0]='whole guard log8MiB limit'
            if log.exists()and log.stat().st_size>4*MIB:abort[0]='whole stdout log4MiB limit'
            return current
        def popen(*values,**kwargs):
            kwargs['env'].update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
              NUMBA_CACHE_DIR=str(self.L['cache']),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            return subprocess.Popen(*values,**kwargs)
        result=self.supervisor.run_owned([str(x)for x in argv],log,cwd=HERE,timeout=timeout,clock=clock,popen=popen,groups_reader=groups)
        result.update(guard_abort_reason=abort[0],guard_policy=self.memory.GUARD_POLICY,argv=[str(x)for x in argv])
        save(self.L['out']/(name+'.exit.json'),result);self.jobs.append({'name':name,'exit':result});save(self.L['out']/'stages.json',{'stages':self.jobs})
        self.check_bindings();self.disk()
        original=load_originals();require(original.clean(result),'owned phase failed/stopped/guarded/unquiescent: '+name)
        return result
    def cleanup(self):
        deadline=time.monotonic()+15
        for sig in(signal.SIGTERM,signal.SIGKILL):
            groups=self.supervisor.owned_groups(os.getpid());self.supervisor.signal_confirmed(groups,sig)
            until=min(deadline,time.monotonic()+5)
            while time.monotonic()<until and self.supervisor.owned_groups(os.getpid()):time.sleep(.1)
        return not self.supervisor.owned_groups(os.getpid())
    def __exit__(self,kind,value,trace):
        try:
            quiet=self.cleanup();owned=self.active.is_file()and not self.active.is_symlink()and read(self.active)==self.token
            success=kind is None and quiet and owned
            save(self.L['out']/'COMPLETE.json',{'controller_finished':True,'success':success,'owned_tree_quiescent':quiet,
              'failure':({'type':'OwnershipChanged','message':'shared active marker no longer matches'}if not owned else None if kind is None else{'type':kind.__name__,'message':str(value)[:1000]}),'jobs':self.jobs,'controller_identity':self.token,
              'source_manifest_sha256':sha(HERE/'SOURCE_MANIFEST.json'),'YT_writes':0,'remote_actions':0,'scope':'New local public rerun; no equivalence of hardware/timings asserted.'})
            if quiet and owned:self.active.unlink()
        finally:
            for sig,handler in self.previous.items():signal.signal(sig,handler)
            self.lock.close()
        if kind is None:require(success,'owned quiescence required')
        return False
