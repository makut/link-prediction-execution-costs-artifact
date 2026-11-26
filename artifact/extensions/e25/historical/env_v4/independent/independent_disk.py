"""Independent ENVv4 recorded disk guard checks; no recorded path is opened."""

GIB=1<<30
KEYS={'observed_monotonic_ns','path','device','bytes_available','bytes_total',
 'inodes_available','inodes_total','inodes_reported','mount_readonly','scope'}

def need(ok,message):
 if not ok:raise ValueError('Independent disk guard: '+message)
def counter(value):return type(value)is int and 0<=value<1<<128
def linux_major_minor(device):
 need(type(device)is int and 0<=device<1<<64,'recorded Linux dev_t range')
 major=((device>>8)&0xfff)|((device>>32)&0xfffff000)
 minor=(device&0xff)|((device>>12)&0xffffff00)
 return str(major)+':'+str(minor)

class Facts:
 def __init__(self):
  self.count=0;self.first=self.last=None;self.devices=set();self.paths=set()
  self.minimum_bytes=None;self.minimum_inodes=None;self.inodes_exposed=0;self.inodes_unreported=0
 def ingest(self,value):
  need(set(value)==KEYS,'complete raw observation schema')
  tick=value['observed_monotonic_ns'];need(type(tick)is int and tick>0 and(self.last is None or tick>self.last),'strictly ordered raw disk clocks')
  need(isinstance(value['path'],str)and 0<len(value['path'])<=2048 and counter(value['device']),'bounded original path/device')
  need(all(counter(value[k])for k in('bytes_available','bytes_total','inodes_available','inodes_total')),'nonnegative original integer capacities')
  need(4*GIB<=value['bytes_available']<=value['bytes_total']and value['mount_readonly']is False,'observed low bytes or readonly mount cannot heal')
  reported=value['inodes_total']>0
  need(value['inodes_reported']is reported and(not reported or 0<value['inodes_available']<=value['inodes_total']),
   'positive available inodes required only where a total is exposed')
  need(isinstance(value['scope'],str)and 0<len(value['scope'])<=1024,'bounded raw scope')
  self.first=tick if self.first is None else self.first;self.last=tick;self.count+=1
  self.devices.add(value['device']);self.paths.add(value['path']);need(len(self.devices)==len(self.paths)==1,'output device/path changed between observations')
  self.minimum_bytes=value['bytes_available']if self.minimum_bytes is None else min(self.minimum_bytes,value['bytes_available'])
  if reported:
   self.inodes_exposed+=1;self.minimum_inodes=value['inodes_available']if self.minimum_inodes is None else min(self.minimum_inodes,value['inodes_available'])
  else:self.inodes_unreported+=1
 def bind(self,device,path,inventory):
  need(self.count>0 and self.devices=={device}and self.paths=={path},'all runtime rows must match root-admitted output device/path')
  need(set(inventory)=={'observed_monotonic_ns','disk_observed_monotonic_ns','observed_non_runtime_bytes',
   'observed_non_runtime_files','vanished_entries','complete_entry_observations','scope'},'bounded latest inventory schema')
  need(all(counter(inventory[k])for k in('observed_monotonic_ns','disk_observed_monotonic_ns','observed_non_runtime_bytes',
   'observed_non_runtime_files','vanished_entries'))and inventory['disk_observed_monotonic_ns']==self.last<=inventory['observed_monotonic_ns'],
   'latest inventory must bind final raw disk observation')
  need(inventory['observed_non_runtime_bytes']<=2*GIB and inventory['complete_entry_observations']is(inventory['vanished_entries']==0),
   'sampled output cap/vanished-entry completeness contradicts raw values')
  need(isinstance(inventory['scope'],str)and 0<len(inventory['scope'])<=1024,'bounded latest inventory scope')
  return {'raw_disk_observations_checked':self.count,'first_observed_ns':self.first,'last_observed_ns':self.last,
   'device':device,'path':path,'minimum_reported_bytes_available':self.minimum_bytes,
   'inode_reporting_observations':self.inodes_exposed,'inode_unreported_observations':self.inodes_unreported,
   'minimum_reported_inodes_available':self.minimum_inodes,'latest_inventory':inventory,
   'runtime_thresholds_independently_checked':True,'raw_statvfs_block_arithmetic_independently_recomputed':False,
   'scope':'Each recorded available-byte/inode/readonly/device threshold and latest sampled inventory checked. Byte availability was derived by pinned statvfs code; raw block components were not serialized. Initial80GiB/64K admission is separate from runtime4GiB/positive exposed inodes. No instantaneous quota, capacity reservation, node eviction or continuous-health guarantee.'}
