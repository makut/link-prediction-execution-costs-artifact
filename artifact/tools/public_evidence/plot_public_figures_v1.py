#!/usr/bin/env python3
"""Render three public EA figures from recorded evidence; no new measurements.

This derived public adapter verifies its inputs, recomputes all plotted values,
and writes only a fresh output directory. It never edits manuscript/source data,
loads models or full candidate-score NPZs, or performs remote operations.
"""
from pathlib import Path
import argparse,hashlib,io,json,math,os,shutil,statistics,sys,tarfile,zipfile

sys.dont_write_bytecode=True

import numpy as np

MIB=1<<20
QUALITY={'bytes':404147012,'sha256':'e17cdc9678d00a3f372fc8083741a200a63662226f730f47eeda79f89b4ad3b2'}
QUALITY_MANIFEST='9bf816acdce59b0905998adc47437092427cfa8d4c7154da69e5343a8eb15f7d'
QUALITY_PREFIX='lpp_public_quality_evidence_v1/'
RUNS=[f'{arm}_seed{seed}'for arm in('union','pv')for seed in(42,43,44)]
METHODS=('stock_scipy','corrected_local_csr','ours_pairwise')
GRID=(3000000,30000000,65000000,180000000)

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def identity(path):
    path=Path(path);require(path.is_file()and not path.is_symlink(),'regular input required')
    h=hashlib.sha256();size=0
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block);size+=len(block)
    return {'bytes':size,'sha256':h.hexdigest()}
def decode(raw):
    def unique(pairs):
        result={}
        for k,v in pairs:require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-14)

class Inputs:
    def __init__(self,root):
        self.root=Path(root).resolve();path=self.root/'MANIFEST.json'
        require(path.is_file()and not path.is_symlink()and path.stat().st_size<=4*MIB,'bounded regular systems manifest required')
        self.manifest=decode(path.read_bytes());self.files=self.manifest['files'];self.used={}
        self.manifest_identity=identity(path)
    def raw(self,name,limit=32*MIB):
        path=self.root/name;require(path.resolve().is_relative_to(self.root),'input path escape')
        expected=self.files[name];require(type(expected['bytes'])is int and 0<=expected['bytes']<=limit,'bounded evidence required')
        require(path.is_file()and not path.is_symlink()and path.stat().st_size<=limit,'bounded regular evidence required')
        raw=path.read_bytes();actual={'bytes':len(raw),'sha256':digest(raw)}
        require(actual=={k:expected[k]for k in('bytes','sha256')},'systems input bytes/SHA differ: '+name)
        self.used[name]=actual;return raw
    def obj(self,name):return decode(self.raw(name))

def positive_times(arm,count):
    require(arm.get('status')=='ok'and arm.get('exact_edges')is True and arm.get('exact_drnl')is True,'timing must have exact successful outputs')
    values=arm['edge_seconds'];require(len(values)==count and all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in values),'positive finite original repetitions required')
    return values

def native_panel(inputs):
    panels=[];oracles=None
    for tag,version in(('110','1.10.1'),('115','1.15.3')):
        for width,label in(('int32','native_int32'),('int64','forced_int64')):
            name=f'E8/scipy{tag}_{label}.json';report=inputs.obj(name);config=report['configuration']
            require(report['provenance']['scipy']==version and config['stock_index_dtype']==width,'SciPy/temporary width differs')
            require(config['fixture']is True and config['limit']==8 and config['repeats']==5 and config['warmup']==2,'frozen fixture protocol differs')
            require(report['provenance']['graph']['nodes']==9 and report['provenance']['graph']['stored_entries']==18,'nine-node/eighteen-entry fixture required')
            records=report['records'];require(len(records)==32,'all32 fixture records required')
            fingerprints={};groups={n:{m:[]for m in METHODS}for n in GRID}
            for row in records:
                n=row['virtual_graph_nodes'];require(n in groups,'unexpected virtual domain')
                key=(n,row['pair_index']);require(key not in fingerprints,'duplicate fixture query')
                fingerprints[key]=tuple(row[k]for k in('node_set_sha256','canonical_edges_sha256','drnl_sha256'))
                for method in METHODS:groups[n][method].append(statistics.mean(positive_times(row['methods'][method],5)))
            require(all(len(groups[n][method])==8 for n in GRID for method in METHODS),'eight queries per domain required')
            require(oracles is None or fingerprints==oracles,'N/version/width semantic fingerprints differ');oracles=fingerprints
            # Each physical query must preserve its logical outputs across N.
            for pair in{key[1]for key in fingerprints}:require(len({fingerprints[n,pair]for n in GRID})==1,'virtual isolates changed local outputs')
            values={method:[1000*statistics.median(groups[n][method])for n in GRID]for method in METHODS}
            panels.append({'scipy':version,'width':width,'virtual_N':list(GRID),'median_query_mean_ms':values,'source':name})
    return panels

def ranks(values):
    order=np.argsort(values,kind='stable');out=np.empty(len(order),float);i=0
    while i<len(order):
        j=i+1
        while j<len(order)and values[order[j]]==values[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out
def cost_panel(inputs):
    raw=inputs.obj('E1/friendster_exact_int32.json');projection=inputs.obj('E17/public_friendster_analysis.json')
    require(len(raw['records'])==len(projection['rows'])==768,'all768 public fixed sets required')
    require(len(projection['graphs'])==1 and projection['graphs'][0]['graph']=='friendster','Friendster-only projection required')
    require(projection['graphs'][0]['raw_sha256']==inputs.used['E1/friendster_exact_int32.json']['sha256'],'posthoc raw-source binding differs')
    points=[]
    for index,(record,row)in enumerate(zip(raw['records'],projection['rows'])):
        scan=statistics.median(positive_times(record['methods']['ours_scan'],3));pair=statistics.median(positive_times(record['methods']['ours_pairwise'],3))
        k=record['num_nodes'];d=record['sum_out_degrees'];q=k*(k-1)//2
        require(q>0 and record['cap']in(50,200),'declared caps/nonempty pair domain required')
        expected={'record_index':index,'k':k,'D_S':d,'Q':q,'cap':record['cap'],'S_sha256':record['node_set_sha256']}
        require(all(row[key]==value for key,value in expected.items()),'posthoc record identity differs')
        require(all(close(row[key],value)for key,value in{'D_over_Q':d/q,'scan_over_pair':scan/pair,'scan_median_s':scan,'pair_median_s':pair}.items()),'posthoc point arithmetic differs')
        points.append({'record_index':index,'cap':record['cap'],'D_over_Q':d/q,'scan_over_pair':scan/pair,'scan_seconds':scan,'pair_seconds':pair,'proxy_pair':q<=max(1,d)})
    a=math.fsum(p['pair_seconds']for p in points);best=math.fsum(min(p['scan_seconds'],p['pair_seconds'])for p in points)
    proxy=math.fsum(p['pair_seconds']if p['proxy_pair']else p['scan_seconds']for p in points)
    rho=float(np.corrcoef(ranks(np.array([p['D_over_Q']for p in points])),ranks(np.array([p['scan_over_pair']for p in points])))[0,1])
    summary=projection['graphs'][0]
    require(close(rho,summary['spearman_D_over_Q_vs_scan_over_pair'])and close(a/best,summary['hindsight_envelope_gain_over_best_fixed_plan'])and close(a/proxy,summary['declared_proxy_branch_gain_over_best_fixed_plan']),'posthoc aggregate arithmetic differs')
    return {'points':points,'rho':rho,'pair_wins':sum(p['pair_seconds']<p['scan_seconds']for p in points),'hindsight_factor':a/best,'proxy_factor':a/proxy}

def quality_vectors(path):
    require(identity(path)==QUALITY,'exact immutable public quality asset required')
    targets={f'friendster/results/quality_friendster/runs/{run}/run1/{name}.npy'for run in RUNS for name in('query_rr','query_ids')}
    payload={};manifest_raw=None
    with tarfile.open(path,'r|gz')as archive:
        for member in archive:
            name=member.name.removeprefix(QUALITY_PREFIX)
            if name not in targets and name!='MANIFEST.json':continue
            require(member.name==QUALITY_PREFIX+name and member.isfile()and member.size<=2*MIB,'bounded exact quality member required')
            require(name not in payload,'duplicate quality member')
            raw=archive.extractfile(member).read(member.size+1);require(len(raw)==member.size,'complete bounded quality member required')
            payload[name]=raw
            if name=='MANIFEST.json':manifest_raw=raw
    require(manifest_raw is not None and digest(manifest_raw)==QUALITY_MANIFEST and set(payload)==targets|{'MANIFEST.json'},'all12 small quality vectors and exact manifest required')
    manifest=decode(manifest_raw)['files'];arrays={}
    for name in targets:
        raw=payload[name];require({'bytes':len(raw),'sha256':digest(raw)}=={k:manifest[name][k]for k in('bytes','sha256')},'quality vector identity differs')
        value=np.load(io.BytesIO(raw),allow_pickle=False,max_header_size=4096)
        require(isinstance(value,np.ndarray)and value.shape==(4096,),'fixed4096 vector required')
        arrays[name]=value
    return arrays

def degree_panel(inputs,quality):
    geometry_raw=inputs.raw('E4/test_geometry.npz',MIB);meta=inputs.obj('E4/test_geometry.json');plan=inputs.obj('E4/plan.json');saved=inputs.obj('E4/rr_by_degree_balance.json')
    require(meta['geometry_sha256']==digest(geometry_raw)and meta['analysis_plan_sha256']==inputs.used['E4/plan.json']['sha256'],'geometry/plan binding differs')
    require(plan['predeclared_before_joint_freeze']is True and saved['frozen_manifest_sha256']==meta['frozen_manifest_sha256'],'predeclaration/freeze differs')
    with zipfile.ZipFile(io.BytesIO(geometry_raw))as archive:require(len(archive.infolist())<=16 and sum(z.file_size for z in archive.infolist())<=8*MIB,'bounded geometry arrays required')
    with np.load(io.BytesIO(geometry_raw),allow_pickle=False,max_header_size=4096)as geometry:
        ids=np.array(geometry['query_ids']);bins=np.array(geometry['bin_id'])
    require(ids.shape==bins.shape==(4096,)and ids.dtype.kind in'iu'and bins.dtype.kind in'iu'and len(set(ids.tolist()))==4096 and set(bins.tolist())=={-1,0,1,2,3,4},'exact cohort/bins required')
    vectors=quality_vectors(quality);aligned={}
    for run in RUNS:
        key=f'friendster/results/quality_friendster/runs/{run}/run1/';query=vectors[key+'query_ids.npy'];rr=vectors[key+'query_rr.npy']
        require(query.dtype.kind in'iu'and rr.dtype.kind=='f'and len(set(query.tolist()))==4096 and np.all(np.isfinite(rr))and np.all((rr>0)&(rr<=1)),'valid query IDs/RR required')
        require(np.array_equal(np.sort(query),np.sort(ids)),'original query cohort differs')
        order=np.argsort(query);aligned[run]=rr[order[np.searchsorted(query[order],ids)]]
    rows=[];original={r['bin_id']:r for r in saved['rows']}
    for bid in(-1,0,1,2,3,4):
        mask=bins==bid;differences=[float(np.mean(aligned[f'pv_seed{s}'][mask]-aligned[f'union_seed{s}'][mask]))for s in(42,43,44)]
        mean=statistics.mean(differences);sd=statistics.stdev(differences);old=original[bid]
        require(old['query_count']==int(mask.sum())and all(close(a,b['pv_minus_union_mrr'])for a,b in zip(differences,old['seeds']))and close(mean,old['pv_minus_union_mrr']['mean'])and close(sd,old['pv_minus_union_mrr']['sample_sd']),'degree-bin aggregation differs')
        rows.append({'bin_id':bid,'queries':int(mask.sum()),'paired_seed_differences':differences,'mean':mean,'sample_sd':sd})
    require([r['queries']for r in rows]==[44,1502,1154,1036,289,71],'declared bin counts differ')
    return rows

def render(out,native,cost,degree):
    # Keep Matplotlib's font/config cache inside this already fresh output tree.
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#D6572F','#356AA3','#22816C'];labels=['Row gather + SciPy column slice','Corrected local CSR','Compiled pairwise + CSR']
    fig,ax=plt.subplots(figsize=(7.1,3.65));fig.subplots_adjust(top=.71,bottom=.2,left=.105,right=.985)
    for row in native:
        if row['width']!='int32':continue
        old=row['scipy']=='1.10.1'
        for method,color in zip(METHODS,colors):ax.plot(GRID,row['median_query_mean_ms'][method],color=color,ls='-'if old else'--',marker='o'if old else's',mfc=color if old else'white',lw=1.5,ms=4)
    ax.set(xscale='log',yscale='log',xlabel='Virtual global column count N',ylabel='Edge induction time (ms; log scale)',ylim=(.08,1500))
    ax.set_xticks(GRID,['3M','30M','65M','180M']);ax.minorticks_off();ax.grid(axis='y',color='#DFE5EA',lw=.7)
    fig.legend(handles=[Line2D([],[],color=c,label=l)for c,l in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.53,1),ncol=3,frameon=False,fontsize=8)
    fig.legend(handles=[Line2D([],[],color='#555',ls=ls,marker=mk,mfc=face,label='SciPy '+version)for version,ls,mk,face in[('1.10.1','-','o','#555'),('1.15.3','--','s','white')]],loc='upper center',bbox_to_anchor=(.53,.895),ncol=2,frameon=False,fontsize=8)
    fig.savefig(out/'native_int32_control_paper.png',dpi=300,facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.5,2.6));fig.subplots_adjust(left=.24,bottom=.23,right=.98,top=.87)
    for cap,color in((50,'#356AA3'),(200,'#C46B32')):
        points=[p for p in cost['points']if p['cap']==cap];ax.scatter([p['D_over_Q']for p in points],[p['scan_over_pair']for p in points],s=13,color=color,alpha=.4,edgecolors='none',label='cap '+str(cap))
    ax.set(xscale='log',yscale='log',xlabel=r'$D_S/[k(k-1)/2]$',ylabel='Median scan / pair time\n>1 means pair is faster',title='Friendster: fixed-set cost regimes')
    ax.tick_params(which='both',labelsize=7);ax.yaxis.label.set_size(8);ax.title.set_size(9)
    ax.axhline(1,color='#555',ls=':',lw=1);ax.axvline(1,color='#555',ls='--',lw=1);ax.legend(frameon=False,fontsize=7,loc='lower right')
    fig.savefig(out/'posthoc_cost_regimes_friendster.png',dpi=300,facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,3.65));fig.subplots_adjust(left=.105,right=.99,top=.79,bottom=.28)
    ax.axvspan(-.5,.5,color='#F0F4F8');ax.axvspan(4.5,5.5,color='#FFF4E6');ax.axhline(0,color='#708090',ls='--',lw=.9);ax.grid(axis='y',color='#DFE5EA',lw=.7);ax.set_axisbelow(True)
    for j,(color,marker,offset)in enumerate(zip(('#286DA8','#138777','#C46B32'),('o','s','^'),(-.19,-.08,.19))):ax.scatter(np.arange(6)+offset,[r['paired_seed_differences'][j]for r in degree],color=color,marker=marker,s=26,edgecolors='white',linewidths=.5,zorder=4)
    means=[r['mean']for r in degree];sd=[r['sample_sd']for r in degree]
    ax.errorbar(range(6),means,yerr=sd,fmt='D',color='#172B3B',mfc='white',ms=4.5,lw=1.15,capsize=3,zorder=5)
    for i,row in enumerate(degree):ax.text(i,-.012 if i==0 else max(row['mean']+row['sample_sd'],*row['paired_seed_differences'])+.006,f"{row['mean']:+.4f}",ha='center',va='top'if i==0 else'bottom',fontsize=8)
    names=['Cold\nmin degree 0','[1, 2)','[2, 4)','[4, 16)','[16, 64)','≥64'];ax.set_xticks(range(6),[name+f"\nn = {r['queries']:,}"for name,r in zip(names,degree)])
    ax.set(xlim=(-.52,5.52),ylim=(-.021,.158),xlabel='Positive-endpoint train-degree ratio (max / min)',ylabel='Paired MRR difference (PV − union)');ax.set_yticks([0,.04,.08,.12]);ax.tick_params(length=0,labelsize=8)
    handles=[Line2D([],[],marker=marker,ls='none',color=color,label=f'Seed {seed}')for seed,marker,color in zip((42,43,44),('o','s','^'),('#286DA8','#138777','#C46B32'))]+[Line2D([],[],marker='D',color='#172B3B',mfc='white',label='Mean ± sample SD')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.98),ncol=4,frameon=False,fontsize=8)
    fig.savefig(out/'degree_imbalance_mrr_paper.png',dpi=300,facecolor='white');plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--quality-asset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    require(not a.out.exists()and not a.out.is_symlink(),'fresh figure output required');require(not a.out.resolve().is_relative_to(a.asset_root.resolve()),'output must be outside immutable input asset')
    inputs=Inputs(a.asset_root);native=native_panel(inputs);cost=cost_panel(inputs);degree=degree_panel(inputs,a.quality_asset)
    a.out.parent.mkdir(parents=True,exist_ok=True);require(shutil.disk_usage(a.out.parent).free>=int(1.5*(1<<30))+16*MIB,'1.5GiB reserve plus16MiB render budget required')
    a.out.mkdir();render(a.out,native,cost,degree)
    outputs={p.name:identity(p)for p in a.out.glob('*.png')};require(len(outputs)==3 and sum(p.stat().st_size for p in a.out.rglob('*')if p.is_file())<=16*MIB,'bounded complete3-figure output including font cache required')
    require(shutil.disk_usage(a.out.parent).free>=int(1.5*(1<<30))+MIB,'final1.5GiB reserve plus numeric-receipt budget required')
    result={'status':'RECORDED_PUBLIC_VALUES_RECOMPUTED_AND_THREE_FIGURES_RENDERED','adapter':'Derived public plotting adapter; not a frozen original experiment.','adapter_source':identity(__file__),'systems_manifest':inputs.manifest_identity,'new_measurements':0,'input_member_identities':inputs.used,'quality_asset':QUALITY,'quality_manifest_sha256':QUALITY_MANIFEST,'native_panel':native,'posthoc_panel':cost,'degree_panel':degree,'figures':outputs,'limitations':['Visual PNG bytes may vary with Matplotlib/fonts. Numeric panels are recomputed and compared to recorded sources.','No independent confidence interval, causal test, new experiment, model loading or full-score NPZ extraction.']}
    (a.out/'NUMERIC_PANELS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':result['status'],'figures':list(outputs),'out':str(a.out)}));return 0
if __name__=='__main__':raise SystemExit(main())
