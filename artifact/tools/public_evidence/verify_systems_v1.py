#!/usr/bin/env python3
"""Verify extracted public evidence and recompute recorded numeric summaries.

Reads only. No graph, model, full candidate-score NPZ, network or YT access.
This checks historical measurements; it does not make new timing claims.
"""
import argparse,collections,hashlib,io,json,math,statistics
from pathlib import Path,PurePosixPath
import numpy as np

MIB=1<<20
ARMS=('reference_union_stock','reference_union_corrected','reference_union_compiled','production_union','production_pv')
def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(MIB),b''):h.update(block)
    return h.hexdigest()
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-13)
def safe(name):
    p=PurePosixPath(name)
    require(name and not p.is_absolute()and '..'not in p.parts and '\\'not in name and '\x00'not in name and str(p)==name,'unsafe/noncanonical manifest path')
    return name
class Inputs:
    def __init__(self,root):
        self.root=Path(root).resolve();self.manifest=json.loads((self.root/'MANIFEST.json').read_text());self.used={}
        require(self.manifest.get('schema_version')==1,'manifest schema differs')
    def path(self,name):
        candidate=self.root/safe(name)
        require(candidate.is_file()and not candidate.is_symlink()and candidate.resolve().is_relative_to(self.root),'regular payload within root required')
        require(not any(p.is_symlink()for p in candidate.parents if p!=self.root.parent),'symlink parent refused')
        return candidate
    def raw(self,name,limit=32*MIB):
        row=self.manifest['files'][name];path=self.path(name)
        require(row['bytes']<=limit and path.stat().st_size==row['bytes'],'bounded payload length differs')
        value=path.read_bytes();require(hashlib.sha256(value).hexdigest()==row['sha256'],'payload hash differs: '+name)
        self.used[name]={'bytes':len(value),'sha256':row['sha256']};return value
    def obj(self,name):return json.loads(self.raw(name))
    def verify(self):
        for name,row in self.manifest['files'].items():
            path=self.path(name);require(path.stat().st_size==row['bytes']and digest(path)==row['sha256'],'payload identity differs: '+name)
        return len(self.manifest['files'])
def times(row,key,count):
    values=row[key];require(len(values)==count and all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in values),'finite positive raw repetitions required')
    return values
def summarize(inputs):
    result={}
    # Every exact32/64 case is retained; equality is a case-wise identity check,
    # not a second independent draw of sampled subgraphs.
    reports=[inputs.obj('E1/friendster_exact_int'+str(w)+'.json')for w in(32,64)]
    require(all(len(r['records'])==768 for r in reports),'both full768-record reports required')
    stock_width_ratios=[];groups=collections.defaultdict(lambda:collections.defaultdict(list));method_cases=0
    for a,b in zip(*(r['records']for r in reports)):
        require(all(a[k]==b[k]for k in('node_set_sha256','nodes','canonical_edges_sha256','drnl_sha256')),'width case/oracle identity differs')
        for row in(a,b):
            require(set(row['methods'])=={'stock_scipy','corrected_local_csr','ours_scan','ours_pairwise','ours_auto'},'all5exact methods required')
            for measured in row['methods'].values():
                require(measured['status']=='ok'and measured['exact_edges']is True and measured['exact_drnl']is True,'all7680exact method-cases required')
                times(measured,'edge_seconds',3);times(measured,'drnl_seconds',3);method_cases+=1
        stock_width_ratios.append(statistics.median(b['methods']['stock_scipy']['edge_seconds'])/statistics.median(a['methods']['stock_scipy']['edge_seconds']))
        key=(a['sampler'],a['cap'],a['workload'])
        for method in('corrected_local_csr','ours_auto'):groups[key][method].extend(a['methods'][method]['edge_seconds'])
    require(len(groups)==12 and all(len(group['ours_auto'])==192 for group in groups.values()),'all12declared strata of64cases×3reps required')
    ratios=[statistics.median(g['corrected_local_csr'])/statistics.median(g['ours_auto'])for g in groups.values()]
    result['E1']={'records_per_width':768,'shared_case_oracles':True,'exact_method_cases':method_cases,
      'median_per_case_stock_int64_over_int32':statistics.median(stock_width_ratios),
      'int32_12strata_corrected_over_auto_pooled_repetition_medians':{'minimum':min(ratios),'maximum':max(ratios),'median':statistics.median(ratios)}}
    memory=inputs.obj('E6/v3/friendster/report.json');rows=[]
    for run in memory['runs']:
        require(run['status']=='complete'and run['consumers']==16 and run['returncode']==0,'all4 completed16consumer memory arms required')
        path='E6/v3/friendster/'+Path(run['samples_file']).name
        samples=[json.loads(line)for line in inputs.raw(path).splitlines()]
        valid=[s for s in samples if all(s.get(k)is True for k in('steady_eligible','publication_eligible','pss_complete','membership_stable','phase_stable'))and s.get('phase')=='warm_shared_consumers'and s.get('process_count')==16]
        peaks=[s['tree_pss_bytes']for s in samples if s.get('publication_eligible')is True and s.get('tree_pss_bytes')is not None]
        require(len(valid)>=2 and peaks,'eligible memory samples required')
        median=statistics.median(s['tree_pss_bytes']for s in valid);peak=max(peaks)
        require(close(median,run['steady_shared']['median_tree_pss_bytes'])and close(peak,run['peak']['tree_pss_bytes']),'raw memory median/peak differs')
        rows.append({'representation':run['representation'],'steady_pss_gib':median/(1<<30),'sampled_peak_pss_gib':peak/(1<<30),'eligible_steady_samples':len(valid),'logical_gib':run['logical']['total_gib']})
    require(len(rows)==4,'all4memory representations required');result['tab:memory']=rows
    pipeline=[]
    for dataset in('citation2','friendster'):
        for arm in ARMS:
            for workers in(1,8):
                name=f'{dataset}_{arm}_workers{workers}';report=inputs.obj('E10/jobs/'+name+'/report.json')
                require(len(report['runs'])==1,'one frozen run per job required');run=report['runs'][0];t=run['timing']
                require(run['status']=='complete'and t['measured_graphs']==3200 and t['measured_steps']==100 and len(run['audits'])==32,'complete V4timing/prefix required')
                require(close(3200/t['steady_seconds'],t['graphs_per_second']),'raw pipeline wall arithmetic differs')
                pipeline.append({'dataset':dataset,'arm':arm,'workers':workers,'steady_seconds':t['steady_seconds'],'graphs_per_second':t['graphs_per_second']})
    comparisons=inputs.obj('E10/projections/summary.json')['reference_correction_audits'];require(len(comparisons)==12,'all12raw prefix comparisons required')
    result['tab:pipeline']=pipeline;result['E10_prefix_comparisons']=comparisons
    summary=inputs.obj('E12/summary.json');rr=np.load(io.BytesIO(inputs.raw('E12/reciprocal_ranks.npy',MIB)),allow_pickle=False)
    require(rr.shape==(4096,5)and np.isfinite(rr).all(),'five full heuristicRR vectors required')
    heuristics={}
    for index,name in enumerate(summary['methods']):
        mean=float(rr[:,index].mean());require(close(mean,summary['metrics'][name]['mrr']),'raw heuristic mean differs');heuristics[name]=mean
    result['tab:heuristics']=heuristics
    native=[]
    for mode in('reference_full_clear','persistent_touched_reset','upstream_torch_sparse_direct'):
        report=inputs.obj('E18/friendster/'+mode+'.json');records=report['records']
        require(report['status']=='complete'and report['record_count']==128 and len(records)==128 and report['all_executed_outputs_exact']is True,'all128native exact outputs required')
        totals={key:math.fsum(statistics.median(times(r,key,3))for r in records)for key in('raw_operator_seconds','edge_seconds','drnl_seconds')}
        require(all(r['status']=='ok'and r['exact_edges']is True and r['exact_drnl']is True and r['exact_ordered_rows_cols_and_original_eids']is True for r in records),'native exact ordered output required')
        native.append({'mode':mode,'records':128,**totals})
    result['tab:native-reset']=native
    # This adapter recomputes E8 and E17 independently from original raw timers.
    import plot_public_figures_v1 as plotting
    result['E8']=plotting.native_panel(inputs)
    cost=plotting.cost_panel(inputs);result['tab:posthoc-cost-regimes']={k:v for k,v in cost.items()if k!='points'}
    result['E7_scope']='The unchanged512-query report is retained; its shared-priority diagnostic has guarded denominators and does not establish sampler-law equivalence or causal MRR.'
    result['limits']=['No new timing, graph loading, training or inference.','Quality tables and E4degree figure also require the separately pinned quality asset.','Outer historical controller exit/lock release remains UNKNOWN where disclosed.']
    return result
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True);args=p.parse_args(argv)
    inputs=Inputs(args.asset_root);count=inputs.verify();result=summarize(inputs)
    print(json.dumps({'status':'PASS','verified_payloads':count,'new_measurements':0,'numeric_summaries':result},indent=2,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
