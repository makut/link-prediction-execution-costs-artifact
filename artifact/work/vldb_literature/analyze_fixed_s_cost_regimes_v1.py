#!/usr/bin/env python3
"""Posthoc descriptive analysis only: existing fixed-S medians, no experiments/HPO."""
from pathlib import Path
import hashlib,json,math,statistics
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'outputs/seal_revision_20261124/cpu/extracted/results/cpu'
OUT=ROOT/'outputs/seal_revision_20261124/posthoc_cost_regimes_v1'
PINS={'friendster':(9040463,'bed119047100f89080081fe638ed60816d0256bfeea9750418668078cdb2d975'),
      'vk':(8987220,'ce8e1065f2a740b0ee16bee5fb9e35fc37364b2d152be411089197c0f7483b04')}
BINS=(0,.25,1,4,16,64,float('inf'))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    a=np.asarray(values,dtype=float)
    return {'n':int(len(a)),**{key:float(value)for key,value in zip(('min','p25','p50','p75','p95','max'),np.quantile(a,[0,.25,.5,.75,.95,1]))}}if len(a)else{'n':0}
def summarize(rows):
    if not rows:return {'n':0}
    scan=sum(r['scan_median_s']for r in rows);pair=sum(r['pair_median_s']for r in rows)
    best=sum(r['observed_best_median_s']for r in rows);selected=sum(r['proxy_selected_median_s']for r in rows)
    sr=spearmanr([r['D_over_Q']for r in rows],[r['scan_over_pair']for r in rows]).statistic if len(rows)>2 else None
    return {'n':len(rows),'scan_wins':sum(r['observed_winner']=='scan'for r in rows),
        'pair_wins':sum(r['observed_winner']=='pairwise'for r in rows),'ties':sum(r['observed_winner']=='tie'for r in rows),
        'scan_repeat_range_entirely_faster':sum(r['clear_winner']=='scan'for r in rows),
        'pair_repeat_range_entirely_faster':sum(r['clear_winner']=='pairwise'for r in rows),
        'overlapping_observed_repeat_ranges':sum(r['clear_winner']=='overlap'for r in rows),
        'proxy_selects_scan':sum(r['proxy_selected']=='scan'for r in rows),
        'proxy_selects_pair':sum(r['proxy_selected']=='pairwise'for r in rows),
        'proxy_matches_observed_median_winner':sum(r['proxy_selected']==r['observed_winner']or r['observed_winner']=='tie'for r in rows),
        'D_over_Q':stats([r['D_over_Q']for r in rows]),'scan_over_pair':stats([r['scan_over_pair']for r in rows]),
        'proxy_branch_hindsight_regret':stats([r['proxy_branch_regret']for r in rows]),
        'spearman_D_over_Q_vs_scan_over_pair':None if sr is None or not math.isfinite(sr)else float(sr),
        'sum_per_record_medians_s':{'always_scan':scan,'always_pair':pair,'observed_best_branch':best,'declared_proxy_branch':selected},
        'hindsight_envelope_gain_over_best_fixed_plan':min(scan,pair)/best,
        'declared_proxy_branch_regret_ratio_of_sums':selected/best,
        'declared_proxy_branch_gain_over_best_fixed_plan':min(scan,pair)/selected}

def analyze(graph):
    path=INPUT/(graph+'_exact_int32.json');size,pin=PINS[graph]
    assert path.stat().st_size==size and digest(path)==pin,'frozen raw input pin changed'
    raw=json.loads(path.read_text());assert len(raw['records'])==768
    cfg=raw['configuration'];assert cfg['stock_index_dtype']=='int32'and cfg['repeats']==3 and cfg['warmup']==1 and cfg['auto_ratio']==1
    assert not cfg['directed']and raw['provenance']['numba_available']is True
    rows=[]
    for index,record in enumerate(raw['records']):
        k=record['num_nodes'];D=record['sum_out_degrees'];Q=k*(k-1)//2
        assert k==len(record['nodes'])and k>=2 and D>=0
        times={}
        for name in('ours_scan','ours_pairwise'):
            result=record['methods'][name]
            assert result['status']=='ok'and result['exact_edges']is True and result['exact_drnl']is True
            values=result['edge_seconds'];assert len(values)==3 and all(math.isfinite(v)and v>0 for v in values)
            median=statistics.median(values);assert median==result['edge_summary']['p50_s']
            times[name]=(median,min(values),max(values))
        scan,slow,shigh=times['ours_scan'];pair,plow,phigh=times['ours_pairwise']
        selected='pairwise'if Q<=max(1,D)else'scan'
        assert selected==record['auto_selected'],'stored alpha1 branch differs from declared proxy'
        winner='scan'if scan<pair else'pairwise'if pair<scan else'tie'
        chosen=pair if selected=='pairwise'else scan;best=min(scan,pair)
        rows.append({'graph':graph,'record_index':index,'S_sha256':record['node_set_sha256'],'pair_index':record['pair_index'],
            'sampler':record['sampler'],'cap':record['cap'],'workload':record['workload'],'k':k,'D_S':D,'Q':Q,'D_over_Q':D/Q,
            'scan_median_s':scan,'pair_median_s':pair,'scan_over_pair':scan/pair,'observed_winner':winner,
            'clear_winner':'scan'if shigh<plow else'pairwise'if phigh<slow else'overlap',
            'proxy_selected':selected,'observed_best_median_s':best,'proxy_selected_median_s':chosen,
            'proxy_branch_regret':chosen/best,'stored_auto_timer_used':False})
    strata=[]
    for key in sorted({(r['sampler'],r['cap'],r['workload'])for r in rows}):
        subset=[r for r in rows if(r['sampler'],r['cap'],r['workload'])==key];assert len(subset)==64
        strata.append({'sampler':key[0],'cap':key[1],'workload':key[2],**summarize(subset)})
    bins=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        subset=[r for r in rows if lo<=r['D_over_Q']<hi]
        bins.append({'D_over_Q_lower_inclusive':lo,'D_over_Q_upper_exclusive':None if math.isinf(hi)else hi,**summarize(subset)})
    graph_summary={'graph':graph,'raw_file':str(path.relative_to(ROOT)),'raw_bytes':size,'raw_sha256':pin,
        'N':raw['provenance']['graph']['nodes'],'M_stored':raw['provenance']['graph']['stored_entries'],
        'unique_S_hashes':len({r['S_sha256']for r in rows}),'unique_physical_query_pairs':len({tuple(x['pair'])for x in raw['records']}),
        'k':stats([r['k']for r in rows]),'D_S':stats([r['D_S']for r in rows]),**summarize(rows),'strata':strata,'proxy_bins':bins}
    return graph_summary,rows

def main():
    assert not OUT.exists(),'fresh analysis outputs; no overwrite'
    OUT.mkdir(parents=True)
    summaries=[];rows=[]
    for graph in PINS:
        summary,items=analyze(graph);summaries.append(summary);rows.extend(items)
    result={'schema_version':1,'kind':'posthoc_fixed_S_cost_regimes','scope':'Descriptive reuse of existing matched-width fixed-S edge construction timings only; no new experiment, HPO, threshold fitting, selector change or production-auto timing claim.',
        'analysis_source_sha256':digest(Path(__file__)),'proxy':'Q=k(k-1)/2; choose pairwise iff Q<=max(1,D_S), frozen alpha1',
        'timing':'per-record median of three saved edge_seconds repetitions; sampling/setup/oracle/DRNL excluded, target removal and local CSR packaging included',
        'regret':'saved fixed-mode chosen-branch median divided by min(scan,pair) median; hindsight lower envelope is optimistic/noisy and not measured execution of an oracle/dispatcher',
        'uncertainty':'No p-values/CI/population claims: fixed correlated query pairs, repeated S, shared graph/cache and repeated timings. Spearman is descriptive only. Repeat-range separation is not significance.',
        'bins':'Fixed powers-of-four diagnostic bins centered on declared threshold1; entire analysis is posthoc, not a predeclared heldout test.',
        'graphs':summaries,'rows':rows}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['Posthoc fixed-S cost regimes; existing int32 reports only. No new experiment/selector tuning.','']
    for g in summaries:
        n=g['n'];counts=(g['scan_wins'],g['pair_wins']);acc=g['proxy_matches_observed_median_winner']/n
        lines += [f"{g['graph']}: {n} records, {g['unique_physical_query_pairs']} physical pairs, {g['unique_S_hashes']} distinct S hashes.",
            f"Scan wins {counts[0]}, pair wins {counts[1]}, ties {g['ties']}; clear repeat-range scan/pair: {g['scan_repeat_range_entirely_faster']}/{g['pair_repeat_range_entirely_faster']}; overlapping {g['overlapping_observed_repeat_ranges']}.",
            f"Frozen proxy chooses scan/pair {g['proxy_selects_scan']}/{g['proxy_selects_pair']}; observed-median branch agreement {acc:.2%}.",
            f"Spearman(D/Q, scan/pair)={g['spearman_D_over_Q_vs_scan_over_pair']:.4f}; scan/pair median={g['scan_over_pair']['p50']:.4f}, p95={g['scan_over_pair']['p95']:.4f}.",
            f"Branch hindsight regret median={g['proxy_branch_hindsight_regret']['p50']:.4f}, p95={g['proxy_branch_hindsight_regret']['p95']:.4f}, max={g['proxy_branch_hindsight_regret']['max']:.4f}.",
            f"Sum-of-record-median envelope gain over best fixed plan={g['hindsight_envelope_gain_over_best_fixed_plan']:.4f}; proxy regret={g['declared_proxy_branch_regret_ratio_of_sums']:.4f}; proxy gain over best fixed={g['declared_proxy_branch_gain_over_best_fixed_plan']:.4f}.",'']
        for s in g['strata']:
            lines.append(f"  {s['sampler']}, cap{s['cap']}, {s['workload']}: scan/pair wins {s['scan_wins']}/{s['pair_wins']}; median scan/pair {s['scan_over_pair']['p50']:.3f}; proxy regret p95 {s['proxy_branch_hindsight_regret']['p95']:.3f}")
        lines.append('')
    lines += ['Hindsight sums are descriptive sums of per-record medians, not measured end-to-end seconds/throughput.',
        'Historical ours_auto wall times are deliberately unused. Branch decisions are checked against its saved auto_selected label only.',
        'The two graphs/12 equal-size strata are sampled workload matrices, not production traffic weights.',
        'No comparison to the pending C++ touched-reset/native controls or PENCIL results is inferred.']
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
    for ax,g in zip(axes,summaries):
        for cap,color in((50,'#2768a6'),(200,'#c16a21')):
            items=[r for r in rows if r['graph']==g['graph']and r['cap']==cap]
            ax.scatter([r['D_over_Q']for r in items],[r['scan_over_pair']for r in items],s=12,alpha=.36,color=color,label=f'cap {cap}',edgecolors='none')
        ax.axvline(1,color='#555555',linestyle='--',linewidth=.8);ax.axhline(1,color='#555555',linestyle=':',linewidth=.9)
        ax.set(xscale='log',yscale='log',xlabel=r'$D_S / [k(k-1)/2]$',title=g['graph'].title());ax.grid(alpha=.13)
    axes[0].set_ylabel('Median scan / pair time\n>1 means pair is faster');axes[1].legend(frameon=False)
    fig.suptitle('Posthoc fixed-S diagnostic: stored timings, no selector tuning',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'cost_regimes.png',dpi=250);plt.close(fig)
    files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)}for p in OUT.iterdir()if p.is_file()}
    (OUT/'MANIFEST.json').write_text(json.dumps({'analysis_source_sha256':digest(Path(__file__)),'files':files,'remote_jobs':0},indent=2)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'graphs':[{k:g[k]for k in('graph','n','scan_wins','pair_wins','proxy_matches_observed_median_winner','spearman_D_over_Q_vs_scan_over_pair','hindsight_envelope_gain_over_best_fixed_plan','declared_proxy_branch_regret_ratio_of_sums')}for g in summaries]},indent=2))

if __name__=='__main__':main()
