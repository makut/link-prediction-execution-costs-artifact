#!/usr/bin/env python3
"""Separate stdlib-only recomputation and frozen-source check for E17."""
from pathlib import Path
import hashlib,json,math

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/seal_revision_20261124/posthoc_cost_regimes_v1'

def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__);result=[0.]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order)and values[order[end]]==values[order[start]]:end+=1
        rank=(start+end-1)/2
        for index in order[start:end]:result[index]=rank
        start=end
    return result

def spearman(x,y):
    a,b=ranks(x),ranks(y);ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b)
    numerator=math.fsum((u-ma)*(v-mb)for u,v in zip(a,b))
    return numerator/math.sqrt(math.fsum((u-ma)**2 for u in a)*math.fsum((v-mb)**2 for v in b))

def main():
    data=json.loads((OUT/'analysis.json').read_text());checked=[]
    for expected in data['graphs']:
        p=ROOT/expected['raw_file'];raw=json.loads(p.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['raw_sha256']
        sources=[]
        for name,pin in raw['provenance']['source_sha256'].items():
            source=ROOT/'outputs/vldb_submission_2026/artifact/src'/name
            assert hashlib.sha256(source.read_bytes()).hexdigest()==pin
            sources.append({'path':str(source.relative_to(ROOT)),'sha256':pin})
        cfg=raw['configuration'];assert cfg['auto_ratio']==1 and not cfg['directed']and cfg['repeats']==3 and cfg['warmup']==1
        rows=[]
        for r in raw['records']:
            a=r['methods']['ours_scan'];b=r['methods']['ours_pairwise']
            assert a['status']==b['status']=='ok'and a['exact_edges']and b['exact_edges']and a['exact_drnl']and b['exact_drnl']
            assert len(a['edge_seconds'])==len(b['edge_seconds'])==3 and r['sum_in_degrees']==0
            scan=sorted(a['edge_seconds'])[1];pair=sorted(b['edge_seconds'])[1]
            Q=r['num_nodes']*(r['num_nodes']-1)//2;D=r['sum_out_degrees'];chosen=pair if Q<=max(1,D)else scan
            assert r['auto_selected']==('pairwise'if Q<=max(1,D)else'scan')
            rows.append((scan,pair,min(scan,pair),chosen,D/Q,scan/pair))
        sums=[math.fsum(r[i]for r in rows)for i in range(4)]
        for key,value in zip(('always_scan','always_pair','observed_best_branch','declared_proxy_branch'),sums):
            assert math.isclose(value,expected['sum_per_record_medians_s'][key],rel_tol=1e-13)
        scan_wins=sum(a<b for a,b,*rest in rows);pair_wins=sum(b<a for a,b,*rest in rows)
        assert (scan_wins,pair_wins)==(expected['scan_wins'],expected['pair_wins'])
        rho=spearman([r[4]for r in rows],[r[5]for r in rows]);assert abs(rho-expected['spearman_D_over_Q_vs_scan_over_pair'])<1e-12
        checked.append({'graph':expected['graph'],'n':len(rows),'all_original_source_pins_match':sources,
            'sums_of_independently_recomputed_medians_s':dict(zip(('scan','pair','hindsight_min','proxy_branch'),sums)),
            'scan_wins':scan_wins,'pair_wins':pair_wins,'stdlib_average_tie_rank_spearman':rho,
            'hindsight_envelope_factor_over_pair':sums[1]/sums[2],'projected_proxy_branch_factor_over_pair':sums[1]/sums[3],
            'hindsight_sum_time_reduction_percent':100*(1-sums[2]/sums[1]),'projected_proxy_sum_time_reduction_percent':100*(1-sums[3]/sums[1]),
            'all_unweighted_proxy_labels_match':True})
    result={'status':'pass','kind':'independent_implementation_numeric_and_source_check','checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method':'Independent stdlib sorted-middle median, math.fsum and average-tie-rank Pearson recomputation, all compared to primary analysis; no SciPy/statistics/NumPy helper reused.',
        'frozen_proxy':'Q=k(k-1)/2 <= alpha*max(1,D_S), alpha=1; undirected/no incoming CSR, unweighted Q; no logarithms/search-length/cache/output constants in selector.',
        'actual_kernel':'Both count then fill. Scan gathers D_S neighbors, sorts S and binary-searches S. Pair tests each unordered pair using binary search in the shorter endpoint row. These log-search costs are not included in frozen Q.',
        'timer':'exact_benchmark.measure: t0 before method(), t1 after returned local CSR; edge_seconds=(t1-t0). DRNL t1..t2 separate; sampling/setup/independent oracle/validation outside. One warmup excluded; three saved timings.',
        'production_auto_warning':'Historical benchmark auto degree_sum is a Python loop; production extraction uses vectorized degree gathering. No historical ours_auto timing is used here or attributed to production.',
        'graphs':checked,'remote_jobs':0,'new_performance_experiments':0,'selector_changes':0}
    target=OUT/'independent_numeric_check.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'pass','graphs':len(checked),'records':sum(g['n']for g in checked),'source_pins_each_graph':len(checked[0]['all_original_source_pins_match'])}))

if __name__=='__main__':main()
