#!/usr/bin/env python3
"""Combine exactly three pinned, independently audited, complete v6 blocks.

Descriptive statistics across allocations only. No raw experiment, producer
import, GPU/cluster/YT call, CI, population inference or historical pooling.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS=('reference_union_compiled','reference_union_native_persistent')
HORIZONS=('short320','long6400')
PINS=dict(source_manifest_sha256='3fd8783173e4a850018d860459881aafd1dd49e8d4cda101eb273641e47af086',
    protocol_sha256='e89b371089bd01f71579fdc00ec06004373bb549542658cc223914759f392e18',
    wrapper_sha256='072575b15555a83f223e15643b869874f9a5e4f83172c06e0f4f5d032bdae4c9',
    wrapper_ready_sha256='88b06c57d7de42c062fbd2ea76cd268999d3bbabe8bbd31130a9ec79dfa5a785')


def need(ok,message):
    if not ok:raise ValueError(message)


def decode(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,'duplicate JSON key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))


def read_pinned(path,expected):
    path=Path(path);need(path.is_file()and not path.is_symlink()and path.stat().st_size<=32*2**20,'bounded regular report required')
    raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==expected,'independently known block audit SHA differs')
    return decode(raw),dict(file=path.name,bytes=len(raw),sha256=expected)


def finite(value):
    return type(value)in(int,float)and math.isfinite(value)


def describe(values):
    need(set(values)=={'1','2','3'}and all(finite(v)for v in values.values()),'three finite block observations required')
    data=list(values.values())
    return dict(n=3,by_block=values,minimum=min(data),median=statistics.median(data),maximum=max(data),
        sample_sd=statistics.stdev(data),scope='Descriptive variation across the three fresh allocations; no independence or population-uncertainty claim.')


def aggregate(reports,pins,auditor_sha):
    need(set(reports)=={1,2,3},'all three complete original global repetitions required')
    for block,r in reports.items():
        need(r['kind']=='INDEPENDENT_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT'
            and r['status']=='PASS_ACTUAL_E25_V6_EIGHT_APP_BLOCK_AUDIT'
            and type(r['block'])is int and r['block']==block and r['block_jobs']==8
            and r['global_protocol_jobs']==24 and r['global24_complete']is False
            and r['all_three_complete_blocks_required']is True,'only actual complete eight-app block audits can be aggregated')
        need(all(r[k]==v for k,v in PINS.items())and r['script']['sha256']==auditor_sha,'same frozen science and independently supplied audit source required')
        need(r['independent_environment_module']['sha256']=='96959b28102ea112538783843b1fe50634a733fac7d19ee77eeff54732ed7c27',
            'same independently decoded CPU boundary/envelope semantics required')
        need(r['all_app_waits_zero_normal_worker_close']is True and r['all_initial_models_exact']is True
            and r['original_summary_independently_recomputed']is True and r['source_bound_consumed_digest_equality_checked']is True
            and r['consumed_tensor_bytes_independently_rehashed']is False,'actual checked scope must remain explicit')
        need(r['platform']['block']==block,'original actual platform block differs')
    need(len({r['platform']['pod_uid']for r in reports.values()})==3,'three fresh Pod allocations required')
    need(len({r['archive']['sha256']for r in reports.values()})==3,'three distinct original raw archives required')
    reference=reports[1]
    streams=reference['source_bound_consumed_input_streams'];prefixes=reference['source_bound_semantic_prefixes_by_horizon']
    need(set(streams)==set(prefixes)==set(HORIZONS),'both horizon input identity maps required')
    for r in reports.values():
        need(r['source_bound_consumed_input_streams']==streams and r['source_bound_semantic_prefixes_by_horizon']==prefixes,
            'full consumed model-input and checkpoint-prefix identities must match across allocations')
        need(streams['short320']==prefixes['short320']['576']==prefixes['long6400']['576']
            and streams['long6400']==prefixes['long6400']['6656']
            and r['cross_horizon_input_prefix576']['short_final_stream_sha256']==streams['short320'],
            'global short/long576 prefix identity differs')
    cells={};memory={};paired={};names=set()
    for block,r in reports.items():
        need(len(r['raw8_metrics'])==8 and len(r['rows'])==4 and len(r['paired_compiled_over_native'])==4,'complete block rows required')
        local={}
        for row in r['raw8_metrics']:
            j=row['job'];key=(j['horizon'],j['workers'],j['arm'])
            need(key not in local and j['repetition']==block and j['name']==f'{j["horizon"]}_w{j["workers"]}_r{block}_{j["arm"]}'
                and j['name']not in names,'exact unique original job identity required')
            names.add(j['name']);local[key]=row['metrics']
        expected={(h,w,a)for h in HORIZONS for w in(1,8)for a in ARMS}
        need(set(local)==expected,'each allocation must include the complete eight configurations')
        for key,value in local.items():
            need(all(finite(v)and v>=0 for v in value.values())and value['launch_to_actual_wait_seconds']>0,'finite raw clock metrics required')
            cells.setdefault(key,{})[str(block)]=value
        seen_pairs=set()
        for row in r['paired_compiled_over_native']:
            key=(row['horizon'],row['workers']);left=local[(*key,ARMS[0])];right=local[(*key,ARMS[1])]
            need(key not in seen_pairs,'duplicate paired configuration');seen_pairs.add(key)
            need(row['repetition']==block and row['numerator']==ARMS[0]and row['denominator']==ARMS[1]
                and set(left)==set(right)==set(row['metrics']),'within-block ratio axes differ')
            values={}
            for metric in left:
                need(right[metric]>0,'positive within-block ratio denominator required');value=left[metric]/right[metric]
                need(math.isclose(value,row['metrics'][metric],rel_tol=1e-11,abs_tol=1e-11),'independently recomputed within-block ratio differs')
                values[metric]=value
            paired.setdefault(key,{})[str(block)]=values
        need(seen_pairs=={(h,w)for h in HORIZONS for w in(1,8)},'all four within-block paired configurations required')
        peaks=r['headline_application_PSS_all_phases']['runs'];need(len(peaks)==8,'all sampled PSS outcomes, including unobserved, required')
        seen=set()
        for row in peaks:
            j=row['job'];key=(j['horizon'],j['workers'],j['arm']);need(key in expected and key not in seen,'exact PSS configuration required');seen.add(key)
            need(j['repetition']==block,'PSS repetition differs');peak=row['maximum']
            if peak is None:need(row['status']=='unobserved'and row['eligible_samples']==0,'unknown PSS must remain unobserved');value=None
            else:
                need(row['status']=='observed'and row['eligible_samples']>0 and type(peak['tree_pss_bytes'])is int and peak['tree_pss_bytes']>=0,'observed sampled PSS maximum required');value=peak['tree_pss_bytes']
            memory.setdefault(key,{})[str(block)]=value
    need(len(names)==24,'complete global24 required')
    rows=[]
    for (h,w,a),blocks in sorted(cells.items()):
        keys=set(blocks['1']);need(all(set(v)==keys for v in blocks.values()),'same raw metrics per configuration required')
        values=memory[h,w,a];observed=[v for v in values.values()if v is not None]
        rows.append(dict(horizon=h,workers=w,arm=a,
            metrics={metric:describe({b:v[metric]for b,v in blocks.items()})for metric in sorted(keys)},
            sampled_application_PSS_bytes=dict(by_block=values,observed_blocks=len(observed),unobserved_blocks=[b for b,v in values.items()if v is None],
                complete_three_block_range=len(observed)==3,minimum_of_observed=min(observed)if observed else None,
                maximum_of_observed=max(observed)if observed else None,
                scope='Range of per-application maxima over every eligible phase; sampled/non-atomic/partial residency. Missing is unobserved, never zero.')))
    ratios=[]
    for (h,w),blocks in sorted(paired.items()):
        keys=set(blocks['1']);need(set(blocks)=={'1','2','3'}and all(set(v)==keys for v in blocks.values()),'three within-block paired metric sets required')
        ratios.append(dict(horizon=h,workers=w,numerator=ARMS[0],denominator=ARMS[1],
            metrics={metric:describe({b:v[metric]for b,v in blocks.items()})for metric in sorted(keys)},
            scope='Statistics of three within-block ratios; never a ratio of across-allocation medians.'))
    return dict(kind='E25_V6_COMPLETE_THREE_ALLOCATION_AGGREGATE',status='PASS_COMPLETE_THREE_BLOCKS_24_APPS',global24_complete=True,
        blocks=3,jobs=24,source_hashes=PINS,auditor_source_sha256=auditor_sha,block_audit_pins=pins,
        raw_archive_pins={str(b):r['archive']for b,r in reports.items()},root_collection_pins={str(b):r['root_collection']for b,r in reports.items()},
        platforms={str(b):r['platform']for b,r in reports.items()},rows=rows,paired_compiled_over_native=ratios,
        source_bound_consumed_input_streams=streams,source_bound_semantic_prefixes_by_horizon=prefixes,
        scope='All three prospective fresh complete allocation blocks, original24 apps only. Paired ratios within blocks; across-allocation variation descriptive. OS/PVC caches link allocations. No confidence interval, time-to-quality, historical/E23 causal ratio, best-possible native cost or cold-cache claim.',
        input_identity_scope='Source-bound typed/shaped/ordered consumed model-input digests, not independently rehashed tensor bytes.',
        native_scope='Evaluated fresh file-backed persistent-native path starting from compact CSR; preconverted reuse or shared-memory promotion defines different setup boundaries.',
        prior_failed_attempts_included_in_measurements=False,actual_new_experiments_cluster_GPU_YT_actions=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--block-audit',nargs=2,action='append',metavar=('JSON','SHA256'),required=True)
    parser.add_argument('--auditor-source-sha256',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    need(len(a.block_audit)==3 and not a.output.exists()and not a.output.is_symlink(),'three pinned actual audits and fresh output required')
    need(len(a.auditor_source_sha256)==64 and set(a.auditor_source_sha256)<=set('0123456789abcdef'),'explicit independent auditor source SHA required')
    reports={};pins={}
    for path,expected in a.block_audit:
        r,pin=read_pinned(path,expected);block=r['block'];need(type(block)is int and block not in reports,'three distinct block IDs required');reports[block]=r;pins[str(block)]=pin
    result=aggregate(reports,pins,a.auditor_source_sha256)
    with a.output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],jobs=24,output=str(a.output)),indent=2))


if __name__=='__main__':main()
