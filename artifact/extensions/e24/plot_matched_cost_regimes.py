#!/usr/bin/env python3
"""Single-column figure from existing matched-cohort observations only."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'recorded/derived_cost_regimes.json'
COLORS = {'Friendster_E20':'#0072B2', 'Citation2_strict_clean_E22':'#D55E00'}
LABELS = {'Friendster_E20':'Friendster', 'Citation2_strict_clean_E22':'Citation2'}
MARKERS = {50:'o',200:'^'}
STEM = 'matched_cost_regimes'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--out',type=Path,default=HERE)
    args = parser.parse_args(argv)
    source = args.input.resolve()
    raw = json.loads(source.read_text())
    datasets = raw['datasets']
    assert {d['dataset']for d in datasets} == set(COLORS)
    assert all(d['matched_cases'] == len(d['cases']) == 128 for d in datasets)
    assert sum(len(d['cases'])for d in datasets) == 256
    assert raw['thresholds_optimized'] == 0
    plotted, counts, points = [], {}, []
    for d in datasets:
        assert len({c['archive_record_index']for c in d['cases']}) == 128
        for c in d['cases']:
            assert c['cap'] in MARKERS
            assert all(math.isfinite(c[key]) and c[key]>0 for key in ('Ds_over_Q','scan_over_pair'))
            points.append((c['Ds_over_Q'],c['scan_over_pair']))
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
        'xtick.major.size':2.8,'ytick.major.size':2.8,
        'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    })
    fig, ax = plt.subplots(figsize=(3.4,2.25))
    fig.subplots_adjust(left=.167,right=.985,bottom=.215,top=.84)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.12,21)
    ax.set_ylim(.32,3.55)
    ax.axvline(1,color='#666666',linewidth=.7,linestyle=(0,(3,2)),zorder=1)
    ax.axhline(1,color='#666666',linewidth=.7,linestyle=(0,(3,2)),zorder=1)
    # Fixed styling and original coordinates only: no jitter, fitted curves,
    # confidence bands, threshold calibration, transparency or point filtering.
    for cap in (50,200):
        for d in datasets:
            cases = [c for c in d['cases']if c['cap']==cap]
            handle = ax.scatter([c['Ds_over_Q']for c in cases],
                [c['scan_over_pair']for c in cases],s=12.5 if cap==50 else 14,
                marker=MARKERS[cap],facecolors=COLORS[d['dataset']],
                edgecolors='white',linewidths=.22,alpha=1.0,zorder=3)
            assert len(handle.get_offsets()) == len(cases)
            key=d['dataset']+'/cap'+str(cap)
            counts[key]=len(cases)
            plotted.extend((d['dataset'],c['archive_record_index'])for c in cases)
    assert len(plotted)==len(set(plotted))==256
    for x,y in points:
        assert ax.get_xlim()[0]<x<ax.get_xlim()[1]
        assert ax.get_ylim()[0]<y<ax.get_ylim()[1]
    ax.set_xlabel(r'$D_S/Q$',labelpad=2)
    ax.set_ylabel('Scan / pair edge time',labelpad=3)
    ax.xaxis.set_major_locator(FixedLocator([.2,.5,1,2,5,10]))
    ax.yaxis.set_major_locator(FixedLocator([.4,.7,1,2,3]))
    formatter=FuncFormatter(lambda value,pos:f'{value:g}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='minor',length=0)
    ax.text(.025,.96,'pair faster',transform=ax.transAxes,ha='left',va='top',color='#4D4D4D')
    ax.text(.975,.045,'scan faster',transform=ax.transAxes,ha='right',va='bottom',color='#4D4D4D')
    handles=[
        Line2D([],[],color=COLORS['Friendster_E20'],linewidth=2.2,label='Friendster'),
        Line2D([],[],color=COLORS['Citation2_strict_clean_E22'],linewidth=2.2,label='Citation2'),
        Line2D([],[],linestyle='none',marker='o',markersize=3.7,color='#333333',label='cap 50'),
        Line2D([],[],linestyle='none',marker='^',markersize=3.9,color='#333333',label='cap 200'),
    ]
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.548,.997),
        frameon=False,handlelength=.85,handletextpad=.35,columnspacing=.8,borderpad=0)
    fig.canvas.draw()
    # Detect any figure-level text outside the actual publication-size canvas.
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    clipped=[]
    for obj in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*fig.legends]:
        box=obj.get_window_extent(renderer)
        if box.x0<bounds.x0-1 or box.y0<bounds.y0-1 or box.x1>bounds.x1+1 or box.y1>bounds.y1+1:
            clipped.append(type(obj).__name__)
    assert not clipped,clipped
    for suffix in ('png','pdf','svg'):
        kwargs={'dpi':360} if suffix=='png' else {}
        fig.savefig(args.out/(STEM+'.'+suffix),**kwargs)
    summary=[{
        'dataset':d['dataset'],'cases':len(d['cases']),'wins':d['wins'],
        'spearman':d['spearman_Ds_over_Q_vs_scan_over_pair'],
        'frozen_proxy_agreement':d['fixed_proxy']['agreement'],
        'best_constant_over_proxy':d['factors_baseline_time_over_policy_time']['frozen_proxy'][
            min(('always_scan','always_pairwise'),key=lambda key:d['sums_of_case_median_edge_seconds'][key])]
    }for d in datasets]
    manifest={
        'kind':'MATCHED128_TWO_DATASET_SCIENTIFIC_FIGURE',
        'status':'RENDERED_FROM_RECORDED_OBSERVATIONS',
        'input':{'path':str(source),'bytes':source.stat().st_size,'sha256':sha(source)},
        'script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'render':{'python':sys.executable,'matplotlib':matplotlib.__version__,
            'width_inches':3.4,'height_inches':2.25,'font_points':8,'png_dpi':360},
        'semantics':{'x':'Ds/Q','y':'median scan edge time / median pairwise edge time',
            'both_axes':'logarithmic','vertical_line':1,'horizontal_line':1,
            'proxy_alpha':1,'threshold_fit':False,'fitted_curve':False,
            'confidence_intervals':False,'jitter':False,'opacity':1.0},
        'coverage':{'input_cases':256,'plotted_cases':256,'omitted_cases':0,
            'unique_dataset_record_keys':len(set(plotted)),'group_counts':counts,
            'all_centers_inside_axis_limits':True,'all_text_within_canvas':not clipped},
        'summaries_copied_from_input':summary,
        'limits':['Posthoc fixed-cohort repeated-cache observations; no population confidence interval.',
            'Datasets retain separate cohorts; query pairs and two Citation2 ordered sets repeat.',
            'Policy factors reuse per-case medians and exclude dispatch overhead.'],
        'outputs':{suffix:{'bytes':(args.out/(STEM+'.'+suffix)).stat().st_size,
            'sha256':sha(args.out/(STEM+'.'+suffix))}for suffix in ('png','pdf','svg')},
        'canonical_figures_LaTeX_repo_or_published_v0_5_modified':False,
        'visual_QA':'LOCAL_VISUAL_INSPECTION_REQUIRED_FOR_NEW_RENDER'
    }
    (args.out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plt.close(fig)
    print(json.dumps({'outputs':manifest['outputs'],'coverage':manifest['coverage']},indent=2))


if __name__=='__main__':main()
