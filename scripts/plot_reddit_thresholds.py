"""One-panel threshold tradeoff from measured frozen-holdout aggregates.

Contract: show that short-record exclusion reduces record count faster than
own-text volume. Single quantitative panel, Python only, 183 x 120 mm,
editable PDF/SVG and 600 dpi PNG. No semantic-accuracy or optimality claim.
"""
import argparse
import csv
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa-scripts', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out = Path(__file__).resolve().parents[1] / 'results/reddit-threshold-audit-2026-09-25'
    with (out/'word-threshold-curve.csv').open() as h:
        rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(h)]
    assert len(rows) == 101
    assert all(a['min_words'] < b['min_words'] and a['retained'] >= b['retained'] for a,b in zip(rows,rows[1:]))
    # Display the decision-relevant 0..50 range; the source CSV retains 0..100.
    display = [r for r in rows if r['min_words'] <= 50]
    x = [r['min_words'] for r in display]
    y = [r['retention_pct'] for r in display]
    mpl.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9, 'axes.labelsize':9, 'xtick.labelsize':8,
                         'ytick.labelsize':8, 'legend.fontsize':8,
                         'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.right':False,'axes.spines.top':False})
    fig, ax = plt.subplots(figsize=(7.2047244094,4.7244094488))
    fig.subplots_adjust(left=.11,right=.96,bottom=.26,top=.80)
    fig.text(.11,.94,'Minimum-word threshold: record count and text volume',fontsize=11,weight='bold')
    fig.text(.11,.895,'Frozen random sample: 9,992 records; 9,407 pass enhanced denoising',fontsize=8)
    ax.plot(x,y,color='#286A98',lw=2,label='Records retained / original sample')
    ax.fill_between(x,[r['retention_ci_low'] for r in display],[r['retention_ci_high'] for r in display],
                    color='#286A98',alpha=.14,lw=0)
    ax.plot(x,[r['candidate_chars_retained_pct'] for r in display],color='#62666B',lw=1.8,
            ls='--',label='Text characters retained / enhanced candidates')
    for threshold, offset in [(5,(0,-25)),(10,(0,-25)),(20,(0,-25))]:
        row=rows[threshold]
        ax.scatter([threshold],[row['retention_pct']],s=22,color='#286A98',zorder=3)
        ax.annotate(f"x={threshold}: {row['retention_pct']:.1f}%",(threshold,row['retention_pct']),
                    xytext=offset,textcoords='offset points',fontsize=8,color='#286A98',
                    ha='center')
    ax.set(xlim=(0,50),ylim=(0,105),xlabel='Minimum words per record (x)',ylabel='Retained (%)')
    ax.set_xticks(range(0,51,5)); ax.set_yticks(range(0,101,20))
    ax.grid(axis='y',color='#E5E7EA',lw=.6); ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.10,.10),frameon=False,ncol=1)
    fig.text(.11,.064,'Blue band: pointwise 95% Wilson interval; words counted by whitespace, without context.',fontsize=7)
    fig.text(.11,.030,'Characters are not billed tokens. CSV includes x=0–100. This curve does not measure semantic accuracy.',fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(out/'threshold-curve.alignment.json'),
        overlay_svg=str(out/'threshold-curve.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(out/'threshold-curve.pdf')
    fig.savefig(out/'threshold-curve.svg')
    fig.savefig(out/'threshold-curve.png',dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
