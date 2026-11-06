"""S7.1 and S7.5: integrate all 8,000 draws, without creating ML samples."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .common import *

def run(out):
    out=Path(out);data=load_inputs();draws=load_draws();cut=np.sqrt(40.*60.)
    rows=[]
    for rid,rec in data['records'].iterrows():
        x=draws[rec.curve_id];lo,hi=np.quantile(x,[.05,.95]);q=data['labels'].loc[rid,Q].to_numpy(float)
        rows.append(dict(record_id=rid,curve_id=rec.curve_id,cv_group_id=rec.cv_group_id,
            original_modal_region=int(np.argmax(q)),p_below_split=float(np.mean(x<cut)),
            posterior_median=float(np.median(x)),ci90_low=lo,ci90_high=hi,
            covers_40_and_60=bool(lo<=40 and hi>=60)))
    df=pd.DataFrame(rows);low=df[df.original_modal_region.eq(0)]
    count=lambda f:{'n':len(f),'below_with_p_gt_0_9':int(f.p_below_split.gt(.9).sum()),
        'ambiguous_0_1_to_0_9':int(f.p_below_split.between(.1,.9).sum()),
        'above_with_p_gt_0_9':int(f.p_below_split.lt(.1).sum()),'ci90_covers_40_and_60':int(f.covers_40_and_60.sum())}
    pooled=np.concatenate(list(draws.values()))
    # Equal draws per record => the pooled density is exactly an equal-record mixture.
    # Fine-grid Gaussian convolution: dx=0.02, no downsampling; retain the full support.
    dx=.02;edges=np.arange(np.floor(pooled.min())-80,np.ceil(pooled.max())+80+dx,dx)
    hist,_=np.histogram(pooled,bins=edges);grid=(edges[1:]+edges[:-1])/2
    density=pd.DataFrame({'slope_mV_dec':np.arange(0,160.0001,.2)})
    modes={}
    for h in [2.5,5.,10.]:
        y=gaussian_filter1d(hist.astype(float)/(len(pooled)*dx),h/dx,mode='constant',truncate=8)
        density[f'bandwidth_{h:g}']=np.interp(density.slope_mV_dec,grid,y)
        modes[str(h)]=float(density.loc[density[f'bandwidth_{h:g}'].idxmax(),'slope_mV_dec'])
    summary={'cut_mV_dec':cut,'draws_per_record':8000,'all':count(df),'original_low':count(low),
        'mixture_mass_below_split':float(np.mean(pooled<cut)),'mass_below_zero':float(np.mean(pooled<0)),
        'mass_above_160':float(np.mean(pooled>160)),'medians_above_160':int(df.posterior_median.gt(160).sum()),
        'density_modes_mV_dec':modes,'density_method':'Gaussian convolution of full histogram, dx=0.02 mV/dec; no truncation or renormalization to displayed range'}
    save_csv(df,out/'record_resolution.csv');save_csv(density,out/'mixture_density.csv');save_json(summary,out/'summary.json')
    boundary=[]
    for upper in [132.,138.,144.,150.,156.]:
        for rid,rec in data['records'].iterrows():
            q=float(np.mean(draws[rec.curve_id]>upper))
            boundary.append(dict(boundary_mV_dec=upper,record_id=rid,cv_group_id=rec.cv_group_id,q_high=q,high_focus=q>=.5))
    b=pd.DataFrame(boundary);base=set(b.loc[b.boundary_mV_dec.eq(144)&b.high_focus,'record_id']);s=[]
    for u,g in b.groupby('boundary_mV_dec'):
        f=g[g.high_focus];ids=set(f.record_id)
        s.append(dict(boundary_mV_dec=u,n_records=len(f),n_groups=f.cv_group_id.nunique(),
            same_records_as_144=ids==base,record_ids=';'.join(sorted(ids,key=int)),group_ids=';'.join(sorted(set(f.cv_group_id)))))
    save_csv(b,out/'high_membership_by_boundary.csv');save_csv(pd.DataFrame(s),out/'high_boundary_summary.csv')
    keys=[];groups=sorted(set(data['records'].loc[list(base),'cv_group_id']))
    for i,gid in enumerate(groups,1):
        sub=data['records'][data['records'].cv_group_id.eq(gid)]
        for rid,r in sub.iterrows():keys.append(dict(group_label=f'G{i}',cv_group_id=gid,record_id=rid,curve_id=r.curve_id,
            high_focus=rid in base,source_doi=r.get('doi',r.get('source_doi','')),sample_id=r.get('sample_id','')))
    save_csv(pd.DataFrame(keys),out/'high_group_registry.csv')
    return summary
