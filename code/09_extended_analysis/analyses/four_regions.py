"""Nested Low-region subdivision on the grouped outer splits."""
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from .common import *

def _one(repeat,fold):
    d=load_inputs();m=engine();q4=regional_targets(np.sqrt(60*120),144.,np.sqrt(40*60))
    names=['q_lower_low','q_upper_low','q_120_side','q_high_off']
    ids=d['outer'].loc[d['outer'].repeat.eq(repeat)&d['outer'].fold.eq(fold),'record_id'].tolist()
    tr=[x for x in d['records'].index if x not in set(ids)]
    xt=d['records'].loc[tr].reset_index(drop=True);xv=d['records'].loc[ids].reset_index(drop=True)
    assert not set(xt.cv_group_id)&set(xv.cv_group_id)
    par=m._selection_row(d['parent_selections'],repeat=repeat,fold=fold)
    pol=m._selection_row(d['policy_selections'],repeat=repeat,fold=fold)
    high=m._fit_high_branch(xt,xt.cv_group_id.tolist(),xv,d['labels'].loc[tr,Q].to_numpy(float),
        probability_penalty=float(par.selected_penalty),policy_penalty=float(pol.selected_policy_penalty))
    builder,et,ev,_=m._fold_local_engineered_features(xt,xv,panel='context_plus_ionomer_ratio',rare_min_frequency=m.RARE_MIN_FREQUENCY)
    est=m.SoftGowerKNeighborsClassifier(numeric_columns=builder.numeric_columns_out_,categorical_columns=builder.categorical_columns_out_,
        class_names=['Lower-Low','Upper-Low','Intermediate','High'],blocks=builder.get_blocks_out(),n_neighbors=max(m.NEIGHBOR_COUNTS),
        distance_power=m.DIRECT_DISTANCE_POWER,epsilon=m.DIRECT_EPSILON,neighbor_mode='knn',bandwidth=1.).fit(et,q4.loc[tr,names].to_numpy(float))
    distances=est.distance_model_._pairwise_distances(ev)
    direct=np.array([np.mean([est._knn_probability(row,min(k,len(tr))) for k in m.NEIGHBOR_COUNTS],axis=0) for row in distances])
    cond=direct[:,:3]/direct[:,:3].sum(axis=1,keepdims=True)
    h=high['baseline_probability'][:,2];p=np.column_stack(((1-h)[:,None]*cond,h))
    return pd.DataFrame({'record_id':ids,'repeat':repeat,'fold':fold,**{f'p{i}':p[:,i] for i in range(4)},
        'margin':high['baseline_policy_score']-float(pol.selected_high_threshold)})

def scores(y,p):return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),
    'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0))}

def run(out,jobs=2,smoke=False):
    out=Path(out);d=load_inputs();keys=list(d['outer'].groupby(['repeat','fold']).groups)
    if smoke:keys=keys[:1]
    frames=Parallel(n_jobs=jobs,verbose=5)(delayed(_one)(int(r),int(f)) for r,f in keys)
    oof=pd.concat(frames,ignore_index=True);save_csv(oof,out/'four_region_oof.csv')
    if smoke:return {'smoke_only':True,'n_outer_folds':len(keys),'n_oof_rows':len(oof)}
    assert len(oof)==970 and oof.groupby(['repeat','record_id']).size().eq(1).all()
    means=oof.groupby('record_id')[[f'p{i}' for i in range(4)]+['margin']].mean().reindex(d['records'].index)
    q=regional_targets(np.sqrt(60*120),144.,np.sqrt(40*60)).reindex(means.index)
    y4=q[['q_lower_low','q_upper_low','q_120_side','q_high_off']].to_numpy().argmax(axis=1)
    p4=means[[f'p{i}' for i in range(4)]].to_numpy();alert=means.margin.to_numpy()>=0
    pred4=np.where(alert,3,p4[:,:3].argmax(axis=1))
    merged=np.column_stack((p4[:,0]+p4[:,1],p4[:,2],p4[:,3]))
    pred3=np.where(alert,2,merged[:,:2].argmax(axis=1));y3=d['labels'].loc[means.index,Q].to_numpy().argmax(axis=1)
    old,oldpolicy=frozen_oof(d);oldmean=old.groupby('record_id')[P].mean().reindex(means.index)
    oldpolicy=oldpolicy.copy();oldpolicy['margin_calc']=oldpolicy.high_score-oldpolicy.selected_high_threshold
    oldmargin=oldpolicy.groupby('record_id').margin_calc.mean().reindex(means.index).to_numpy()
    oldpred=np.where(oldmargin>=0,2,oldmean[P[:2]].to_numpy().argmax(axis=1))
    cm=confusion_matrix(y4,pred4,labels=[0,1,2,3]);save_csv(pd.DataFrame(cm,columns=['Lower-Low','Upper-Low','Intermediate','High']),out/'confusion_four_regions.csv')
    detail=means.reset_index();detail['target4']=y4;detail['prediction4']=pred4;detail['prediction3_merged']=pred3;detail['original_prediction3']=oldpred
    save_csv(detail,out/'record_decisions.csv')
    summary={'n_records':97,'n_outer_folds':50,'n_repeats':10,'n_lower_low':int((y4==0).sum()),
        'correct_lower_low':int(((y4==0)&(pred4==0)).sum()),'changed_common_three_class_decisions':int((pred3!=oldpred).sum()),
        'changed_high_alerts':int((alert!=(oldmargin>=0)).sum()),'max_merged_probability_difference':float(np.max(abs(merged-oldmean[P].to_numpy()))),
        'original_three_class':scores(y3,oldpred),'merged_four_region':scores(y3,pred3),'confusion_four_regions':cm.tolist()}
    save_json(summary,out/'summary.json');bootstrap_scores(out);return summary


def bootstrap_scores(out,repetitions=2000,seed=20261021):
    """Paired connected-group percentile intervals from fixed record decisions."""
    out=Path(out);d=load_inputs();dec=read_csv(out/'record_decisions.csv').set_index('record_id')
    ids=d['records'].index;dec=dec.loc[ids];y=d['labels'].loc[ids,Q].to_numpy().argmax(axis=1)
    groups=d['records'].cv_group_id.to_numpy();unique=np.array(sorted(set(groups)),object)
    lookup={g:np.flatnonzero(groups==g) for g in unique};rng=np.random.default_rng(seed);rows=[]
    for k in range(repetitions):
        ix=np.concatenate([lookup[g] for g in rng.choice(unique,len(unique),replace=True)])
        for model,col in [('Original three-region','original_prediction3'),('Merged four-region','prediction3_merged')]:
            yp=dec[col].to_numpy();cm=confusion_matrix(y[ix],yp[ix],labels=[0,1,2]).astype(float)
            tp=cm.diagonal();support=cm.sum(axis=1);pred=cm.sum(axis=0)
            recall=np.divide(tp,support,out=np.full(3,np.nan),where=support>0)
            f1=np.divide(2*tp,support+pred,out=np.zeros(3),where=(support+pred)>0)
            rows.append(dict(replicate=k,model=model,accuracy=float(tp.sum()/cm.sum()),
                balanced_accuracy=float(np.nanmean(recall)),macro_f1=float(f1.mean())))
    b=pd.DataFrame(rows);save_csv(b,out/'paired_group_bootstrap.csv');intervals=[]
    for model,col in [('Original three-region','original_prediction3'),('Merged four-region','prediction3_merged')]:
        pt=scores(y,dec[col].to_numpy());sub=b[b.model.eq(model)]
        for metric,v in pt.items():
            lo,hi=np.quantile(sub[metric],[.025,.975]);intervals.append(dict(model=model,metric=metric,estimate=v,ci_lower=lo,ci_upper=hi))
    save_csv(pd.DataFrame(intervals),out/'common_three_class_intervals.csv')
    save_json(dict(repetitions=repetitions,seed=seed,unit='connected group',paired=True,refitted=False),out/'bootstrap_contract.json')
    return intervals
