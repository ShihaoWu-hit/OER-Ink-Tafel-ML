"""Shared nested grouped-CV refits and stage-specific test-fold permutations.

Reuses the frozen model mathematics. New targets/deletions trigger inner
selection again. A cache is accepted only for matching inputs, engine source,
settings and numerical-library versions; cache files are trusted local data.
"""
import hashlib,json,time
import joblib,scipy,sklearn
from joblib import Parallel,delayed,parallel_config
import numpy as np
import pandas as pd
from .common import *

def _run_id(labels,drop_group,n_permutations,reselect):
    h=hashlib.sha256(labels[Q].to_csv().encode())
    for f in [Path(__file__),ENGINE_DIR/'stage_reliance.py',CORE/'ink_tafel_study/posterior_cardinal_contiguous_threeclass_presto_gate.py',
              CORE/'ink_tafel_study/posterior_cardinal_contiguous_threeclass_presto_policy_v2.py']:
        h.update(f.read_bytes())
    for f in ['records.csv','outer_split_manifest.csv','inner_split_manifest.csv','probability_selections.csv','policy_selections.csv']:
        h.update(input_path(f).read_bytes())
    h.update(json.dumps([drop_group,n_permutations,reselect,np.__version__,pd.__version__,scipy.__version__,sklearn.__version__]).encode())
    return h.hexdigest()

def _fold(repeat,fold,records,labels,inner,outer,n_permutations,reselect,run_id,cache_dir):
    path=Path(cache_dir)/f'r{repeat:02d}_f{fold:02d}.joblib'
    if path.exists():
        pack=joblib.load(path)
        if pack.get('run_id')==run_id:return pack['result'],pack['selection']
    m=engine();m.BASELINE_TOLERANCE=1e-3
    d=load_inputs();test=outer.loc[outer.repeat.eq(repeat)&outer.fold.eq(fold),'record_id'].tolist()
    train=[rid for rid in records.index if rid not in set(test)]
    assert not set(records.loc[train,'cv_group_id'])&set(records.loc[test,'cv_group_id'])
    if reselect:
        from ink_tafel_study import posterior_cardinal_contiguous_threeclass_presto_gate as gate
        from ink_tafel_study import posterior_cardinal_contiguous_threeclass_presto_policy_v2 as policy
        dev=gate._inner_oof_development(records,labels,inner,repeat=repeat,fold=fold,outer_train_ids=train)
        sel,grid=policy._select_policy_penalty_threshold(dev['candidate_oof'])
        pp=float(dev['selected_penalty']);sp=float(sel['penalty']);threshold=float(sel['threshold'])
        parent=pd.DataFrame([dict(repeat=repeat,fold=fold,selected_penalty=pp)])
        policy_sel=pd.DataFrame([dict(repeat=repeat,fold=fold,selected_policy_penalty=sp,selected_high_threshold=threshold)])
        xt=records.loc[train].reset_index(drop=True);xv=records.loc[test].reset_index(drop=True);y=labels.loc[train,Q].to_numpy(float)
        hi=m._fit_high_branch(xt,xt.cv_group_id.tolist(),xv,y,probability_penalty=pp,policy_penalty=sp)
        con=m._fit_conditional_branch(xt,xv,y);p=m._hybrid_probabilities(hi['baseline_probability'],con['baseline_probability'])
        savedp=pd.DataFrame({'record_id':test,'repeat':repeat,'fold':fold,**{col:p[:,j] for j,col in enumerate(P)}})
        savedh=pd.DataFrame({'record_id':test,'repeat':repeat,'fold':fold,'high_score':hi['baseline_policy_score'],'selected_high_threshold':threshold})
    else:
        parent=d['parent_selections'];policy_sel=d['policy_selections'];savedp,savedh=frozen_oof(d)
        pp=float(m._selection_row(parent,repeat=repeat,fold=fold).selected_penalty)
        sel=m._selection_row(policy_sel,repeat=repeat,fold=fold);sp=float(sel.selected_policy_penalty);threshold=float(sel.selected_high_threshold)
    result=m._run_outer_split(records,labels,parent,policy_sel,savedp,savedh,repeat=repeat,fold=fold,test_ids=test,n_permutations=n_permutations)
    selection=dict(repeat=repeat,fold=fold,n_train=len(train),n_test=len(test),n_train_groups=records.loc[train].cv_group_id.nunique(),
        n_test_groups=records.loc[test].cv_group_id.nunique(),selected_probability_penalty=pp,selected_policy_penalty=sp,
        selected_threshold=threshold,reselected=reselect,probability_refit_error=result['probability_max_abs_error'],
        policy_refit_error=result['policy_score_max_abs_error'])
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'run_id':run_id,'result':result,'selection':selection},path,compress=3)
    return result,selection

def _aggregate_subset(results,records,labels,n_permutations):
    # The legacy aggregator verifies ten predictions per record. For a single
    # repeat, repeat its fold results ten times; this leaves every mean unchanged.
    m=engine();use=results
    repeats=len({x['repeat'] for x in results})
    if repeats==1:use=list(results)*10
    return m._aggregate_outer_results(use,records,labels,n_permutations=n_permutations)

def run(out,*,labels=None,drop_group=None,reselect=True,n_permutations=100,jobs=3,smoke=False,bootstrap=0):
    out=Path(out);d=load_inputs();records=d['records'];labels=d['labels'] if labels is None else labels
    inner=d['inner'];outer=d['outer']
    if drop_group is not None:
        keep=records.index[~records.cv_group_id.eq(drop_group)];records=records.loc[keep];labels=labels.loc[keep]
        inner=inner[inner.record_id.isin(keep)];outer=outer[outer.record_id.isin(keep)]
    labels=labels.loc[records.index]
    key=_run_id(labels,drop_group,n_permutations,reselect)
    splits=list(outer.groupby(['repeat','fold']).groups)
    if smoke:splits=splits[:1]
    start=time.time()
    with parallel_config(backend='loky',inner_max_num_threads=1):
        packs=Parallel(n_jobs=jobs,verbose=5)(delayed(_fold)(int(r),int(f),records,labels,inner,outer,n_permutations,reselect,key,out/'fold_cache') for r,f in splits)
    results=[x[0] for x in packs];selections=[x[1] for x in packs];save_csv(pd.DataFrame(selections),out/'fold_selections.csv')
    if smoke:
        summary={'smoke_only':True,'n_outer_folds':len(packs),'elapsed_seconds':time.time()-start};save_json(summary,out/'summary.json');return summary
    m=engine();agg=_aggregate_subset(results,records,labels,n_permutations)
    baseline,permutations,derived=m._observed_and_permutation_metrics(agg)
    metrics=permutations.groupby('feature',sort=False)[['importance_soft_log_loss','importance_soft_f0_5']].mean().reset_index()
    metrics['probability_rank']=metrics.importance_soft_log_loss.rank(method='min',ascending=False).astype(int)
    metrics['screening_rank']=metrics.importance_soft_f0_5.rank(method='min',ascending=False).astype(int)
    save_csv(metrics,out/'feature_importance.csv');save_csv(permutations,out/'permutation_metrics.csv')
    baseframe=pd.DataFrame({'record_id':agg['record_ids'] if 'record_ids' in agg else records.index,'cv_group_id':records.cv_group_id.to_numpy(),
        **{col:agg['baseline_probability'][:,j] for j,col in enumerate(P)},'mean_margin':agg['baseline_margin']})
    save_csv(baseframe,out/'record_oof.csv')
    repeated=[]
    for repeat in sorted({x['repeat'] for x in results}):
        ar=_aggregate_subset([x for x in results if x['repeat']==repeat],records,labels,n_permutations)
        br,pr,dr=m._observed_and_permutation_metrics(ar)
        mr=pr.groupby('feature',sort=False)[['importance_soft_log_loss','importance_soft_f0_5']].mean().reset_index()
        mr['repeat']=repeat;mr['rank']=mr.importance_soft_log_loss.rank(method='first',ascending=False).astype(int);repeated.append(mr)
    save_csv(pd.concat(repeated,ignore_index=True),out/'repeat_feature_importance.csv')
    if bootstrap:
        groups=records.cv_group_id.to_numpy();unique=np.array(sorted(set(groups)),object);lookup={g:np.flatnonzero(groups==g) for g in unique}
        q=labels[Q].to_numpy(float);base_loss=-np.sum(q*np.log(np.clip(agg['baseline_probability'],1e-15,1)),axis=1)
        delta={spec.feature:np.mean(-np.sum(q[None,:,:]*np.log(np.clip(agg['permuted_probability'][spec.feature],1e-15,1)),axis=2),axis=0)-base_loss for spec in m.FEATURES}
        rng=np.random.default_rng(m.BOOTSTRAP_SEED);rows=[]
        for i in range(bootstrap):
            ids=np.concatenate([lookup[g] for g in rng.choice(unique,size=len(unique),replace=True)])
            rows.extend(dict(replicate=i,feature=f,importance_soft_log_loss=float(v[ids].mean())) for f,v in delta.items())
        save_csv(pd.DataFrame(rows),out/'group_bootstrap.csv')
    summary={'n_records':len(records),'n_groups':records.cv_group_id.nunique(),'n_outer_folds':len(results),'permutations':n_permutations,
        'n_bootstrap':bootstrap,'reselected_in_each_outer_training_set':reselect,'deleted_group':drop_group,'baseline':baseline,
        'solvent':metrics[metrics.feature.eq(FEATURE)].iloc[0].to_dict(),'elapsed_seconds':time.time()-start,'run_id':key}
    save_json(summary,out/'summary.json');return summary
