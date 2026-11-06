"""S7.5: High-focus policy-score reliance using training-marginal replacements."""
import numpy as np
import pandas as pd
from .common import *

def fast_perturb(fitted,engineered_train,engineered_test,target,threshold,*,split_kind,split_id,seed,n_permutations):
    source=engine('high_focus');hg=source.HG;target=np.asarray(target,float);focus=target>=source.HIGH_THRESHOLD
    baseline=source._policy_score(fitted,engineered_test);rows=[];nt=len(target)
    def losses(s,mask):
        s=np.clip(s[...,mask],1e-9,1-1e-9);q=target[mask]
        return -(q*np.log(s)+(1-q)*np.log1p(-s)).mean(axis=-1)
    allmask=np.ones(nt,bool)
    for index,spec in enumerate(hg.FEATURES):
        row=dict(split_kind=split_kind,split_id=str(split_id),feature=spec.feature,display_name=spec.display_name,
            source_columns=';'.join(spec.source_columns),high_gate_column=spec.high_column or '',n_test=nt,n_high_focus=int(focus.sum()),
            threshold=threshold,baseline_soft_log_loss_all=float(losses(baseline,allmask)),baseline_mean_margin_all=float((baseline-threshold).mean()),
            baseline_soft_log_loss_high_focus=float(losses(baseline,focus)) if focus.any() else np.nan,
            baseline_mean_margin_high_focus=float((baseline[focus]-threshold).mean()) if focus.any() else np.nan,n_permutations=n_permutations)
        if spec.high_column is None or spec.high_column not in engineered_test:
            row.update(importance_soft_log_loss_all=0.,importance_soft_log_loss_high_focus=0.,importance_mean_margin_change_all=0.,
                importance_mean_margin_change_high_focus=0.,positive_importance_fraction_all=1.,positive_importance_fraction_high_focus=1.,structural_status='not_used_by_high_gate')
        else:
            rng=np.random.default_rng(seed+1009*index)
            batch=pd.concat([engineered_test]*n_permutations,ignore_index=True)
            batch[spec.high_column]=rng.choice(engineered_train[spec.high_column].to_numpy(),size=n_permutations*nt,replace=True)
            score=source._policy_score(fitted,batch).reshape(n_permutations,nt)
            la=losses(score,allmask);lf=losses(score,focus) if focus.any() else np.array([np.nan])
            row.update(importance_soft_log_loss_all=float(la.mean()-row['baseline_soft_log_loss_all']),
                importance_soft_log_loss_high_focus=float(lf.mean()-row['baseline_soft_log_loss_high_focus']),
                importance_mean_margin_change_all=float((score-baseline).mean()),
                importance_mean_margin_change_high_focus=float((score[:,focus]-baseline[focus]).mean()) if focus.any() else np.nan,
                positive_importance_fraction_all=float((la>row['baseline_soft_log_loss_all']).mean()),
                positive_importance_fraction_high_focus=float((lf>row['baseline_soft_log_loss_high_focus']).mean()) if focus.any() else np.nan,
                structural_status='active_high_gate_feature')
        rows.append(row)
    return rows

def run(out,mode='all',smoke=False):
    out=Path(out);d=load_inputs();src=engine('high_focus')
    data={'records':d['records'].reset_index(drop=True),'labels':d['labels'],'outer':d['outer'],
          'policy_selection':d['policy_selections'],'probability_selection':d['parent_selections']}
    # Vectorization changes only execution, not RNG samples, targets or scores.
    src._perturbation_rows=fast_perturb
    if smoke:
        data['outer']=data['outer'].query('repeat==0 and fold==0');src.N_PERMUTATIONS_REPEAT=3
    if mode in ['all','repeat']:
        result=src._run_repeated_cv(data)
        for k,v in result.items():
            if isinstance(v,pd.DataFrame):save_csv(v,out/f'repeat_{k}.csv')
        print('repeat output keys',list(result),flush=True)
    if mode in ['all','loco'] and not smoke:
        result=src._run_loco(data)
        for k,v in result.items():
            if isinstance(v,pd.DataFrame):save_csv(v,out/f'loco_{k}.csv')
    save_json({'mode':mode,'smoke_only':smoke,'metric':'binary soft log loss of policy score on High-focus records',
       'perturbation':'replacement from outer-training marginal','seed':src.SEED,
       'repeat_replacements':src.N_PERMUTATIONS_REPEAT,'loco_replacements':src.N_PERMUTATIONS_LOCO},out/'analysis_contract.json')
