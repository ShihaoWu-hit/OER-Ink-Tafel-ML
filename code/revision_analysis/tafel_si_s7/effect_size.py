"""Cross-fitted ALE responses and connected-group bootstrap intervals."""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel,delayed,parallel_config
from .common import *

def _ale():
    m=engine();m.BASELINE_TOLERANCE=1e-3
    sys.modules['run_feature_importance']=m
    return engine('ale')

def _fold(repeat,fold):
    a=_ale();d=load_inputs();p,h=frozen_oof(d)
    ids=d['outer'].loc[d['outer'].repeat.eq(repeat)&d['outer'].fold.eq(fold),'record_id'].tolist()
    return a._run_outer_split(d['records'],d['labels'],d['parent_selections'],d['policy_selections'],p,h,repeat=repeat,fold=fold,test_ids=ids)

def run(out,jobs=3,bootstrap=2000,smoke=False):
    out=Path(out);d=load_inputs();a=_ale();keys=list(d['outer'].groupby(['repeat','fold']).groups)
    if smoke:keys=keys[:1]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=jobs,verbose=5)(delayed(_fold)(int(r),int(f)) for r,f in keys)
    local=pd.concat([x['local'] for x in results],ignore_index=True);support=pd.concat([x['support'] for x in results],ignore_index=True)
    save_csv(local,out/'local_differences.csv');save_csv(support,out/'raw_support.csv')
    if smoke:return {'smoke_only':True}
    rb,rs,_=a._record_aggregate(local,support);summary=a._support_summary(support,rs,rb)
    assert summary.support_gate_pass.all()
    point=pd.concat([a._curve_from_record_bins(rb,feature=s['feature']) for s in a.FEATURES],ignore_index=True)
    boot,invalid=a._bootstrap_curves(rb,d['records'].cv_group_id.astype(str).tolist(),repetitions=bootstrap)
    intervals=boot.groupby(['feature','class_label','bin_index'],as_index=False).ale_probability.agg(
        ci_lower=lambda x:np.quantile(x,.025),ci_upper=lambda x:np.quantile(x,.975),bootstrap_median='median',n_valid_bootstrap='count')
    point=point.merge(intervals,on=['feature','class_label','bin_index'],validate='one_to_one')
    save_csv(point,out/'cross_fitted_ale.csv');save_csv(summary,out/'cross_fitted_ale_support.csv');save_csv(rb,out/'record_bin_differences.csv')
    save_csv(boot,out/'group_bootstrap_curves.csv')
    upper=point[point.feature.eq(FEATURE)&point.class_label.eq('high_off')&point.bin_index.eq(4)].iloc[0].to_dict()
    save_json({'upper_solvent_bin':upper,'bootstrap_repetitions':bootstrap,'invalid_bootstraps':invalid,'hard_alert_ale_performed':False},out/'summary.json')
    return upper
