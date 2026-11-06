"""Paths, immutable inputs and helpers. No source-machine path is required."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
from repository import ROOT, CORE, INPUT_FILES, input_path
Q = ['q_low_60_side', 'q_120_side', 'q_high_off']
P = ['p_low_60_side', 'p_120_side', 'p_high_off']
FEATURE = 'solvent_polarity_coordinate'
ENGINE_DIR = Path(__file__).resolve().parents[1] / 'engines'

def read_csv(path):
    return pd.read_csv(path, dtype={'record_id':str,'curve_id':str,'cv_group_id':str,'held_out_group':str})

def load_inputs():
    out={k:read_csv(input_path(f)) for k,f in {
        'records':'records.csv','labels':'soft_labels.csv','outer':'outer_split_manifest.csv',
        'inner':'inner_split_manifest.csv','parent_selections':'probability_selections.csv',
        'policy_selections':'policy_selections.csv','probability_oof':'oof_probabilities.csv',
        'policy_oof':'policy_oof_decisions.csv'}.items()}
    for k in ['records','labels']:out[k]=out[k].set_index('record_id',drop=False)
    assert len(out['records'])==len(out['labels'])==97
    assert out['records'].cv_group_id.nunique()==80
    assert np.allclose(out['labels'][Q].sum(axis=1),1)
    assert not out['outer'].groupby(['repeat','cv_group_id']).fold.nunique().gt(1).any()
    return out

def engine(name='stage_reliance'):
    module_name='_s7_'+name
    if module_name in sys.modules:return sys.modules[module_name]
    path=ENGINE_DIR/f'{name}.py'
    spec=importlib.util.spec_from_file_location(module_name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[module_name]=mod;spec.loader.exec_module(mod)
    return mod

def frozen_oof(data=None):
    data=load_inputs() if data is None else data;m=engine()
    prob=data['probability_oof'];pol=data['policy_oof']
    if 'evaluation_model' in prob:prob=prob[prob.evaluation_model.eq(m.EXPECTED_PROBABILITY_MODEL)]
    if 'decision_policy' in pol:pol=pol[pol.decision_policy.eq(m.EXPECTED_POLICY)]
    assert len(prob)==len(pol)==970
    return prob,pol

def save_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,float_format='%.12g')

def save_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')

def load_draws():
    frame=read_csv(input_path('posterior_target_draws.csv.gz'))
    counts=frame.groupby('curve_id').size()
    assert len(counts)==97 and counts.eq(8000).all()
    return {str(k):g.slope_mV_dec.to_numpy(float) for k,g in frame.groupby('curve_id',sort=False)}

def regional_targets(lower,upper,split=None):
    data=load_inputs();draws=load_draws();rows=[]
    for rid,rec in data['records'].iterrows():
        x=draws[rec.curve_id]
        vals=[np.mean(x<lower),np.mean((x>=lower)&(x<=upper)),np.mean(x>upper)]
        row={**data['labels'].loc[rid].to_dict(),'record_id':rid,**dict(zip(Q,vals))}
        row.update(threeclass_modal_label=['low_60_side','120_side','high_off'][int(np.argmax(vals))],
            threeclass_max_probability=float(max(vals)),threeclass_entropy=float(-sum(v*np.log(v) for v in vals if v>0)),
            lower_boundary_mV_dec=float(lower),upper_boundary_mV_dec=float(upper),
            label_scheme_id=f'S7_L{lower:.8g}_U{upper:.8g}')
        if split is not None:row.update(q_lower_low=np.mean(x<split),q_upper_low=np.mean((x>=split)&(x<lower)))
        rows.append(row)
    return pd.DataFrame(rows).set_index('record_id',drop=False)
