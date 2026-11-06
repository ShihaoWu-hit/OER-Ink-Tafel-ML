"""Posterior reintegration and nested retraining across region boundaries."""
import numpy as np
import pandas as pd
from .common import *
from .model_pipeline import run as run_cv

def run(out,jobs=3,n_permutations=100,smoke=False):
    out=Path(out);rows=[]
    grid=[(l,u) for l in [72.,np.sqrt(60*120),96.] for u in [120.,132.,144.,156.,168.]]
    if smoke:grid=[(np.sqrt(60*120),144.)]
    for l,u in grid:
        print(f'BOUNDARY L={l:.5f}, U={u:g}',flush=True)
        labels=regional_targets(l,u)
        target=out/f'L{l:.5f}_U{u:g}';save_csv(labels.reset_index(drop=True),target/'regional_targets.csv')
        summary=run_cv(target,labels=labels,reselect=True,jobs=jobs,n_permutations=n_permutations,smoke=smoke)
        if not smoke:
            s=summary['solvent'];rows.append(dict(lower_mV_dec=l,upper_mV_dec=u,**s))
            save_csv(pd.DataFrame(rows),out/'boundary_ranks.csv')
    return {'settings_completed':len(rows),'smoke_only':smoke}
