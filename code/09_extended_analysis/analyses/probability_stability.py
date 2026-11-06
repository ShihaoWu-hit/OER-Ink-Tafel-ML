"""Repeat-level reliance, group-deletion refits, and group bootstrap."""
from .common import *
from .model_pipeline import run as run_cv

def run(out,jobs=3,n_permutations=100,groups=None,baseline=True,smoke=False):
    out=Path(out)
    if baseline:run_cv(out/'full_data',reselect=False,n_permutations=n_permutations,jobs=jobs,bootstrap=2000,smoke=smoke)
    data=load_inputs();high_ids=data['labels'].index[data['labels'].q_high_off.ge(.5)]
    gids=sorted(set(data['records'].loc[high_ids,'cv_group_id']));rows=[]
    for i,gid in enumerate(gids,1):
        name=f'G{i}'
        if groups is not None and name not in groups:continue
        print('FULL GROUP DELETION',name,gid,flush=True)
        result=run_cv(out/'deletions'/name,drop_group=gid,reselect=True,n_permutations=n_permutations,jobs=jobs,smoke=smoke)
        if not smoke:rows.append({'group_label':name,'cv_group_id':gid,'n_records':result['n_records'],**result['solvent']})
        save_csv(pd.DataFrame(rows),out/'deletion_summary.csv')
    return rows
