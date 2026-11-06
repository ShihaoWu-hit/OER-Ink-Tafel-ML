"""Repository-relative inputs shared by the extended analyses."""
from pathlib import Path
import os, sys
ROOT = Path(os.environ.get("TAFEL_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
CORE = ROOT / "code/Ink_Tafel_Study/src"
sys.path.insert(0, str(CORE))
MODEL = ROOT / "Data/04_results/RESULTS/04_MODEL/PRESTO_policy_v2_complete"
REFIT = ROOT / "Data/03_outputs/tafel_refitting/outputs/20260723_bayesian_local_line_formal_target_v2"
INPUT_FILES = {
 "records.csv": MODEL / "policy_v2/inputs/records.csv",
 "soft_labels.csv": MODEL / "policy_v2/registry/contiguous_three_class_soft_labels.csv",
 "outer_split_manifest.csv": MODEL / "policy_v2/folds/outer_split_manifest.csv",
 "inner_split_manifest.csv": MODEL / "policy_v2/folds/inner_split_manifest.csv",
 "probability_selections.csv": MODEL / "probability_engine/analyses/outer_selections.csv",
 "policy_selections.csv": MODEL / "policy_v2/analyses/outer_selections.csv",
 "oof_probabilities.csv": MODEL / "probability_engine/analyses/oof_probabilities.csv",
 "policy_oof_decisions.csv": MODEL / "policy_v2/analyses/policy_oof_decisions.csv",
 "parent_oof_probabilities.csv": MODEL / "policy_v2/inputs/parent_oof_probabilities.csv",
 "parent_policy_record_aggregated.csv": MODEL / "policy_v2/analyses/policy_record_aggregated.csv",
 "posterior_target_draws.csv.gz": REFIT / "posterior_target_draws.csv.gz",
 "window_selection_replicates.csv.gz": REFIT / "window_selection_replicates.csv.gz",
}
def input_path(name):
    return INPUT_FILES[name]
