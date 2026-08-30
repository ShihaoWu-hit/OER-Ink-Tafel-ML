# Formal Bayesian local-line target evidence

This is the formal evidence source for `b_bayesian_local_line_mV_dec` under policy `bayesian_local_line_eta015_strict_then_pava_fallback_v2`.
All 97 canonical records have 8,000 pooled posterior draws. Ninety-five use
the strict observed-progression selector. Records 23 and 74 use the registered
quantization-aware PAVA fallback after the strict selector finds no supported
window.

The robust Bayesian local-linear model propagates parameter, digitisation, and
window-selection uncertainty through an equal-weight mixture over 16
coordinate-perturbed selections. The fallback keeps the same local-line
estimand and does not substitute a selected-model or Equation-2 target. The
method does not use slope magnitude, literature slopes, or ML performance to
choose a window. Formal ML remains blocked until all 97 PPCs receive a new
named human review.
