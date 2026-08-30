from __future__ import annotations

import numpy as np
import pytest

from tafel_refit.bayes import oer_overpotential_V, render_posterior_predictive_plot


def test_oer_overpotential_uses_absolute_rhe_potential() -> None:
    potential_rhe = np.array([1.23, 1.40, 1.65])

    assert oer_overpotential_V(potential_rhe) == pytest.approx([0.0, 0.17, 0.42])


def test_ppc_render_reports_overpotential_limits_without_changing_y(tmp_path) -> None:
    potential_rhe = np.array([1.30, 1.40, 1.50])
    observed = np.array([-2.0, -1.5, -1.0])
    draws = np.vstack([observed - 0.05, observed, observed + 0.05])
    output = tmp_path / "ppc.png"

    diagnostics = render_posterior_predictive_plot(
        curve_id="sr_test",
        model_name="linear",
        potential_rhe_V=potential_rhe,
        observed_log10_current=observed,
        posterior_predictive_draws=draws,
        output_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert diagnostics["overpotential_min_V"] == pytest.approx(0.07)
    assert diagnostics["overpotential_max_V"] == pytest.approx(0.27)
    assert diagnostics["posterior_predictive_rmse_log10"] == pytest.approx(0.0)
