import numpy as np
import pandas as pd

from tafel_refit.analysis import _scientific_boundary_fraction
from tafel_refit.models import (
    _bounds,
    latent_low_current_floor_bounds,
    latent_plateau_bounds,
)


def test_latent_plateau_and_floor_sensitivity_bounds_allow_noisy_extrema():
    x = np.linspace(0.0, 0.2, 20)
    y = np.linspace(-2.0, 1.0, 20)

    paper_bounds = _bounds("paper_single_plateau", x, y)
    floor_sensitivity_bounds = _bounds("single_background_plateau", x, y)

    assert paper_bounds[2][0] == np.max(y) - 1.0
    assert paper_bounds[2][1] == np.max(y) + 0.5
    assert floor_sensitivity_bounds[0][1] == np.min(y) + 1.0
    assert latent_plateau_bounds(y) == paper_bounds[2]
    assert latent_low_current_floor_bounds(y) == floor_sensitivity_bounds[0]


def test_scientific_boundary_fraction_rejects_noise_or_plateau_boundaries():
    draws = pd.DataFrame(
        {
            "success": [True, True, True, False],
            "boundary_hits": ["", "log_sigma_lower", "L_lower", ""],
        }
    )
    fraction = _scientific_boundary_fraction(draws, "paper_single_plateau")
    assert fraction == 2.0 / 3.0
