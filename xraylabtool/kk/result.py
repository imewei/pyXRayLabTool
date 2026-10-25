"""Result container of the Kramers-Kronig calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class KKResult:
    """Real and imaginary scattering factors of a formula unit.

    Attributes
    ----------
    formula : str
        Chemical formula as given.
    energy_kev : numpy.ndarray, shape (m,)
        Photon energies in keV, in the order they were calculated.
    f1 : numpy.ndarray, shape (m,)
        Real part of the scattering factor, electrons per formula unit.
    f2 : numpy.ndarray, shape (m,)
        Imaginary part of the scattering factor, electrons per formula unit.
    relativistic_correction : float
        Relativistic correction Z* (electrons) added to the transform.
    merge_points_kev : tuple of float or None
        Energies (keV) at which near-edge data was merged into the tabulated
        spectrum, or ``None`` when the tabulated spectrum alone was used.
    n_refined_points : int
        Number of points added by adaptive refinement (0 if not refined).
    backend : str
        Array backend that ran the transform, ``"numpy"`` or ``"jax"``.
    nearedge_scaled_kev_f2 : tuple of numpy.ndarray or None
        ``(energy_kev, f2)`` of the near-edge data on the scattering-factor
        scale, on its own energy grid; ``None`` without near-edge data.

    References
    ----------
    Upstream-only/package addition: a container for the output of the
    transform of B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (9).

    Examples
    --------
    >>> from xraylabtool.kk import kk_calculate_real
    >>> result = kk_calculate_real("Si", energies_kev=[1.0, 2.0])
    >>> result.f1.shape == result.f2.shape == result.energy_kev.shape
    True
    >>> result.n_refined_points
    0
    """

    formula: str
    energy_kev: NDArray[np.float64]
    f1: NDArray[np.float64]
    f2: NDArray[np.float64]
    relativistic_correction: float
    merge_points_kev: tuple[float, float] | None
    n_refined_points: int
    backend: str
    nearedge_scaled_kev_f2: tuple[NDArray[np.float64], NDArray[np.float64]] | None = (
        None
    )
