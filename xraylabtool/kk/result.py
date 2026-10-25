"""Result container of the Kramers-Kronig calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Imported at runtime (not under TYPE_CHECKING) so that
# ``typing.get_type_hints(KKResult)`` resolves the string annotations.
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from xraylabtool.calculators.xray_result import XRayResult


@dataclass(frozen=True, eq=False)
class KKResult:
    """Real and imaginary scattering factors of a formula unit.

    The dataclass is frozen and compares by identity (``eq=False``): the
    array fields have no element-wise-safe ``==`` or ``hash``.

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

    def to_xray_result(self, density: float) -> XRayResult:
        """Convert to an :class:`~xraylabtool.calculators.xray_result.XRayResult`.

        Computes δ, β, the critical angle, attenuation length and SLD from
        ``f1`` and ``f2`` of this result, with the same code as
        ``calculate_single_material_properties``. Negative ``f1`` near an
        absorption edge is kept (critical angle 0 there); the energy order is
        preserved.

        Parameters
        ----------
        density : float
            Mass density in g/cm^3; finite and positive.

        Returns
        -------
        XRayResult
            Optical properties at ``energy_kev``, arrays of shape ``(m,)``.

        Raises
        ------
        EnergyError
            If any energy lies outside the tabulated 0.03-30 keV range of the
            package. This is the only place the range is enforced; the
            transform itself accepts 10 eV-500 keV.
        ValidationError
            If ``density`` is not finite and positive.

        References
        ----------
        Package addition: the adapter to the repository's result type; the
        transform it carries is B. Watts, Opt. Express 22(19), 23628 (2014),
        Eq. (9).

        Examples
        --------
        >>> from xraylabtool.kk import kk_calculate_real
        >>> kk = kk_calculate_real("Si", energies_kev=[2.0, 8.0])
        >>> xr = kk.to_xray_result(2.33)
        >>> xr.energy_kev.shape
        (2,)
        >>> bool((xr.scattering_factor_f1 == kk.f1).all())
        True
        """
        from xraylabtool.calculators.core import (  # noqa: PLC0415
            calculate_properties_from_factors,
        )
        from xraylabtool.exceptions import EnergyError  # noqa: PLC0415

        low, high = 0.03, 30.0
        if self.energy_kev.size and not (
            np.all(self.energy_kev >= low) and np.all(self.energy_kev <= high)
        ):
            raise EnergyError(
                "Energies must lie in the tabulated range to build an XRayResult; "
                f"got {self.energy_kev.min():g}-{self.energy_kev.max():g} keV",
                valid_range="0.03-30 keV",
            )
        return calculate_properties_from_factors(
            self.formula, self.energy_kev, density, self.f1, self.f2
        )
