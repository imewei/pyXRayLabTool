"""Result container of the Kramers-Kronig calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

# Imported at runtime (not under TYPE_CHECKING) so that
# ``typing.get_type_hints(KKResult)`` resolves the string annotations.
import numpy as np
from numpy.typing import ArrayLike, NDArray

from xraylabtool.constants import TABULATED_MAX_KEV, TABULATED_MIN_KEV
from xraylabtool.exceptions import EnergyError, ValidationError

if TYPE_CHECKING:
    from xraylabtool.calculators.xray_result import XRayResult


def _readonly_1d(
    name: str, values: ArrayLike, *, finite: bool = True
) -> NDArray[np.float64]:
    """Return a read-only float64 copy of a 1-D array, or raise ``ValidationError``."""
    if np.iscomplexobj(values):
        raise ValidationError(f"{name} must be real numbers, got complex", name)
    arr = np.array(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be a 1-D array", name, arr.shape)
    if finite and not np.isfinite(arr).all():
        raise ValidationError(f"{name} must be finite", name)
    arr.setflags(write=False)
    return arr


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
        Stored as a tuple of floats.
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
    transform of B. Watts, Opt. Express 22(19), 23628 (2014), Eqs. (7)-(10)
    (Z* from Eq. (9)).

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
    backend: Literal["numpy", "jax"]
    nearedge_scaled_kev_f2: tuple[NDArray[np.float64], NDArray[np.float64]] | None = (
        None
    )

    def __post_init__(self) -> None:
        energy = _readonly_1d("energy_kev", self.energy_kev)
        f1 = _readonly_1d("f1", self.f1)
        f2 = _readonly_1d("f2", self.f2)
        if not energy.size == f1.size == f2.size:
            raise ValidationError(
                "energy_kev, f1 and f2 must have the same length",
                "f1",
                (energy.size, f1.size, f2.size),
            )
        if self.n_refined_points < 0:
            raise ValidationError(
                "n_refined_points must be >= 0",
                "n_refined_points",
                self.n_refined_points,
            )
        if self.backend not in ("numpy", "jax"):
            raise ValidationError(
                "backend must be 'numpy' or 'jax'", "backend", self.backend
            )
        merge = self.merge_points_kev
        if merge is not None and not (
            len(merge) == 2 and np.isfinite(merge).all() and merge[0] < merge[1]
        ):
            raise ValidationError(
                "merge_points_kev must be a finite (low, high) pair with low < high",
                "merge_points_kev",
                merge,
            )
        scaled = self.nearedge_scaled_kev_f2
        if scaled is not None:
            scaled_e = _readonly_1d("nearedge_scaled_kev_f2[0]", scaled[0])
            scaled_f2 = _readonly_1d("nearedge_scaled_kev_f2[1]", scaled[1])
            if scaled_e.size != scaled_f2.size:
                raise ValidationError(
                    "nearedge_scaled_kev_f2 arrays must have the same length",
                    "nearedge_scaled_kev_f2",
                    (scaled_e.size, scaled_f2.size),
                )
            object.__setattr__(self, "nearedge_scaled_kev_f2", (scaled_e, scaled_f2))
        if merge is not None:
            object.__setattr__(
                self, "merge_points_kev", (float(merge[0]), float(merge[1]))
            )
        object.__setattr__(self, "energy_kev", energy)
        object.__setattr__(self, "f1", f1)
        object.__setattr__(self, "f2", f2)

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
        Eqs. (7)-(10) (Z* from Eq. (9)).

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

        low, high = TABULATED_MIN_KEV, TABULATED_MAX_KEV
        if self.energy_kev.size and not (
            np.all(self.energy_kev >= low) and np.all(self.energy_kev <= high)
        ):
            raise EnergyError(
                "Energies must lie in the tabulated range to build an XRayResult; "
                f"got {self.energy_kev.min():g}-{self.energy_kev.max():g} keV",
                valid_range=f"{low:g}-{high:g} keV",
            )
        return calculate_properties_from_factors(
            self.formula, self.energy_kev, density, self.f1, self.f2
        )
