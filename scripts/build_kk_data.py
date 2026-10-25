#!/usr/bin/env python
"""Build ``xraylabtool/data/kkcalc/asf.npz`` from upstream kkcalc's ``ASF.json``.

Upstream: https://github.com/benajamin/kkcalc (Zlib licence), pinned to one
commit. Only ``E``, ``Im``, ``mass`` and ``symbol`` are kept; the upstream
``Re`` tables are not used by the Kramers-Kronig transform.

The npz layout is ragged-free: per-element arrays are concatenated and indexed
by offsets.

``E`` : float64, all knot energies (eV), concatenated in Z order
``E_offsets`` : int64 (93,), element ``z`` owns ``E[E_offsets[z-1]:E_offsets[z]]``
``Im`` : float64 (sum(n_z), 5), per-segment coefficients of
    ``a1*E + a0 + a-1/E + a-2/E**2 + a-3/E**3``, with ``n_z = len(E_z) - 1``
``Im_offsets`` : int64 (93,)
``mass`` : float64 (92,)
``symbol`` : str (92,)
``upstream_commit``, ``asf_json_sha256`` : str

Usage: ``python scripts/build_kk_data.py [--src path/to/ASF.json]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

UPSTREAM_COMMIT = "f331ca31a654af946f7d550a22cf4e9c11fa8afc"
ASF_JSON_SHA256 = "9edf4fac0d8dbc0552938abb106039a71c34e9c7ea71969964a41b56eaf79f66"
ASF_URL = (
    "https://raw.githubusercontent.com/benajamin/kkcalc/"
    f"{UPSTREAM_COMMIT}/kkcalc/ASF.json"
)
OUT = (
    Path(__file__).resolve().parents[1] / "xraylabtool" / "data" / "kkcalc" / "asf.npz"
)


def build(raw: bytes, out: Path = OUT) -> None:
    """Verify ``raw`` against the pinned hash and write the npz to ``out``."""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ASF_JSON_SHA256:
        raise SystemExit(
            f"ASF.json sha256 mismatch: got {digest}, expected {ASF_JSON_SHA256}"
        )
    db = json.loads(raw)
    zs = range(1, 93)
    if sorted(int(k) for k in db) != list(zs):
        raise SystemExit("ASF.json does not hold exactly Z = 1..92")
    e_parts = [np.asarray(db[str(z)]["E"], dtype=np.float64) for z in zs]
    im_parts = [np.asarray(db[str(z)]["Im"], dtype=np.float64) for z in zs]
    for z, e, im in zip(zs, e_parts, im_parts, strict=True):
        if im.shape != (len(e) - 1, 5):
            raise SystemExit(
                f"Z={z}: Im shape {im.shape} inconsistent with {len(e)} knots"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        E=np.concatenate(e_parts),
        E_offsets=np.concatenate(([0], np.cumsum([len(e) for e in e_parts]))).astype(
            np.int64
        ),
        Im=np.concatenate(im_parts),
        Im_offsets=np.concatenate(([0], np.cumsum([len(i) for i in im_parts]))).astype(
            np.int64
        ),
        mass=np.array([db[str(z)]["mass"] for z in zs], dtype=np.float64),
        symbol=np.array([db[str(z)]["symbol"] for z in zs]),
        upstream_commit=np.array(UPSTREAM_COMMIT),
        asf_json_sha256=np.array(ASF_JSON_SHA256),
    )
    print(f"wrote {out} ({out.stat().st_size} bytes), sha256(ASF.json)={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src", type=Path, help="local ASF.json (default: download at pinned SHA)"
    )
    args = parser.parse_args()
    if args.src is not None:
        raw = args.src.read_bytes()
    else:
        with urllib.request.urlopen(ASF_URL, timeout=60) as resp:  # noqa: S310
            raw = resp.read()
    build(raw)


if __name__ == "__main__":
    main()
