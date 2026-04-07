# xraylabtool — Unified Audit & Remediation Plan

**Date:** 2026-04-07
**Reviewers:** security (SRE), architecture (Architect), testing (Quality), secops (CI/CD)
**Scope:** xraylabtool/ (~20,000 LOC, 52 files) + tests/ + .github/workflows/

---

## Executive Summary

| Severity | Security | Architecture | Testing | CI/CD | **Total** |
|----------|----------|-------------|---------|-------|-----------|
| Critical | 0        | 1           | 3       | 0     | **4**     |
| High     | 2        | 3           | 7       | 4     | **16**    |
| Medium   | 4        | 6           | 5       | 5     | **20**    |
| Low      | 3        | 0*          | 2       | 2     | **7**     |

*\*2 positive findings (ARCH-011, ARCH-012) omitted from counts.*

**50 findings total.** After deduplication and cross-referencing, these reduce to **9 remediation work streams** organized into 4 priority tiers.

---

## Cross-Referenced Root Causes

Several findings from different reviewers point to the same underlying issue:

| Root Cause | Findings | Impact |
|-----------|----------|--------|
| **Formula parsing fragmentation** | ARCH-003, SEC-009, TEST-007 | 3 parsers with inconsistent behavior; parenthesized formulas silently produce wrong results |
| **`core.py` god module** | ARCH-001, ARCH-002, TEST-004 | 1400+ lines, bidirectional coupling, difficult to test components in isolation |
| **Completion/installer fragility** | SEC-001, SEC-003, SEC-004, SEC-005, SEC-006, TEST-006 | Shell injection, script corruption, subprocess hangs, zero test coverage |
| **CI gates are advisory-only** | CICD-005, CICD-006 | Security linting and dependency audits run but never block merges |
| **Dead code shipped** | ARCH-006, TEST-003 | optimization/ is deprecated, untested, and confusing |

---

## Prioritized Remediation Plan

### Tier 1 — Immediate (Week 1) — Correctness & Quick Security Wins

These are small-effort fixes that address correctness violations or close real attack surfaces.

| # | Work Item | Findings | Effort | Why Now |
|---|-----------|----------|--------|---------|
| 1.1 | **Consolidate formula parsing** — Unify into a single recursive-descent parser in `utils.py` that handles parentheses and decimal stoichiometry. All other call sites delegate. Add `FormulaError` for unparseable input. | ARCH-003, SEC-009, TEST-007 | **Small** | Silent incorrect calculations violate the project's #1 priority (Correctness) and #2 (No silent data loss). This is the highest-impact fix. |
| 1.2 | **Sanitize export filenames** — Apply `re.sub(r'[^\w\-.]', '_', formula)` before constructing any filesystem path from formula strings in `main_window.py`. | SEC-002 | **Small** | Path traversal on file export. |
| 1.3 | **Fix tautological test** — Replace `assert True` in `test_force_gc_with_cache_clearing` with a meaningful assertion (e.g., verify cache size decreases). | TEST-013 | **Small** | 5-minute fix; currently masks real bugs. |
| 1.4 | **Make CI security gates blocking** — Remove `continue-on-error: true` from ruff security step; remove `|| true` from `safety`/`pip-audit` in dependencies.yml. | CICD-005, CICD-006 | **Small** | Security findings currently produce reports nobody reads. |
| 1.5 | **Guard performance tests** — Add `@pytest.mark.performance` and exclude from default `pytest` runs. Add CI environment detection. | TEST-010 | **Small** | Spurious CI failures erode trust in the test suite. |
| 1.6 | **Delete `optimization/` dead code** — Remove the entire package. Move `bottleneck_analyzer.py` to `dev_tools/` if profiling is still needed. | ARCH-006, TEST-003 | **Small** | 800 lines of confusion removed. |

**Tier 1 total effort: ~2-3 developer days**

---

### Tier 2 — Short-term (Weeks 2-3) — Security Hardening & CI

| # | Work Item | Findings | Effort | Why Now |
|---|-----------|----------|--------|---------|
| 2.1 | **Harden completion/installer module** | SEC-001, SEC-003, SEC-004, SEC-005, SEC-006 | **Medium** | Multiple injection and corruption vectors in shell-facing code. |
| | — Canonicalize VIRTUAL_ENV/CONDA_PREFIX paths with `.resolve()` + subpath assertion (SEC-001) | | | |
| | — Use `shlex.quote()` for all paths in generated shell scripts (SEC-003) | | | |
| | — Add sentinel comment markers for activate script injection; write `.bak` before modify (SEC-004) | | | |
| | — Validate env_cache.json schema after load; set 0o600 permissions (SEC-005) | | | |
| | — Add `timeout=10` to all subprocess.run() calls (SEC-006) | | | |
| 2.2 | **Pin GitHub Actions by SHA** — Replace all `@v4`/`@v1` tags with full commit SHA in all 4 workflow files. Use `astral-sh/setup-uv` action instead of `curl | sh`. | CICD-001, CICD-002, CICD-009, CICD-010 | **Small** | Supply-chain hardening. |
| 2.3 | **Scope CI permissions** — Move `id-token: write` to job level in release.yml. Remove `secrets: inherit` from `pre-release-tests`. | CICD-003, CICD-008 | **Small** | Least-privilege principle. |
| 2.4 | **Add Dependabot + secret detection** — Create `.github/dependabot.yml`. Add gitleaks pre-commit hook. | CICD-006, CICD-007 | **Small** | Automated dependency and secret scanning. |
| 2.5 | **Add backend dispatch tests** — Unit tests for `NumpyBackend` and `JaxBackend` covering array creation, interpolation, dtype handling, and backend switching. | TEST-002 | **Medium** | Highest-risk surface for JAX migration. |
| 2.6 | **Add compound_analysis tests** — Test compound-to-element decomposition with edge cases: nested parentheses, hydrates, decimal stoichiometry. | TEST-001 | **Medium** | Foundational correctness module with zero tests. |

**Tier 2 total effort: ~5-7 developer days**

---

### Tier 3 — Medium-term (Weeks 4-6) — Architecture & Coverage

| # | Work Item | Findings | Effort | Why Now |
|---|-----------|----------|--------|---------|
| 3.1 | **Decompose `calculators/core.py`** — Extract into: `xray_result.py`, `scattering_data.py`, `cache.py`, `kernels.py`. Keep `core.py` as orchestration. Introduce `InterpolatorRegistry` protocol to break bidirectional coupling with `atomic_cache.py`. | ARCH-001, ARCH-002 | **Large** | Central maintainability improvement. Do after formula parsing is consolidated (1.1) so the new modules have clean dependencies. |
| 3.2 | **Refactor MainWindow** — Extract `TableFormatter`, `SingleTabController`, `MultiTabController`. Move `_OverlayScrollbarMarginHelper` to standalone utility. | ARCH-004, ARCH-005 | **Medium** | Unblocks future GUI development. |
| 3.3 | **Clean up XRayResult** — Remove or extract 14 deprecated property aliases. If backward compat is needed, use a `LegacyXRayResultMixin`. Update integration tests to use snake_case. | ARCH-009, TEST-015 | **Small** | v0.4.0 is a reasonable breaking-change point. |
| 3.4 | **Standardize exception types** — Audit all `raise ValueError` in core modules. Replace with domain exceptions (`FormulaError`, `EnergyError`, `ValidationError`). | ARCH-010 | **Small** | Enables reliable error handling at GUI/CLI boundaries. |
| 3.5 | **Remove NumpyProxy anti-pattern** — Replace `_NumpyProxy` in `analysis/__init__.py` with `import numpy as np`. | ARCH-007 | **Small** | IDE/autocomplete fix. |
| 3.6 | **Fix backend boundary violation** — Add `is_jax()` method to `_OpsProxy`/`ArrayBackend` protocol. Use it in `interpolation.py` instead of inspecting `_backend`. | ARCH-008 | **Small** | Clean abstraction boundary. |
| 3.7 | **Test CalculationWorker error path** — Remove `pragma: no cover` from `CalculationWorker.run`. Test exception-to-error signal propagation. | TEST-012 | **Small** | GUI could silently freeze on calculation errors. |
| 3.8 | **Add file_operations tests** — Test valid file loading, malformed files, encoding edge cases. | TEST-005, TEST-009 | **Medium** | I/O is a trust boundary. |

**Tier 3 total effort: ~8-12 developer days**

---

### Tier 4 — Long-term (Backlog) — Polish & Hardening

| # | Work Item | Findings | Effort |
|---|-----------|----------|--------|
| 4.1 | **Completion module test suite** — Integration tests for installer, shell hook generation, uninstall cleanup. | TEST-006 | Medium |
| 4.2 | **GUI widget tests** — Material form validation (negative density, empty formula, boundary values). | TEST-011 | Medium |
| 4.3 | **Batch processing partial-failure tests** — Mixed valid/invalid formulas in batch mode. | TEST-014 | Small |
| 4.4 | **Concurrent cache access tests** — Thread-safety of `lru_cache` functions used by GUI workers. | TEST-016 | Medium |
| 4.5 | **Resolve subprocess PATH security** — Use `shutil.which()` for all external tool invocations. | SEC-008 | Small |
| 4.6 | **Switch MD5 to SHA256 in cache keys** — Advisory; future-proofing only. | SEC-007 | Small |
| 4.7 | **License compliance check** — Add dependency license audit to CI. | CICD-011 | Small |
| 4.8 | **Fix `release.yml` main-push bypass** — Scope release commit to a protected service account or use a separate branch. | CICD-004 | Medium |
| 4.9 | **Restructure lint-as-tests** — Move `test_ci_cd_integration.py` and `test_code_quality.py` checks into CI pipeline steps. | TEST-018 | Small |
| 4.10 | **Device detection tests** — Basic coverage for `device.py`. | TEST-017 | Small |
| 4.11 | **Improve critical angle test** — Replace 4-point monotonicity check with physics-aware test that accounts for absorption edges. | TEST-008 | Medium |
| 4.12 | **Structure factor tests** — Multi-atom unit cells with non-trivial phase factors. | TEST-004 | Medium |

---

## Findings Cross-Reference Index

| Finding ID | Tier | Work Item | Severity |
|-----------|------|-----------|----------|
| ARCH-001 | 3.1 | Decompose core.py | Critical |
| ARCH-002 | 3.1 | Decompose core.py | High |
| ARCH-003 | 1.1 | Consolidate formula parsing | High |
| ARCH-004 | 3.2 | Refactor MainWindow | High |
| ARCH-005 | 3.2 | Refactor MainWindow | Medium |
| ARCH-006 | 1.6 | Delete optimization/ | Medium |
| ARCH-007 | 3.5 | Remove NumpyProxy | Medium |
| ARCH-008 | 3.6 | Fix backend boundary | Medium |
| ARCH-009 | 3.3 | Clean up XRayResult | Medium |
| ARCH-010 | 3.4 | Standardize exceptions | Medium |
| SEC-001 | 2.1 | Harden completion module | High |
| SEC-002 | 1.2 | Sanitize export filenames | High |
| SEC-003 | 2.1 | Harden completion module | Medium |
| SEC-004 | 2.1 | Harden completion module | Medium |
| SEC-005 | 2.1 | Harden completion module | Medium |
| SEC-006 | 2.1 | Harden completion module | Medium |
| SEC-007 | 4.6 | MD5 to SHA256 | Low |
| SEC-008 | 4.5 | PATH security | Low |
| SEC-009 | 1.1 | Consolidate formula parsing | Low* |
| TEST-001 | 2.6 | compound_analysis tests | Critical |
| TEST-002 | 2.5 | Backend dispatch tests | Critical |
| TEST-003 | 1.6 | Delete optimization/ | Critical |
| TEST-004 | 4.12 | Structure factor tests | High |
| TEST-005 | 3.8 | file_operations tests | High |
| TEST-006 | 4.1 | Completion module tests | High |
| TEST-007 | 1.1 | Consolidate formula parsing | High |
| TEST-008 | 4.11 | Critical angle test | High |
| TEST-009 | 3.8 | file_operations tests | High |
| TEST-010 | 1.5 | Guard performance tests | High |
| TEST-011 | 4.2 | GUI widget tests | Medium |
| TEST-012 | 3.7 | CalculationWorker tests | Medium |
| TEST-013 | 1.3 | Fix tautological test | Medium |
| TEST-014 | 4.3 | Batch failure tests | Medium |
| TEST-015 | 3.3 | Clean up XRayResult | Medium |
| TEST-016 | 4.4 | Concurrent cache tests | Medium |
| TEST-017 | 4.10 | Device tests | Low |
| TEST-018 | 4.9 | Restructure lint-as-tests | Low |
| CICD-001 | 2.2 | Pin Actions by SHA | High |
| CICD-002 | 2.2 | Pin Actions by SHA | High |
| CICD-003 | 2.3 | Scope CI permissions | High |
| CICD-004 | 4.8 | Release push bypass | High |
| CICD-005 | 1.4 | Make gates blocking | Medium |
| CICD-006 | 1.4/2.4 | Dependabot + blocking | Medium |
| CICD-007 | 2.4 | Secret detection | Medium |
| CICD-008 | 2.3 | Scope permissions | Medium |
| CICD-009 | 2.2 | Pin Actions by SHA | Medium |
| CICD-010 | 2.2 | Pin Actions by SHA | Low |
| CICD-011 | 4.7 | License compliance | Low |

*\*SEC-009 rated Low by security reviewer as a data-integrity issue, but cross-referenced with ARCH-003/TEST-007 it is effectively **High** for a scientific computing package.*

---

## What's Already Good

- **Clean dependency direction** (ARCH-011): `gui -> services -> calculators -> backend/constants/utils` — no upward leaks.
- **Backend abstraction** (ARCH-012): `ArrayBackend` protocol + proxy pattern enables transparent NumPy/JAX switching.
- **Trusted PyPI publishing** via OIDC — no stored API tokens.
- **Ruff `S` (Bandit) rules** enabled — just needs to be made blocking.
- **`safety` + `pip-audit`** tooling present — just needs to block merges.
- **Workflow-level `permissions: contents: read`** — good default restrictiveness.
- **No unsafe deserialization, eval, exec, or wildcard imports** anywhere in the codebase.
