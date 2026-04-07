# CI/CD Security Audit Report

**Date:** 2026-04-07  
**Scope:** `.github/workflows/`, `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`

---

## CICD-001

- **Severity:** High
- **Category:** CI / Permissions
- **Location:** `.github/workflows/ci.yml`, `.github/workflows/docs.yml`, `.github/workflows/dependencies.yml`
- **Description:** All GitHub Actions are pinned by version tag (e.g. `actions/checkout@v5`) rather than immutable SHA digest. A tag can be silently moved by a supply-chain attacker (tag mutation attack). This is the primary vector for GitHub Actions supply-chain compromise.
- **Current state:** All `uses:` lines use semver tags (`@v4`, `@v5`, `@v6`, `@v7`, `@release/v1`). No SHA pins anywhere.
- **Recommendation:** Pin every external action to a full commit SHA. Example:
  ```yaml
  uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
  ```
  Use a tool like `pinact` or Dependabot's `github-actions` ecosystem to manage upgrades.
- **Effort:** Medium

---

## CICD-002

- **Severity:** High
- **Category:** CI
- **Location:** `.github/workflows/ci.yml` (lines 183–184, 261–264), `.github/workflows/release.yml` (line 447)
- **Description:** `uv` is installed by piping a remote shell script directly into `sh` (`curl -LsSf https://astral.sh/uv/install.sh | sh`). This is a classic supply-chain risk: a compromised or MITM'd script runs with full shell privileges. There is no checksum verification.
- **Current state:** Three jobs (`lint`, `test`, `build`) each re-download and execute the install script without integrity verification.
- **Recommendation:** Either (a) use the official `astral-sh/setup-uv` action (pinned by SHA), or (b) download the binary, verify its SHA-256 against a pinned checksum, then install:
  ```yaml
  - uses: astral-sh/setup-uv@f0ec1fc3b38f5e7cd731bb6ce540c5af426f5a27  # v5
  ```
- **Effort:** Small

---

## CICD-003

- **Severity:** High
- **Category:** Permissions
- **Location:** `.github/workflows/release.yml` (lines 38–40)
- **Description:** The release workflow requests `contents: write` and `id-token: write` at the workflow level. `id-token: write` is a sensitive permission (OIDC token for cloud auth). It should be scoped only to the `publish-pypi` job via job-level permissions, not granted to all jobs.
- **Current state:** Both permissions are declared at the top-level `permissions:` block, which grants them to all jobs including `validate-version`, `update-version`, and `post-release`.
- **Recommendation:** Move `id-token: write` to the `publish-pypi` job only. Set top-level to `permissions: contents: read` and override per job.
- **Effort:** Small

---

## CICD-004

- **Severity:** High
- **Category:** Secrets
- **Location:** `.github/workflows/release.yml` (lines 152–155)
- **Description:** The release workflow uses a default `GITHUB_TOKEN` to push directly to `main` and create tags (`git push origin main`). This bypasses any branch protection rules (required reviews, status checks) configured on `main`. It also sets `git config user.email "action@github.com"` in plain text.
- **Current state:** `git push origin main` and `git push origin <tag>` are executed unconditionally after version bumping.
- **Recommendation:** (a) Never push directly to `main` from CI — create a PR instead. (b) If direct push is required, use a dedicated deploy key or PAT with minimal scope, stored as a repository secret, not `GITHUB_TOKEN`. (c) Enable branch protection with "Restrict pushes that create matching refs."
- **Effort:** Medium

---

## CICD-005

- **Severity:** Medium
- **Category:** SAST
- **Location:** `.github/workflows/ci.yml`, pyproject.toml
- **Description:** Ruff `S` (Bandit/flake8-bandit) rules ARE enabled in `pyproject.toml`, which is good. However, `ruff check` in CI is set to `continue-on-error: true` (line 207), meaning security rule violations are non-blocking and will never fail the pipeline. A security finding is surfaced only as a warning.
- **Current state:** `ruff check ... || echo "⚠️ Linting warnings detected (non-blocking)"`. No CodeQL or Semgrep workflow.
- **Recommendation:** Remove `continue-on-error: true` from the ruff step, or at minimum split linting into two steps — one for format/style (non-blocking) and one for security rules (blocking). Consider adding a CodeQL workflow (free for public repos) or Semgrep via `semgrep/semgrep-action`.
- **Effort:** Small (making ruff blocking) / Medium (adding CodeQL)

---

## CICD-006

- **Severity:** Medium
- **Category:** Dependencies
- **Location:** `.github/` (missing `dependabot.yml`)
- **Description:** No Dependabot configuration exists. The `dependencies.yml` workflow provides manual/scheduled audit using `safety` and `pip-audit`, but both security checks are also `continue-on-error: true` — they produce reports but never block merges.
- **Current state:** `safety check ... || true` and `pip-audit ... || true`. No `.github/dependabot.yml` for automated PRs.
- **Recommendation:** (1) Add `.github/dependabot.yml` with `pip` and `github-actions` ecosystems. (2) Remove `|| true` from `safety` and `pip-audit` — let them fail the audit job on critical CVEs.
- **Effort:** Small

---

## CICD-007

- **Severity:** Medium
- **Category:** Secrets
- **Location:** `.pre-commit-config.yaml`
- **Description:** Pre-commit is configured but has no secret detection hook. Credentials, API keys, or tokens committed to the repo would not be caught before push.
- **Current state:** Hooks: `pre-commit-hooks` (file hygiene), `ruff` (lint/format). No TruffleHog, `detect-secrets`, or `gitleaks`.
- **Recommendation:** Add a secret-scanning hook, e.g.:
  ```yaml
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
  ```
  Also add the Gitleaks action to CI as a blocking check on PRs.
- **Effort:** Small

---

## CICD-008

- **Severity:** Medium
- **Category:** CI
- **Location:** `.github/workflows/release.yml` (line 88)
- **Description:** `pre-release-tests` calls the CI workflow with `secrets: inherit`. This passes ALL repository secrets to the reusable workflow, including any future secrets added. Least-privilege would pass only the specific secrets needed.
- **Current state:** `secrets: inherit` on the `uses: ./.github/workflows/ci.yml` call.
- **Recommendation:** Replace with explicit secret passing: `secrets: MY_SPECIFIC_SECRET: ${{ secrets.MY_SPECIFIC_SECRET }}`. If CI currently needs no secrets, pass nothing.
- **Effort:** Small

---

## CICD-009

- **Severity:** Medium
- **Category:** Permissions
- **Location:** `.github/workflows/release.yml` (line 270), `.github/workflows/release.yml`
- **Description:** `softprops/action-gh-release@v1` is unpinned by SHA and is a third-party action granted `contents: write`. Third-party actions with write permissions and unpinned versions are a high-risk combination.
- **Current state:** `uses: softprops/action-gh-release@v1` — no SHA pin.
- **Recommendation:** Pin to SHA and consider replacing with the official `actions/create-release` or `gh` CLI, which avoids the third-party dependency.
- **Effort:** Small

---

## CICD-010

- **Severity:** Low
- **Category:** CI
- **Location:** `.github/workflows/ci.yml` (line 63–66)
- **Description:** Tool versions are pinned as env vars (`RUFF_VERSION: "0.14.9"`) and used in install commands. However, `UV_VERSION: "0.5.0"` is declared but never used — uv is installed via the unversioned install script (CICD-002), so the pinned version has no effect.
- **Current state:** `UV_VERSION` env var is set but the install script ignores it.
- **Recommendation:** Either pass `UV_VERSION` to the install script (`--version $UV_VERSION`) or use the `astral-sh/setup-uv` action which accepts a `version:` input.
- **Effort:** Small

---

## CICD-011

- **Severity:** Low
- **Category:** Automation
- **Location:** Not present
- **Description:** No license compliance check. The project installs `jax`, `PySide6`, and scientific libraries with diverse licenses (Apache-2.0, LGPL, MIT). There is no automated check that new dependencies do not introduce incompatible licenses (e.g., GPL in a library distributed under a permissive license).
- **Current state:** No license scanning in any workflow.
- **Recommendation:** Add `pip-licenses` or `liccheck` to the `dependency-audit` job and define an allowed-licenses policy in `pyproject.toml` or a `liccheck.ini`.
- **Effort:** Small

---

## Summary Table

| ID       | Severity | Category    | Effort |
|----------|----------|-------------|--------|
| CICD-001 | High     | CI/Permissions | Medium |
| CICD-002 | High     | CI          | Small  |
| CICD-003 | High     | Permissions | Small  |
| CICD-004 | High     | Secrets/CI  | Medium |
| CICD-005 | Medium   | SAST        | Small  |
| CICD-006 | Medium   | Dependencies| Small  |
| CICD-007 | Medium   | Secrets     | Small  |
| CICD-008 | Medium   | Permissions | Small  |
| CICD-009 | Medium   | Permissions | Small  |
| CICD-010 | Low      | CI          | Small  |
| CICD-011 | Low      | Automation  | Small  |

### What exists and works well
- Global `permissions: contents: read` in CI and docs workflows (principle of least privilege at workflow level).
- Ruff `S` (Bandit) ruleset enabled in `pyproject.toml` — security rules are active locally.
- `concurrency` groups with `cancel-in-progress` prevent queue buildup.
- `dependency-audit` job with `safety` and `pip-audit` tooling (needs to block, not warn).
- Trusted publishing (`id-token: write` + `pypa/gh-action-pypi-publish`) for PyPI — avoids storing PyPI tokens as secrets.
- Pre-commit hooks for basic file hygiene and ruff.
