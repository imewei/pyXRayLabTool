# Security & Reliability Audit — xraylabtool/

**Date:** 2026-04-07
**Auditor:** security-reviewer agent
**Scope:** xraylabtool/ Python package (desktop scientific application, PySide6 GUI)

---

## Summary

| Severity  | Count |
|-----------|-------|
| High      | 2     |
| Medium    | 4     |
| Low       | 3     |

No hardcoded secrets, no pickle/shelve deserialization, no QWebEngine/JS injection surfaces, no wildcard imports, no eval/exec found. The main risks are subprocess path-injection via environment variables, unsafe filename construction from user-controlled formula strings, and the completion system modifying shell activation scripts without integrity checks.

---

## Findings

---

### SEC-001 — High
**File:** xraylabtool/interfaces/completion_v2/environment.py:293
**Description:** _get_python_version() constructs a subprocess call using the result of _find_python_executable(env_path), where env_path is derived from the VIRTUAL_ENV or CONDA_PREFIX environment variable without any canonicalization or bounds-checking. An adversary who can set VIRTUAL_ENV to a path containing a malicious bin/python binary can execute arbitrary code when the completion system probes the Python version.
**Impact:** Local privilege escalation / arbitrary code execution if an attacker controls the VIRTUAL_ENV environment variable (e.g., via a compromised shell config or shared workstation).
**Remediation:** Resolve the python executable path with Path.resolve() and verify it is inside the claimed virtual-environment root before invoking it. Use check=False and validate returncode instead of relying on exception handling only.
**Effort:** Small

---

### SEC-002 — High
**File:** xraylabtool/gui/main_window.py:1059,1102
**Description:** Export filenames are built directly from self.single_result.formula, which comes from user-typed GUI input. The final file path is constructed as Path(folder) / fname. If formula contains path-separator characters (e.g., ../), this could produce path traversal. The chemical formula validator rejects characters outside [A-Za-z0-9().]+, but this check is only applied at calculation time, not at export time.
**Impact:** Potential file overwrite at arbitrary paths if formula is unchecked at export time.
**Remediation:** Sanitize formula before embedding in filenames: safe = re.sub(r'[^\w\-.]', '_', formula). Apply sanitization at export time regardless of prior validation.
**Effort:** Small

---

### SEC-003 — Medium
**File:** xraylabtool/interfaces/completion_v2/installer.py:360-391
**Description:** The conda/venv activation hook installer writes shell scripts that source a completion script at a user-controlled path. The path string is embedded with f-strings directly into shell heredocs: source "{script_path}". If script_path contains a double-quote or shell-metacharacter, the activation script becomes a shell-injection vector executed every time the environment is activated.
**Impact:** Persistent shell injection triggered on every shell login / environment activation.
**Remediation:** Use single-quoted path references in shell scripts: source '${script_path}' (single-quoted strings do not expand variables or metacharacters in bash/zsh). Alternatively, use shlex.quote(str(script_path)) and embed the result.
**Effort:** Small

---

### SEC-004 — Medium
**File:** xraylabtool/interfaces/completion_v2/installer.py:437-468 (_modify_activation_script)
**Description:** When uninstalling, _remove_venv_hooks() rewrites venv activation scripts using heuristic string matching ("fi" in line or "end" in line). This logic can silently corrupt the activation script if the completion block is nested. _modify_activation_script also appends to user-owned activation scripts without taking a backup.
**Impact:** Corrupted virtual environment activation script on uninstall.
**Remediation:** Always backup the original file before modification (activate.bak). Use unique sentinel markers (# XRAYLABTOOL_COMPLETION_BEGIN / END) and remove only the delimited block.
**Effort:** Medium

---

### SEC-005 — Medium
**File:** xraylabtool/interfaces/completion_v2/environment.py:549-563 (_load_cache / _save_cache)
**Description:** The environment discovery cache at ~/.xraylabtool/env_cache.json is loaded with json.load() without schema validation. The deserialized path values are passed directly into Path() and used in file-system operations and subprocess calls. A tampered or symlink-replaced cache file can redirect these operations to arbitrary paths.
**Impact:** If the cache file is writable by another local user, it can be used for path traversal or subprocess redirection.
**Remediation:** After loading, validate each path value with Path(data["path"]).resolve() and confirm it falls under known safe prefixes. Set cache file permissions to 0o600 at creation time.
**Effort:** Small

---

### SEC-006 — Medium
**File:** xraylabtool/interfaces/completion_v2/environment.py:270,293,400,433,500,509
**Description:** Multiple subprocess.run() calls for conda/poetry/python discovery lack a timeout parameter (unlike the _is_mamba_environment call at line 193 which correctly sets timeout=5). These calls can block indefinitely if the invoked tool hangs.
**Impact:** GUI or CLI hangs indefinitely requiring a kill signal.
**Remediation:** Add timeout=10 to all subprocess.run() calls in environment.py that are missing it.
**Effort:** Small

---

### SEC-007 — Low
**File:** xraylabtool/interfaces/completion_v2/cache.py:36
**Description:** MD5 is used as a cache key hash (with usedforsecurity=False). While acceptable for non-cryptographic key generation, if the cache key is ever used in a security-relevant context (e.g., integrity verification), MD5 is insufficient.
**Impact:** Low — advisory only for current non-security use.
**Remediation:** Switch to hashlib.sha256 for consistency and future-proofing.
**Effort:** Small

---

### SEC-008 — Low
**File:** xraylabtool/device.py:18,41 and xraylabtool/interfaces/completion_v2/environment.py (multiple)
**Description:** subprocess.run() calls use bare command names (nvcc, nvidia-smi, conda, poetry, python3, python) without resolving them to absolute paths via shutil.which(). On shared HPC systems or when PATH is manipulated, a malicious binary with the same name earlier in PATH would be executed.
**Impact:** Very low risk on controlled developer workstations; moderate on shared systems.
**Remediation:** Resolve tool paths with shutil.which() before invoking them and pass the absolute path. Log a warning if the expected tool is not found.
**Effort:** Small

---

### SEC-009 — Low
**File:** xraylabtool/validation/validators.py:203-216 (_parse_formula)
**Description:** The chemical formula parser accepts parentheses in the pre-filter regex (^[A-Za-z0-9().]+$) but the inner parser regex ([A-Z][a-z]?)(\d*\.?\d*) ignores them entirely. Ca(OH)2 is silently parsed as Ca:1, O:1, H:1 instead of Ca:1, O:2, H:2. The comment in the source acknowledges this is a simplified parser.
**Impact:** Silent incorrect calculation results for compounds with parenthesized groups. Violates the project's "No silent data loss/truncation" priority.
**Remediation:** Either reject parenthesized formulas with an explicit error, or implement a proper recursive formula parser. Do not silently strip parentheses.
**Effort:** Medium

---

## Non-Findings (checked and clean)

- pickle / shelve / marshal / yaml.unsafe_load: None found.
- eval() / exec() with user input: None found.
- Hardcoded credentials / API keys / tokens: None found.
- subprocess.run(..., shell=True): Not used anywhere.
- QWebEngine / JavaScript injection: Not used. GUI is pure Qt widgets only.
- Unsafe temp file creation: No tempfile usage found.
- XML external entity injection: No XML parsing found.
- Wildcard imports: None found (compliant with project rules).

---

## Prioritized Remediation Order

1. SEC-001: Validate env path before subprocess call (High, Small effort)
2. SEC-003: Fix shell injection in activation script writes (High, Small effort)
3. SEC-002: Sanitize formula in export filenames at export time (High, Small effort)
4. SEC-006: Add subprocess timeouts throughout environment.py (Medium, Small effort)
5. SEC-005: Validate and restrict cache-sourced paths (Medium, Small effort)
6. SEC-009: Fix silent formula parsing error for parentheses (Low-Medium, Medium effort)
7. SEC-004: Backup activation scripts before modification (Medium, Medium effort)
8. SEC-007: Replace MD5 with SHA-256 in cache key (Low, Small effort)
9. SEC-008: Use absolute paths for subprocess tool invocations (Low, Small effort)
