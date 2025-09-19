#!/usr/bin/env python3
"""
Architecture audit script to identify documentation that needs updating
to match the current modular codebase structure.

This script scans documentation for architectural references and import
patterns that need to be updated to reflect the modular design.
"""

import re
from pathlib import Path


class ArchitectureAuditor:
    """Audit documentation for architectural alignment with current codebase."""

    def __init__(self, docs_directory: Path):
        """Initialize auditor with documentation directory."""
        self.docs_dir = docs_directory
        self.violations = []

        # Current modular structure (as specified in CLAUDE.md)
        self.current_modules = {
            "calculators": "Core calculation engines",
            "data_handling": "Data management and processing",
            "interfaces": "User interfaces (CLI and completion)",
            "io": "Input/output operations",
            "validation": "Data validation and error handling",
        }

        # Deprecated/old patterns that should be updated
        self.deprecated_patterns = [
            r"xraylabtool\.core",  # Old core module reference
            r"from xraylabtool\.core import",  # Old import pattern
            r"import xraylabtool\.core",  # Old import pattern
            r"xraylabtool\.utils\.calculate",  # Old calculation location
            r"single_material_calculation",  # Old function name
        ]

        # Current recommended import patterns
        self.recommended_patterns = {
            "calculate_single_material_properties": (
                "from xraylabtool.calculators import calculate_single_material_properties"
            ),
            "calculate_xray_properties": (
                "from xraylabtool.calculators import calculate_xray_properties"
            ),
            "XRayResult": "from xraylabtool.calculators.core import XRayResult",
            "parse_formula": "from xraylabtool.utils import parse_formula",
            "energy_to_wavelength": (
                "from xraylabtool.utils import energy_to_wavelength"
            ),
        }

    def get_documentation_files(self) -> list[Path]:
        """Get all documentation files to audit."""
        doc_files = []
        if self.docs_dir.exists():
            # RST files
            doc_files.extend(self.docs_dir.glob("**/*.rst"))
            # Markdown files
            doc_files.extend(self.docs_dir.glob("**/*.md"))
            # Jupyter notebooks
            doc_files.extend(self.docs_dir.glob("**/*.ipynb"))
        return doc_files

    def audit_deprecated_imports(self) -> list[dict]:
        """Audit for deprecated import patterns."""
        violations = []
        doc_files = self.get_documentation_files()

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for pattern in self.deprecated_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(
                        {
                            "file": str(doc_file.relative_to(self.docs_dir.parent)),
                            "pattern": pattern,
                            "matches": matches,
                            "type": "deprecated_import",
                        }
                    )

        return violations

    def audit_module_coverage(self) -> list[dict]:
        """Audit for adequate coverage of modular architecture."""
        violations = []
        doc_files = self.get_documentation_files()

        # Track which files mention the modular structure
        architecture_coverage = {}

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue

            # Skip specialized docs that don't need architecture coverage
            if any(
                skip in str(doc_file)
                for skip in ["cleanup-system", "optimization", "physics"]
            ):
                continue

            # Count mentions of current modules
            module_mentions = {}
            for module, _description in self.current_modules.items():
                count = len(re.findall(r"\b" + re.escape(module) + r"\b", content))
                if count > 0:
                    module_mentions[module] = count

            if module_mentions:
                architecture_coverage[
                    str(doc_file.relative_to(self.docs_dir.parent))
                ] = module_mentions

        # Check key architectural documentation files
        key_arch_files = ["index.rst", "getting_started.rst", "api/index.rst"]

        for key_file in key_arch_files:
            file_path = str(Path("docs") / key_file)
            if file_path not in architecture_coverage:
                violations.append(
                    {
                        "file": file_path,
                        "issue": "Missing modular architecture references",
                        "type": "missing_architecture",
                    }
                )
            elif len(architecture_coverage[file_path]) < 3:
                violations.append(
                    {
                        "file": file_path,
                        "issue": (
                            f"Limited architecture coverage: only {len(architecture_coverage[file_path])} modules mentioned"
                        ),
                        "modules_mentioned": list(
                            architecture_coverage[file_path].keys()
                        ),
                        "type": "limited_architecture",
                    }
                )

        return violations

    def audit_code_examples(self) -> list[dict]:
        """Audit code examples for current best practices."""
        violations = []
        doc_files = self.get_documentation_files()

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            # Look for code examples
            code_blocks = re.findall(
                r".. code-block:: python\s*\n\n(.*?)\n\n", content, re.DOTALL
            )
            code_blocks.extend(
                re.findall(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
            )

            for block in code_blocks:
                # Check for old import patterns
                if re.search(r"import xraylabtool\s*$", block, re.MULTILINE):
                    # This is fine - main package import
                    continue

                # Check for deprecated specific imports
                deprecated_imports = [
                    r"from xraylabtool\.core import",
                    r"xraylabtool\.core\.",
                    r"from xraylabtool import core",
                ]

                for pattern in deprecated_imports:
                    if re.search(pattern, block):
                        violations.append(
                            {
                                "file": str(doc_file.relative_to(self.docs_dir.parent)),
                                "code_block": (
                                    block[:100] + "..." if len(block) > 100 else block
                                ),
                                "issue": f"Deprecated import pattern: {pattern}",
                                "type": "deprecated_code_example",
                            }
                        )

        return violations

    def audit_api_structure_refs(self) -> list[dict]:
        """Audit API documentation structure references."""
        violations = []

        # Check if API docs reflect current structure
        api_dir = self.docs_dir / "api"
        if not api_dir.exists():
            violations.append(
                {
                    "file": "docs/api/",
                    "issue": "API documentation directory missing",
                    "type": "missing_api_docs",
                }
            )
            return violations

        # Expected API documentation files based on current modules
        expected_api_files = [
            "calculators.rst",
            "data_handling.rst",
            "interfaces.rst",
            "io_operations.rst",
            "validation.rst",
            "constants.rst",
            "utils.rst",
        ]

        for expected_file in expected_api_files:
            api_file_path = api_dir / expected_file
            if not api_file_path.exists():
                violations.append(
                    {
                        "file": f"docs/api/{expected_file}",
                        "issue": "Missing API documentation file for module",
                        "type": "missing_api_file",
                    }
                )

        return violations

    def run_full_audit(self) -> dict:
        """Run complete architecture audit."""
        audit_results = {
            "deprecated_imports": self.audit_deprecated_imports(),
            "module_coverage": self.audit_module_coverage(),
            "code_examples": self.audit_code_examples(),
            "api_structure": self.audit_api_structure_refs(),
        }
        return audit_results

    def print_audit_report(self, results: dict):
        """Print formatted architecture audit report."""
        print("🏗️  ARCHITECTURE AUDIT REPORT")
        print("=" * 50)

        total_issues = sum(len(issues) for issues in results.values())
        print(f"Total architecture issues found: {total_issues}\n")

        for category, issues in results.items():
            if not issues:
                print(f"✅ {category.replace('_', ' ').title()}: No issues found")
                continue

            print(f"❌ {category.replace('_', ' ').title()}: {len(issues)} issue(s)")
            print("-" * 30)

            for issue in issues:
                print(f"  📄 File: {issue['file']}")

                if category == "deprecated_imports":
                    print(f"     Pattern: {issue['pattern']}")
                    print(f"     Matches: {issue['matches']}")
                elif category == "module_coverage":
                    print(f"     Issue: {issue['issue']}")
                    if "modules_mentioned" in issue:
                        print(f"     Modules mentioned: {issue['modules_mentioned']}")
                elif category == "code_examples":
                    print(f"     Issue: {issue['issue']}")
                    print(f"     Code snippet: {issue['code_block']}")
                elif category == "api_structure":
                    print(f"     Issue: {issue['issue']}")

                print()

        if total_issues > 0:
            print("\n🔧 ARCHITECTURE RECOMMENDATIONS:")
            print("1. Update deprecated import patterns to use modular structure")
            print("2. Add modular architecture explanations to key documentation")
            print("3. Update code examples to show current best practices")
            print("4. Ensure API documentation covers all current modules")
            print("5. Reference the 5 main sub-packages in architectural descriptions")
        else:
            print("\n🎉 All architecture documentation is aligned!")


def main():
    """Main audit script execution."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        print(f"❌ Documentation directory not found: {docs_dir}")
        return

    print(f"🔍 Auditing architecture alignment in: {docs_dir}")

    auditor = ArchitectureAuditor(docs_dir)
    results = auditor.run_full_audit()
    auditor.print_audit_report(results)


if __name__ == "__main__":
    main()
