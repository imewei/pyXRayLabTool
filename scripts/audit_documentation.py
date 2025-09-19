#!/usr/bin/env python3
"""
Documentation audit script to identify language and technical issues.

This script scans documentation files for flowery language, technical
inaccuracies, and outdated references that need to be updated.
"""

import re
from pathlib import Path


class DocumentationAuditor:
    """Audit documentation for language and technical accuracy."""

    # Flowery/marketing words that should be avoided
    FLOWERY_WORDS = [
        "empowering",
        "empower",
        "empowers",
        "unprecedented",
        "definitive",
        "cutting-edge",
        "cutting edge",
        "revolutionary",
        "groundbreaking",
        "seamless",
        "seamlessly",
        "elegant",
        "elegantly",
        "robust",
        "robustly",
        "comprehensive",
        "comprehensively",
        "sophisticated",
        "state-of-the-art",
        "world-class",
        "industry-leading",
        "next-generation",
        "breakthrough",
        "innovative",
        "paradigm-shifting",
        "game-changing",
    ]

    def __init__(self, docs_directory: Path):
        """Initialize auditor with documentation directory."""
        self.docs_dir = docs_directory
        self.violations = []

    def get_documentation_files(self) -> list[Path]:
        """Get all documentation files to audit."""
        doc_files = []
        if self.docs_dir.exists():
            # RST files
            doc_files.extend(self.docs_dir.glob("**/*.rst"))
            # Markdown files
            doc_files.extend(self.docs_dir.glob("**/*.md"))
        return doc_files

    def audit_flowery_language(self) -> list[dict]:
        """Audit documentation for flowery language."""
        violations = []
        doc_files = self.get_documentation_files()

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue

            for word in self.FLOWERY_WORDS:
                # Use word boundaries to avoid false positives
                pattern = r"\b" + re.escape(word.lower()) + r"\b"
                matches = re.findall(pattern, content)

                if matches:
                    violations.append(
                        {
                            "file": str(doc_file.relative_to(self.docs_dir.parent)),
                            "word": word,
                            "count": len(matches),
                            "type": "flowery_language",
                        }
                    )

        return violations

    def audit_cli_command_count(self) -> list[dict]:
        """Audit for incorrect CLI command count references."""
        violations = []
        doc_files = self.get_documentation_files()

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            # Check for incorrect "9 commands" references
            nine_commands_pattern = r"9\s+(?:main\s+)?commands?"
            re.findall(nine_commands_pattern, content, re.IGNORECASE)

            # Since 9 commands is actually correct, don't flag this as an error
            # Check for incorrect counts instead
            incorrect_patterns = [
                r"8\s+(?:main\s+)?commands?",
                r"7\s+(?:main\s+)?commands?",
            ]
            for pattern in incorrect_patterns:
                incorrect_matches = re.findall(pattern, content, re.IGNORECASE)
                if incorrect_matches:
                    violations.append(
                        {
                            "file": str(doc_file.relative_to(self.docs_dir.parent)),
                            "issue": (
                                "References incorrect command count, should be '9 commands'"
                            ),
                            "matches": incorrect_matches,
                            "type": "cli_count_error",
                        }
                    )

        return violations

    def audit_performance_claims(self) -> list[dict]:
        """Audit for vague performance claims."""
        violations = []
        doc_files = self.get_documentation_files()

        # Vague performance terms that should be specific
        vague_terms = [
            "fast",
            "faster",
            "fastest",
            "high-performance",
            "high performance",
            "optimized performance",
            "excellent performance",
            "superior speed",
            "blazing fast",
            "lightning fast",
        ]

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue

            for term in vague_terms:
                pattern = r"\b" + re.escape(term.lower()) + r"\b"
                matches = re.findall(pattern, content)

                if matches:
                    violations.append(
                        {
                            "file": str(doc_file.relative_to(self.docs_dir.parent)),
                            "term": term,
                            "suggestion": (
                                "Replace with specific metric (e.g., '150,000+ calculations per second')"
                            ),
                            "type": "vague_performance",
                        }
                    )

        return violations

    def audit_version_references(self) -> list[dict]:
        """Audit for outdated version references."""
        violations = []
        doc_files = self.get_documentation_files()

        for doc_file in doc_files:
            if not doc_file.exists():
                continue

            try:
                content = doc_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            # Look for outdated version patterns
            old_version_patterns = [
                r"v?0\.[01]\.\d+",  # v0.0.x or v0.1.x
                r"version\s+0\.[01]\.\d+",
                r"release\s+0\.[01]\.\d+",
            ]

            for pattern in old_version_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(
                        {
                            "file": str(doc_file.relative_to(self.docs_dir.parent)),
                            "outdated_versions": matches,
                            "type": "outdated_version",
                        }
                    )

        return violations

    def run_full_audit(self) -> dict:
        """Run complete documentation audit."""
        audit_results = {
            "flowery_language": self.audit_flowery_language(),
            "cli_count_errors": self.audit_cli_command_count(),
            "vague_performance": self.audit_performance_claims(),
            "outdated_versions": self.audit_version_references(),
        }
        return audit_results

    def print_audit_report(self, results: dict):
        """Print formatted audit report."""
        print("📋 DOCUMENTATION AUDIT REPORT")
        print("=" * 50)

        total_issues = sum(len(issues) for issues in results.values())
        print(f"Total issues found: {total_issues}\n")

        for category, issues in results.items():
            if not issues:
                print(f"✅ {category.replace('_', ' ').title()}: No issues found")
                continue

            print(f"❌ {category.replace('_', ' ').title()}: {len(issues)} issue(s)")
            print("-" * 30)

            for issue in issues:
                print(f"  📄 File: {issue['file']}")

                if category == "flowery_language":
                    print(
                        f"     Word: '{issue['word']}' (appears {issue['count']} time(s))"
                    )
                elif category == "cli_count_errors":
                    print(f"     Issue: {issue['issue']}")
                elif category == "vague_performance":
                    print(f"     Term: '{issue['term']}'")
                    print(f"     Suggestion: {issue['suggestion']}")
                elif category == "outdated_versions":
                    print(f"     Outdated versions: {issue['outdated_versions']}")

                print()

        if total_issues > 0:
            print("\n🔧 RECOMMENDATIONS:")
            print("1. Replace flowery language with direct technical descriptions")
            print("2. Update CLI command count from 9 to 8")
            print("3. Use specific performance metrics (150,000+ calc/sec)")
            print("4. Update version references to v0.2.3")
        else:
            print("\n🎉 All documentation language checks passed!")


def main():
    """Main audit script execution."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        print(f"❌ Documentation directory not found: {docs_dir}")
        return

    print(f"🔍 Auditing documentation in: {docs_dir}")

    auditor = DocumentationAuditor(docs_dir)
    results = auditor.run_full_audit()
    auditor.print_audit_report(results)


if __name__ == "__main__":
    main()
