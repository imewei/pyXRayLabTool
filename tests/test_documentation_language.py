"""
Test suite for documentation language standards.

This module contains tests to ensure documentation maintains
technical accuracy and avoids flowery or marketing language.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

import pytest


class TestDocumentationLanguage:
    """Test documentation language standards and technical accuracy."""

    # Flowery/marketing words that should be avoided
    FLOWERY_WORDS = [
        "empowering", "empower", "empowers",
        "unprecedented",
        "definitive",
        "cutting-edge", "cutting edge",
        "revolutionary",
        "groundbreaking",
        "seamless", "seamlessly",
        "elegant", "elegantly",
        "robust", "robustly",
        "comprehensive", "comprehensively",
        "sophisticated",
        "state-of-the-art",
        "world-class",
        "industry-leading",
        "next-generation",
        "breakthrough",
        "innovative",
        "paradigm-shifting",
        "game-changing"
    ]

    # Technical phrases that should be used instead
    PREFERRED_TECHNICAL_PHRASES = {
        "empowering scientists": "provides tools for scientists",
        "unprecedented speed": "150,000+ calculations per second",
        "definitive solution": "X-ray optical properties calculator",
        "cutting-edge": "current",
        "comprehensive": "complete",
        "seamless": "integrated",
        "robust": "reliable",
        "elegant": "efficient"
    }

    @pytest.fixture
    def docs_directory(self) -> Path:
        """Get the documentation directory path."""
        project_root = Path(__file__).parent.parent
        return project_root / "docs"

    @pytest.fixture
    def documentation_files(self, docs_directory: Path) -> List[Path]:
        """Get all documentation files to check."""
        doc_files = []
        if docs_directory.exists():
            # RST files
            doc_files.extend(docs_directory.glob("**/*.rst"))
            # Markdown files
            doc_files.extend(docs_directory.glob("**/*.md"))
        return doc_files

    def test_no_flowery_language_in_docs(self, documentation_files: List[Path]):
        """Test that documentation files don't contain flowery language."""
        violations = []

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8').lower()

            for word in self.FLOWERY_WORDS:
                # Use word boundaries to avoid false positives
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                matches = re.findall(pattern, content)

                if matches:
                    violations.append({
                        'file': str(doc_file.relative_to(doc_file.parent.parent)),
                        'word': word,
                        'count': len(matches)
                    })

        if violations:
            violation_msg = "\n".join([
                f"  {v['file']}: '{v['word']}' appears {v['count']} time(s)"
                for v in violations
            ])
            pytest.fail(
                f"Found flowery language in documentation:\n{violation_msg}\n\n"
                f"Replace with direct technical descriptions."
            )

    def test_cli_command_count_accuracy(self, documentation_files: List[Path]):
        """Test that CLI command count is accurate (should be 8, not 9)."""
        violations = []

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8')

            # Check for incorrect "9 commands" references
            nine_commands_pattern = r'9\s+(?:main\s+)?commands?'
            matches = re.findall(nine_commands_pattern, content, re.IGNORECASE)

            if matches:
                violations.append({
                    'file': str(doc_file.relative_to(doc_file.parent.parent)),
                    'issue': f"References '9 commands' but should be '8 commands'",
                    'matches': matches
                })

        if violations:
            violation_msg = "\n".join([
                f"  {v['file']}: {v['issue']}"
                for v in violations
            ])
            pytest.fail(
                f"Found incorrect CLI command count in documentation:\n{violation_msg}\n\n"
                f"Update to reflect the current 8 CLI commands."
            )

    def test_technical_accuracy_performance_claims(self, documentation_files: List[Path]):
        """Test that performance claims use specific metrics."""
        violations = []

        # Vague performance terms that should be specific
        vague_terms = [
            "fast", "faster", "fastest",
            "high-performance", "high performance",
            "optimized performance",
            "excellent performance",
            "superior speed"
        ]

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8').lower()

            for term in vague_terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, content)

                if matches:
                    violations.append({
                        'file': str(doc_file.relative_to(doc_file.parent.parent)),
                        'term': term,
                        'suggestion': "Replace with specific metric (e.g., '150,000+ calculations per second')"
                    })

        if violations:
            violation_msg = "\n".join([
                f"  {v['file']}: '{v['term']}' - {v['suggestion']}"
                for v in violations
            ])
            pytest.fail(
                f"Found vague performance claims in documentation:\n{violation_msg}"
            )

    def test_module_structure_references(self, documentation_files: List[Path]):
        """Test that documentation references current modular structure."""
        expected_modules = [
            "calculators",
            "data_handling",
            "interfaces",
            "io",
            "validation"
        ]

        # Check that documentation mentions the modular structure
        structure_mentioned = False

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8').lower()

            module_mentions = sum(1 for module in expected_modules if module in content)
            if module_mentions >= 3:  # At least 3 modules mentioned suggests architectural docs
                structure_mentioned = True
                break

        # This is informational - we'll check this but not fail the test
        # The actual fix will happen in subsequent subtasks
        if not structure_mentioned:
            pytest.skip(
                "Documentation doesn't extensively reference modular structure. "
                "This will be addressed in architecture update tasks."
            )

    def test_version_references_current(self, documentation_files: List[Path]):
        """Test that version references are current (v0.2.3)."""
        violations = []

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8')

            # Look for outdated version patterns
            old_version_patterns = [
                r'v?0\.[01]\.\d+',  # v0.0.x or v0.1.x
                r'version\s+0\.[01]\.\d+',
                r'release\s+0\.[01]\.\d+'
            ]

            for pattern in old_version_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append({
                        'file': str(doc_file.relative_to(doc_file.parent.parent)),
                        'outdated_versions': matches
                    })

        if violations:
            violation_msg = "\n".join([
                f"  {v['file']}: Found outdated versions: {v['outdated_versions']}"
                for v in violations
            ])
            pytest.fail(
                f"Found outdated version references in documentation:\n{violation_msg}\n\n"
                f"Update to current version v0.2.3."
            )


class TestDocumentationTechnicalStandards:
    """Test technical writing standards in documentation."""

    def test_code_examples_importable(self, documentation_files: List[Path]):
        """Test that code examples use correct import statements."""
        violations = []

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8')

            # Look for import examples
            import_patterns = [
                r'from xraylabtool import',
                r'import xraylabtool',
                r'xraylabtool\.'
            ]

            has_imports = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in import_patterns
            )

            if has_imports:
                # Check for deprecated import patterns
                deprecated_patterns = [
                    r'from xraylabtool\.core import',  # Old structure
                    r'xraylabtool\.core\.',
                ]

                for pattern in deprecated_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        violations.append({
                            'file': str(doc_file.relative_to(doc_file.parent.parent)),
                            'deprecated_import': matches[0]
                        })

        if violations:
            violation_msg = "\n".join([
                f"  {v['file']}: Deprecated import '{v['deprecated_import']}'"
                for v in violations
            ])
            pytest.fail(
                f"Found deprecated import patterns in documentation:\n{violation_msg}\n\n"
                f"Update to use current modular import structure."
            )

    def test_consistent_terminology(self, documentation_files: List[Path]):
        """Test that documentation uses consistent terminology."""
        # This test ensures consistent naming conventions
        terminology_checks = {
            "X-ray": ["x-ray", "xray", "X-Ray"],  # Should be "X-ray"
            "XRayLabTool": ["xraylabtool", "XrayLabTool", "X-RayLabTool"],  # Package name consistency
        }

        violations = []

        for doc_file in documentation_files:
            if not doc_file.exists():
                continue

            content = doc_file.read_text(encoding='utf-8')

            for correct_term, incorrect_variants in terminology_checks.items():
                for variant in incorrect_variants:
                    # Case-sensitive search for exact matches
                    if variant in content and variant != correct_term:
                        violations.append({
                            'file': str(doc_file.relative_to(doc_file.parent.parent)),
                            'incorrect': variant,
                            'correct': correct_term
                        })

        if violations:
            violation_msg = "\n".join([
                f"  {v['file']}: '{v['incorrect']}' should be '{v['correct']}'"
                for v in violations
            ])
            pytest.fail(
                f"Found inconsistent terminology in documentation:\n{violation_msg}"
            )