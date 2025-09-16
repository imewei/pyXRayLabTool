#!/usr/bin/env python3
"""
Documentation Formatting and Consistency Validation Script.

This script validates documentation formatting, structure, and consistency
across all documentation files.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def check_rst_formatting(file_path: Path) -> List[str]:
    """Check RST file formatting."""
    issues = []
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    # Check for proper section headers
    for i, line in enumerate(lines):
        if line and line[0] in '=-~^*+#':
            # Check if header underline length matches title
            if i > 0:
                title_line = lines[i-1]
                if len(line) != len(title_line):
                    issues.append(f"Line {i+1}: Header underline length mismatch")

    # Check for proper code block formatting
    code_block_pattern = r'\.\. code-block::'
    if re.search(code_block_pattern, content):
        # Ensure proper indentation after code-block
        for i, line in enumerate(lines):
            if '.. code-block::' in line:
                # Check next few lines for proper indentation
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('   '):
                        if not lines[j].strip().startswith('..') and lines[j].strip():
                            issues.append(f"Line {j+1}: Code block content not properly indented")
                            break

    return issues


def check_markdown_formatting(file_path: Path) -> List[str]:
    """Check Markdown file formatting."""
    issues = []
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    # Check for consistent heading styles
    heading_levels = []
    for i, line in enumerate(lines):
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            heading_levels.append((i+1, level))

    # Check for heading level skipping
    for i in range(1, len(heading_levels)):
        prev_level = heading_levels[i-1][1]
        curr_level = heading_levels[i][1]
        if curr_level > prev_level + 1:
            line_num = heading_levels[i][0]
            issues.append(f"Line {line_num}: Heading level skipped (from h{prev_level} to h{curr_level})")

    # Check for code fence consistency
    fence_count = content.count('```')
    if fence_count % 2 != 0:
        issues.append("Unmatched code fences (```)")

    return issues


def check_content_consistency() -> List[str]:
    """Check content consistency across documentation."""
    issues = []
    project_root = Path(__file__).parent.parent

    # Check version consistency
    version_files = [
        project_root / "xraylabtool" / "__init__.py",
        project_root / "docs" / "conf.py",
        project_root / "pyproject.toml",
    ]

    versions = {}
    for file_path in version_files:
        if file_path.exists():
            content = file_path.read_text()
            # Look for version patterns
            version_patterns = [
                r'__version__\s*=\s*["\']([^"\']+)["\']',
                r'version\s*=\s*["\']([^"\']+)["\']',
                r'release\s*=\s*["\']([^"\']+)["\']',
            ]

            for pattern in version_patterns:
                match = re.search(pattern, content)
                if match:
                    versions[str(file_path)] = match.group(1)
                    break

    # Check if all versions match
    if len(set(versions.values())) > 1:
        issues.append(f"Version mismatch across files: {versions}")

    return issues


def check_link_formatting() -> List[str]:
    """Check internal link formatting."""
    issues = []
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    # Find all RST files
    rst_files = list(docs_dir.rglob("*.rst"))

    for rst_file in rst_files:
        if not rst_file.exists():
            continue

        content = rst_file.read_text(encoding='utf-8')

        # Check for broken internal references
        ref_pattern = r':doc:`([^`]+)`'
        refs = re.findall(ref_pattern, content)

        for ref in refs:
            # Check if referenced file exists
            ref_file = docs_dir / f"{ref}.rst"
            if not ref_file.exists():
                issues.append(f"{rst_file.name}: Reference to non-existent file '{ref}'")

        # Check for malformed cross-references
        malformed_patterns = [
            r':ref:`[^`]*`[^`]',  # Unclosed ref
            r':doc:`[^`]*`[^`]',  # Unclosed doc
        ]

        for pattern in malformed_patterns:
            if re.search(pattern, content):
                issues.append(f"{rst_file.name}: Malformed cross-reference")

    return issues


def validate_code_block_languages() -> List[str]:
    """Validate code block language specifications."""
    issues = []
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    # Valid language specifiers
    valid_languages = {
        'python', 'bash', 'shell', 'console', 'text', 'json', 'yaml', 'rst',
        'makefile', 'dockerfile', 'sql', 'none'
    }

    doc_files = list(docs_dir.rglob("*.rst")) + list(docs_dir.rglob("*.md"))
    doc_files.append(project_root / "README.md")

    for doc_file in doc_files:
        if not doc_file.exists():
            continue

        content = doc_file.read_text(encoding='utf-8')

        # Check RST code blocks
        rst_pattern = r'\.\. code-block::\s*([^\s\n]+)'
        rst_langs = re.findall(rst_pattern, content)

        for lang in rst_langs:
            if lang not in valid_languages:
                issues.append(f"{doc_file.name}: Unknown code block language '{lang}'")

        # Check Markdown code blocks
        md_pattern = r'```(\w+)'
        md_langs = re.findall(md_pattern, content)

        for lang in md_langs:
            if lang not in valid_languages:
                issues.append(f"{doc_file.name}: Unknown code block language '{lang}'")

    return issues


def main():
    """Run all documentation formatting validation."""
    print("🧪 Documentation Formatting Validation")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    all_issues = []

    # Check RST files
    print("🔍 Checking RST file formatting...")
    rst_files = list(docs_dir.rglob("*.rst"))
    rst_issues = 0

    for rst_file in rst_files:
        if rst_file.exists():
            issues = check_rst_formatting(rst_file)
            if issues:
                rst_issues += len(issues)
                print(f"   ❌ {rst_file.name}: {len(issues)} issues")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"      - {issue}")
                if len(issues) > 3:
                    print(f"      - ... and {len(issues) - 3} more")
            all_issues.extend(issues)

    if rst_issues == 0:
        print("   ✅ All RST files properly formatted")

    # Check Markdown files
    print("🔍 Checking Markdown file formatting...")
    md_files = list(docs_dir.rglob("*.md")) + [project_root / "README.md"]
    md_issues = 0

    for md_file in md_files:
        if md_file.exists():
            issues = check_markdown_formatting(md_file)
            if issues:
                md_issues += len(issues)
                print(f"   ❌ {md_file.name}: {len(issues)} issues")
                for issue in issues[:3]:
                    print(f"      - {issue}")
                if len(issues) > 3:
                    print(f"      - ... and {len(issues) - 3} more")
            all_issues.extend(issues)

    if md_issues == 0:
        print("   ✅ All Markdown files properly formatted")

    # Check content consistency
    print("🔍 Checking content consistency...")
    consistency_issues = check_content_consistency()
    if consistency_issues:
        print(f"   ❌ {len(consistency_issues)} consistency issues")
        for issue in consistency_issues:
            print(f"      - {issue}")
    else:
        print("   ✅ Content consistency validated")
    all_issues.extend(consistency_issues)

    # Check link formatting
    print("🔍 Checking link formatting...")
    link_issues = check_link_formatting()
    if link_issues:
        print(f"   ❌ {len(link_issues)} link issues")
        for issue in link_issues[:5]:
            print(f"      - {issue}")
        if len(link_issues) > 5:
            print(f"      - ... and {len(link_issues) - 5} more")
    else:
        print("   ✅ Link formatting validated")
    all_issues.extend(link_issues)

    # Check code block languages
    print("🔍 Checking code block languages...")
    lang_issues = validate_code_block_languages()
    if lang_issues:
        print(f"   ❌ {len(lang_issues)} language issues")
        for issue in lang_issues[:5]:
            print(f"      - {issue}")
        if len(lang_issues) > 5:
            print(f"      - ... and {len(lang_issues) - 5} more")
    else:
        print("   ✅ Code block languages validated")
    all_issues.extend(lang_issues)

    # Summary
    print(f"\n📊 Documentation Formatting Summary:")
    print(f"   Total issues found: {len(all_issues)}")

    if len(all_issues) == 0:
        print("🎉 All documentation formatting is valid!")
        return True
    else:
        print("❌ Documentation formatting issues found")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)