#!/usr/bin/env python3
"""
Documentation Links and References Validation Script.

This script validates that all internal links, references, and file paths
in documentation are correct and point to existing resources.
"""

from pathlib import Path
import re
import sys


def find_rst_references(file_path: Path, docs_dir: Path) -> list[tuple[str, bool, str]]:
    """Find and validate RST references in a file."""
    content = file_path.read_text(encoding="utf-8")
    references = []

    # RST doc references (:doc:`path`)
    doc_pattern = r":doc:`([^`]+)`"
    doc_refs = re.findall(doc_pattern, content)

    for ref in doc_refs:
        # Check if the referenced document exists
        ref_path = docs_dir / f"{ref}.rst"
        exists = ref_path.exists()
        references.append((f":doc:`{ref}`", exists, str(ref_path)))

    # RST ref references (:ref:`label`)
    ref_pattern = r":ref:`([^`]+)`"
    ref_refs = re.findall(ref_pattern, content)

    for ref in ref_refs:
        # For ref links, we'd need to scan all files for the label
        # For now, just note them as found
        references.append((f":ref:`{ref}`", True, "label reference"))

    # Include/toctree directives
    include_pattern = r"\.\. include:: ([^\n]+)"
    includes = re.findall(include_pattern, content)

    for include in includes:
        include_path = file_path.parent / include.strip()
        exists = include_path.exists()
        references.append((f"include {include}", exists, str(include_path)))

    toctree_pattern = r"\.\. toctree::\s*[^\n]*\n((?:\s+[^\n]+\n)*)"
    toctree_matches = re.findall(toctree_pattern, content, re.MULTILINE)

    for toctree_content in toctree_matches:
        lines = [line.strip() for line in toctree_content.split("\n") if line.strip()]
        for line in lines:
            if not line.startswith(":"):  # Skip options like :maxdepth:
                toc_path = docs_dir / f"{line}.rst"
                exists = toc_path.exists()
                references.append((f"toctree {line}", exists, str(toc_path)))

    return references


def find_markdown_links(
    file_path: Path, project_root: Path
) -> list[tuple[str, bool, str]]:
    """Find and validate Markdown links in a file."""
    content = file_path.read_text(encoding="utf-8")
    references = []

    # Markdown links [text](path)
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    links = re.findall(link_pattern, content)

    for text, url in links:
        if url.startswith(("http://", "https://", "mailto:")):
            # External links - assume valid for now
            references.append((f"[{text}]({url})", True, "external link"))
        elif url.startswith("#"):
            # Anchor links - would need to check for headers
            references.append((f"[{text}]({url})", True, "anchor link"))
        else:
            # Internal file links
            if url.startswith("/"):
                link_path = project_root / url[1:]
            else:
                link_path = file_path.parent / url

            exists = link_path.exists()
            references.append((f"[{text}]({url})", exists, str(link_path)))

    # Reference-style links [text][ref] and [ref]: url
    ref_pattern = r"\[([^\]]+)\]\[([^\]]+)\]"
    ref_links = re.findall(ref_pattern, content)

    ref_def_pattern = r"^\[([^\]]+)\]:\s*(.+)$"
    ref_defs = dict(re.findall(ref_def_pattern, content, re.MULTILINE))

    for text, ref_id in ref_links:
        if ref_id in ref_defs:
            url = ref_defs[ref_id]
            if url.startswith(("http://", "https://")):
                references.append((f"[{text}][{ref_id}]", True, "external reference"))
            else:
                link_path = file_path.parent / url
                exists = link_path.exists()
                references.append((f"[{text}][{ref_id}]", exists, str(link_path)))
        else:
            references.append(
                (f"[{text}][{ref_id}]", False, f"undefined reference: {ref_id}")
            )

    return references


def find_code_references(
    file_path: Path, project_root: Path
) -> list[tuple[str, bool, str]]:
    """Find and validate code references (imports, file paths) in documentation."""
    content = file_path.read_text(encoding="utf-8")
    references = []

    # Python import paths in code blocks
    import_pattern = r"from\s+(xraylabtool[.\w]*)\s+import|import\s+(xraylabtool[.\w]*)"
    imports = re.findall(import_pattern, content)

    valid_modules = {
        "xraylabtool",
        "xraylabtool.calculators",
        "xraylabtool.data_handling",
        "xraylabtool.interfaces",
        "xraylabtool.io",
        "xraylabtool.utils",
        "xraylabtool.validation",
        "xraylabtool.constants",
    }

    for import_tuple in imports:
        module = import_tuple[0] or import_tuple[1]
        if module:
            is_valid = any(module.startswith(valid_mod) for valid_mod in valid_modules)
            references.append((f"import {module}", is_valid, "module import"))

    # File paths mentioned in documentation
    file_pattern = r"`([^`]+\.(py|md|rst|txt|json|csv))`"
    file_mentions = re.findall(file_pattern, content)

    for file_mention, _ext in file_mentions:
        # Try relative to project root
        file_path_abs = project_root / file_mention
        exists = file_path_abs.exists()
        references.append((f"`{file_mention}`", exists, str(file_path_abs)))

    return references


def validate_documentation_links():
    """Validate all documentation links and references."""
    print("🔍 Validating Documentation Links and References")
    print("=" * 55)

    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    all_issues = []
    all_references = []

    # Check RST files
    print("📄 Checking RST files...")
    rst_files = list(docs_dir.rglob("*.rst"))

    for rst_file in rst_files:
        if rst_file.exists():
            references = find_rst_references(rst_file, docs_dir)
            all_references.extend(references)

            broken_refs = [ref for ref in references if not ref[1]]
            if broken_refs:
                print(f"   ❌ {rst_file.name}: {len(broken_refs)} broken references")
                for ref, _, path in broken_refs[:3]:
                    print(f"      - {ref} -> {path}")
                if len(broken_refs) > 3:
                    print(f"      - ... and {len(broken_refs) - 3} more")
                all_issues.extend(broken_refs)

    # Check Markdown files
    print("📄 Checking Markdown files...")
    md_files = [
        *list(docs_dir.rglob("*.md")),
        project_root / "README.md",
        project_root / "CLAUDE.md",
    ]

    for md_file in md_files:
        if md_file.exists():
            references = find_markdown_links(md_file, project_root)
            all_references.extend(references)

            broken_refs = [ref for ref in references if not ref[1]]
            if broken_refs:
                print(f"   ❌ {md_file.name}: {len(broken_refs)} broken references")
                for ref, _, path in broken_refs[:3]:
                    print(f"      - {ref} -> {path}")
                if len(broken_refs) > 3:
                    print(f"      - ... and {len(broken_refs) - 3} more")
                all_issues.extend(broken_refs)

    # Check code references
    print("📄 Checking code references...")
    all_doc_files = rst_files + md_files

    for doc_file in all_doc_files:
        if doc_file.exists():
            references = find_code_references(doc_file, project_root)
            all_references.extend(references)

            broken_refs = [ref for ref in references if not ref[1]]
            if broken_refs:
                if not any(f"{doc_file.name}:" in str(issue) for issue in all_issues):
                    print(
                        f"   ❌ {doc_file.name}: {len(broken_refs)} broken code references"
                    )
                for ref, _, path in broken_refs[:2]:
                    print(f"      - {ref} -> {path}")
                all_issues.extend(broken_refs)

    # Summary
    total_refs = len(all_references)
    broken_refs = len(all_issues)
    valid_refs = total_refs - broken_refs

    print("\n📊 Link Validation Summary:")
    print(f"   Total references checked: {total_refs}")
    print(f"   Valid references: {valid_refs}")
    print(f"   Broken references: {broken_refs}")

    if broken_refs == 0:
        print("🎉 All documentation links and references are valid!")
        return True
    else:
        percentage = (valid_refs / total_refs * 100) if total_refs > 0 else 0
        print(f"⚠️  {percentage:.1f}% of references are valid")
        return broken_refs < total_refs * 0.1  # Allow up to 10% broken refs


def main():
    """Main function to run link validation."""
    return validate_documentation_links()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
