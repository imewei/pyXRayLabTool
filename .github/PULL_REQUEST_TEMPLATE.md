# Pull Request

## Summary

**Brief description of changes:**
Provide a clear and concise summary of what this PR does.

**Related issue(s):**
- Fixes #(issue number)
- Relates to #(issue number)
- Part of #(issue number)

## Type of Change

Please mark the relevant option with an "x":

- [ ] 🐛 **Bug fix** - non-breaking change that fixes an issue
- [ ] ✨ **New feature** - non-breaking change that adds functionality
- [ ] 🔧 **Enhancement** - improvement to existing functionality
- [ ] 📚 **Documentation** - updates to documentation only
- [ ] 🚀 **Performance** - performance improvement
- [ ] 🧪 **Tests** - adding or improving tests
- [ ] 🔨 **Refactoring** - code changes that neither fix bugs nor add features
- [ ] 💥 **Breaking change** - change that would cause existing functionality to not work as expected

## Scientific Context

**X-ray technique relevance:**
- [ ] X-ray reflectivity (XRR)
- [ ] Small-angle X-ray scattering (SAXS)
- [ ] X-ray diffraction (XRD)
- [ ] X-ray absorption spectroscopy (XAS)
- [ ] Grazing incidence X-ray scattering (GIXS)
- [ ] General X-ray optics
- [ ] Other: _________________

**Target users:**
- [ ] Synchrotron scientists
- [ ] Materials researchers
- [ ] X-ray optics developers
- [ ] Students and educators
- [ ] Industrial users

## Changes Made

### Code Changes
- List specific functions, modules, or files modified
- Highlight any API changes or new public methods
- Mention any performance improvements

### Documentation Changes
- [ ] Updated docstrings
- [ ] Added examples
- [ ] Updated README.md
- [ ] Updated CHANGELOG.md
- [ ] Added/updated RST documentation

### Tests
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Added performance benchmarks
- [ ] All existing tests pass

## Testing

**Testing performed:**
```bash
# Commands used to test your changes
make test
python -m pytest tests/test_your_feature.py -v
xraylabtool calc --formula "SiO2" --energy 8.0 --density 2.2
```

**Test coverage:**
- [ ] Added tests for new functionality
- [ ] Maintained or improved overall test coverage
- [ ] All edge cases considered and tested

**Manual testing:**
- [ ] Tested with various chemical formulas
- [ ] Tested with different energy ranges
- [ ] Verified numerical accuracy against known values
- [ ] Tested CLI functionality (if applicable)

## Performance Impact

**Performance considerations:**
- [ ] No performance impact expected
- [ ] Performance improvement (provide details below)
- [ ] Performance regression possible (justified below)
- [ ] Not applicable

**Benchmarks (if applicable):**
```
# Include benchmark results for performance-related changes
Before: X calculations/second
After:  Y calculations/second
Improvement: Z% faster
```

## Breaking Changes

**If this is a breaking change:**
- Describe what will break
- Explain why this change is necessary
- Provide migration path for users
- Update version following semantic versioning

## Dependencies

**New dependencies:**
- [ ] No new dependencies
- [ ] Added runtime dependencies (list below)
- [ ] Added development dependencies (list below)

**Dependency updates:**
- List any dependency version changes
- Explain reasons for updates
- Verify compatibility

## Validation

### Scientific Validation
- [ ] Verified against known reference values
- [ ] Compared with established X-ray databases (CXRO, NIST)
- [ ] Cross-checked with other scientific software
- [ ] Reviewed by domain expert

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Type hints added for new functions
- [ ] Docstrings follow NumPy style
- [ ] No linting errors (flake8, mypy)
- [ ] Code is well-commented

### Documentation
- [ ] Public API documented
- [ ] Examples provided for new features
- [ ] CHANGELOG.md updated
- [ ] Version bumped if necessary

## Checklist

**Before submitting:**
- [ ] I have read the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

**Scientific software specific:**
- [ ] Physical constants are accurate and properly cited
- [ ] Calculations are numerically stable
- [ ] Units are clearly documented and consistent
- [ ] Error messages are informative for scientific users
- [ ] Performance is suitable for typical scientific datasets

## Additional Notes

**Special considerations:**
- Any special setup required for testing?
- Platform-specific considerations?
- Backwards compatibility concerns?
- Future work or follow-up issues?

**Screenshots (if applicable):**
For CLI changes, performance plots, or documentation updates, include relevant screenshots.

---

**Reviewer Notes:**
- Tag specific reviewers if domain expertise is needed
- Highlight areas that need particular attention
- Mention any experimental or advanced features
