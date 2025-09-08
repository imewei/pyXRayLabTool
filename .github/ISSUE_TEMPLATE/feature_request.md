---
name: Feature Request
about: Suggest an idea or enhancement for XRayLabTool
title: '[FEATURE] '
labels: ['enhancement', 'needs-triage']
assignees: ''

---

## Feature Description

**Is your feature request related to a problem? Please describe:**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like:**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered:**
A clear and concise description of any alternative solutions or features you've considered.

## Use Case

**Scientific context:**
- What X-ray analysis technique would benefit? (XRR, SAXS, XRD, XAS, GIXS, etc.)
- Is this for synchrotron or laboratory X-ray work?
- What materials or sample types would this help with?

**Specific example:**
```python
# Example of how you'd like to use the feature
import xraylabtool as xlt

# Your proposed API or usage
result = xlt.new_feature(
    parameter1="value1",
    parameter2=123
)
```

## Implementation Ideas

**Suggested approach:**
- How do you envision this feature working?
- What parameters or options should it have?
- Should it integrate with existing functions?

**API design:**
```python
# Proposed function signature
def new_function(
    required_param: str,
    optional_param: float = 1.0,
    advanced_option: bool = False
) -> ResultType:
    """
    Brief description of what this function would do.

    Parameters
    ----------
    required_param : str
        Description of required parameter
    optional_param : float, default 1.0
        Description of optional parameter
    advanced_option : bool, default False
        Description of advanced option

    Returns
    -------
    ResultType
        Description of return value
    """
```

## Scientific Background

**Physical principles:**
- What X-ray physics principles does this feature involve?
- Are there established formulas or algorithms?
- Any relevant scientific papers or references?

**Data sources:**
- Would this require new atomic data or databases?
- Are there standard reference values to validate against?
- Any integration with existing databases (CXRO, NIST, etc.)?

**Validation approach:**
- How would you verify the feature works correctly?
- Are there known test cases or reference calculations?
- What accuracy requirements are needed?

## Impact and Priority

**Who would benefit:**
- [ ] Synchrotron scientists
- [ ] Materials researchers
- [ ] X-ray optics developers
- [ ] Students and educators
- [ ] Industrial users
- [ ] Software developers

**Expected usage frequency:**
- [ ] Daily use for common workflows
- [ ] Weekly use for specific analyses
- [ ] Occasional use for specialized cases
- [ ] One-time setup or configuration

**Urgency level:**
- [ ] Critical - blocking current research
- [ ] High - would significantly improve workflows
- [ ] Medium - nice to have enhancement
- [ ] Low - future consideration

## Implementation Complexity

**Estimated difficulty:**
- [ ] Simple - straightforward addition to existing functions
- [ ] Moderate - new function but uses existing infrastructure
- [ ] Complex - requires significant new code or algorithms
- [ ] Major - would require architectural changes

**Dependencies:**
- Would this require new Python dependencies?
- Any external libraries or data sources needed?
- Compatibility considerations with existing code?

**Performance considerations:**
- Should this be optimized for large datasets?
- Any memory usage concerns?
- Parallel processing requirements?

## Additional Context

**Related issues:**
- Link any related issues or discussions
- Reference similar features in other software
- Mention any workarounds currently being used

**Examples from other tools:**
- How do similar tools implement this feature?
- What are the pros/cons of different approaches?
- Any standards or best practices to follow?

---

**Additional Information:**
Add any other context, screenshots, plots, or references about the feature request here.
