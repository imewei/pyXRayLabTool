---
name: Bug Report
about: Create a report to help us improve XRayLabTool
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''

---

## Bug Description

**Brief description of the bug:**
A clear and concise description of what the bug is.

**Expected behavior:**
A clear and concise description of what you expected to happen.

**Actual behavior:**
A clear and concise description of what actually happened.

## Reproduction Steps

Please provide the minimal code needed to reproduce the issue:

```python
import xraylabtool as xlt

# Your code that produces the bug
result = xlt.calculate_single_material_properties(
    formula="SiO2",
    energy=8.0,  # keV
    density=2.2  # g/cm³
)
print(result)
```

**Steps to reproduce:**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Environment Information

**XRayLabTool version:** (e.g., 0.1.10)
```bash
# Get version with:
python -c "import xraylabtool; print(xraylabtool.__version__)"
```

**Python version:** (e.g., 3.12.0)
```bash
python --version
```

**Operating System:** (e.g., Ubuntu 22.04, Windows 11, macOS 14.0)

**Dependencies versions:**
```bash
# Get dependency versions with:
pip list | grep -E "(numpy|scipy|pandas|matplotlib)"
```

## Error Output

**Full error traceback:**
```python
# Paste the complete error traceback here
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  ...
```

**Additional error information:**
- Any warnings that appeared
- Screenshots if GUI-related
- Log files if available

## Additional Context

**Chemical formula details:**
- Formula used: (e.g., "SiO2", "Al2O3", "Fe2O3")
- Density value: (e.g., 2.2 g/cm³)
- Energy range: (e.g., 8.0 keV, [1, 2, 3] keV, np.linspace(1, 30, 100))

**Scientific context:**
- What X-ray analysis technique are you working with? (XRR, SAXS, XRD, etc.)
- Is this for synchrotron or laboratory X-ray work?
- Any specific material or measurement constraints?

**Attempted solutions:**
- What have you tried to fix or work around the issue?
- Have you searched existing issues?
- Any related Stack Overflow posts or documentation you've consulted?

## Reproducibility

- [ ] I can reproduce this bug consistently
- [ ] This bug occurs intermittently
- [ ] This bug only occurs with specific inputs
- [ ] I have tested with the latest version of XRayLabTool

**Minimal working example:**
If possible, provide the smallest possible example that demonstrates the bug.

## Impact

**How does this bug affect your work?**
- [ ] Blocks my current work completely
- [ ] Causes incorrect scientific results
- [ ] Performance issue affecting productivity
- [ ] Minor inconvenience but workaround exists
- [ ] Feature request / enhancement

---

**Additional Information:**
Add any other context about the problem here. Screenshots, plots, or data files can be very helpful for understanding the issue.
