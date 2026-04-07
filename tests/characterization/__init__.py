"""Characterization tests: golden-value regression suite for pre-JAX migration.

These tests lock down the exact numerical behaviour of the numpy/scipy stack
(v0.3.0) so that any regression introduced during the JAX migration is caught
immediately.  They must NEVER be relaxed without a deliberate, reviewed change.
"""
