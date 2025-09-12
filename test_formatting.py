#!/usr/bin/env python3
"""
Test file to demonstrate automatic pre-commit formatting.
This file intentionally has formatting issues that will be fixed.
"""

import os
import sys
import time
from typing import Dict, List  # Bad spacing

import numpy as np


def badly_formatted_function(x: int, y: str):
    """Function with bad formatting."""
    if x > 5:
        result = x * 2
        return result
    else:
        return y.upper()


class BadlyFormattedClass:
    """Class with formatting issues."""

    def __init__(self, value: int):
        self.value = value

    def method_with_bad_spacing(self):
        return self.value + 10


if __name__ == "__main__":
    obj = BadlyFormattedClass(42)
    print(badly_formatted_function(5, "test"))
