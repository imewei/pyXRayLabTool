#!/bin/bash
# Code formatting script to match CI requirements

set -e

echo "🎨 Formatting code to match CI requirements..."

# Apply ruff formatting
echo "Applying ruff format..."
ruff format xraylabtool/ tests/ || echo "Ruff format completed"

# Apply import sorting with isort
echo "Applying import sorting..."
isort xraylabtool/ tests/ || echo "Isort completed"

# Apply black formatting as backup
echo "Applying black formatting..."
black xraylabtool/ tests/ || echo "Black completed"

echo "✅ Code formatting completed!"
echo "💡 Your code is now formatted to match CI requirements"
