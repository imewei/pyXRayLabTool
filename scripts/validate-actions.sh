#!/bin/bash

# Script to validate that all GitHub Actions workflows use current (non-deprecated) versions
# This helps prevent workflow failures due to deprecated actions

# Continue on errors to show all issues

echo "🔍 Validating GitHub Actions versions..."
echo "ℹ️  Note: GitHub Pages and benchmarking features have been disabled."
echo ""

WORKFLOWS_DIR=".github/workflows"
ISSUES_FOUND=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check for deprecated actions
check_deprecated_action() {
    local file="$1"
    local pattern="$2"
    local action_name="$3"
    local current_version="$4"
    
    if grep -q "$pattern" "$file"; then
        echo -e "  ${RED}❌ DEPRECATED:${NC} $action_name found in $(basename "$file")"
        echo -e "     ${YELLOW}Should use:${NC} $current_version"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        return 1
    fi
    return 0
}

# Function to check for current actions
check_current_action() {
    local file="$1" 
    local pattern="$2"
    local action_name="$3"
    
    if grep -q "$pattern" "$file"; then
        echo -e "  ${GREEN}✅ CURRENT:${NC} $action_name found in $(basename "$file")"
        return 0
    fi
    return 1
}

# Check if workflows directory exists
if [ ! -d "$WORKFLOWS_DIR" ]; then
    echo -e "${RED}❌ Error: $WORKFLOWS_DIR directory not found${NC}"
    exit 1
fi

# Find all workflow files
WORKFLOW_FILES=$(find "$WORKFLOWS_DIR" -name "*.yml" -o -name "*.yaml" 2>/dev/null)

if [ -z "$WORKFLOW_FILES" ]; then
    echo -e "${RED}❌ Error: No workflow files found in $WORKFLOWS_DIR${NC}"
    exit 1
fi

echo -e "📁 Found workflow files:"
for file in $WORKFLOW_FILES; do
    echo "  - $(basename "$file")"
done
echo ""

# Check each workflow file
for file in $WORKFLOW_FILES; do
    echo -e "🔍 Checking $(basename "$file")..."
    
    # Check for deprecated artifact actions (v3)
    check_deprecated_action "$file" "actions/upload-artifact@v3" "upload-artifact@v3" "upload-artifact@v4"
    check_deprecated_action "$file" "actions/download-artifact@v3" "download-artifact@v3" "download-artifact@v4"
    
    # Check for old setup-python (v4)
    check_deprecated_action "$file" "actions/setup-python@v4" "setup-python@v4" "setup-python@v5"
    
    # Check for old cache action (v3)
    check_deprecated_action "$file" "actions/cache@v3" "cache@v3" "cache@v4"
    
    # Check for old codecov action (v3)
    check_deprecated_action "$file" "codecov/codecov-action@v3" "codecov-action@v3" "codecov-action@v4"
    
    # Check for current versions
    check_current_action "$file" "actions/upload-artifact@v4" "upload-artifact@v4"
    check_current_action "$file" "actions/download-artifact@v4" "download-artifact@v4"
    check_current_action "$file" "actions/setup-python@v5" "setup-python@v5"
    check_current_action "$file" "actions/cache@v4" "cache@v4"
    check_current_action "$file" "codecov/codecov-action@v4" "codecov-action@v4"
    check_current_action "$file" "actions/checkout@v4" "checkout@v4"
    
    echo ""
done

# Summary
echo "📊 Validation Summary:"
echo "===================="

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ All GitHub Actions are using current versions!${NC}"
    echo ""
    echo "🎉 Your workflows should run without deprecation issues."
    echo ""
    echo "Current versions in use:"
    echo "  • actions/checkout@v4"
    echo "  • actions/setup-python@v5"  
    echo "  • actions/cache@v4"
    echo "  • actions/upload-artifact@v4"
    echo "  • actions/download-artifact@v4"
    echo "  • codecov/codecov-action@v4"
    
    exit 0
else
    echo -e "${RED}❌ Found $ISSUES_FOUND deprecated action(s)${NC}"
    echo ""
    echo "🚨 CRITICAL: Some actions are deprecated and will stop working on January 30th, 2025!"
    echo ""
    echo "📋 Required updates:"
    echo "  • actions/upload-artifact@v3 → @v4"
    echo "  • actions/download-artifact@v3 → @v4"
    echo "  • actions/setup-python@v4 → @v5"
    echo "  • actions/cache@v3 → @v4"
    echo "  • codecov/codecov-action@v3 → @v4"
    echo ""
    echo "🔧 Please update the workflows before pushing to avoid failures."
    
    exit 1
fi
