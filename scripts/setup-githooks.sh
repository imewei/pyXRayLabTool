#!/bin/bash
#
# Setup script to install custom Git hooks for XRayLabTool
# Run this after cloning the repository to enable automatic formatting commits
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up XRayLabTool Git hooks...${NC}"

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"
CUSTOM_HOOKS_DIR="$REPO_ROOT/.githooks"

# Check if custom hooks directory exists
if [ ! -d "$CUSTOM_HOOKS_DIR" ]; then
    echo -e "${RED}❌ Custom hooks directory not found: $CUSTOM_HOOKS_DIR${NC}"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install post-commit hook
if [ -f "$CUSTOM_HOOKS_DIR/post-commit" ]; then
    echo -e "${BLUE}📝 Installing post-commit hook...${NC}"
    cp "$CUSTOM_HOOKS_DIR/post-commit" "$HOOKS_DIR/post-commit"
    chmod +x "$HOOKS_DIR/post-commit"
    echo -e "${GREEN}✅ Post-commit hook installed${NC}"
else
    echo -e "${YELLOW}⚠️  Post-commit hook not found in $CUSTOM_HOOKS_DIR${NC}"
fi

# Install pre-commit hooks if pre-commit is available
if command -v pre-commit >/dev/null 2>&1; then
    echo -e "${BLUE}🔍 Installing pre-commit hooks...${NC}"
    pre-commit install
    echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}⚠️  pre-commit not found. Install with: pip install pre-commit${NC}"
    echo -e "${BLUE}💡 After installing pre-commit, run: pre-commit install${NC}"
fi

echo -e "\n${GREEN}🎉 Git hooks setup completed!${NC}"
echo -e "\n${BLUE}📋 What was installed:${NC}"
echo -e "  • ${GREEN}Post-commit hook${NC}: Automatically commits pre-commit formatting changes"
echo -e "  • ${GREEN}Pre-commit hooks${NC}: Code quality and formatting checks"

echo -e "\n${BLUE}💡 Usage:${NC}"
echo -e "  • Make changes and commit normally"
echo -e "  • Pre-commit hooks will run automatically"
echo -e "  • If formatting changes are made, they'll be auto-committed"
echo -e "  • All changes will be properly tracked in git history"

echo -e "\n${BLUE}🔧 Configuration:${NC}"
echo -e "  • To enable auto-push of formatting commits, edit:"
echo -e "    ${YELLOW}.githooks/post-commit${NC} (uncomment push lines)"
echo -e "  • To disable auto-commit, remove: ${YELLOW}.git/hooks/post-commit${NC}"
