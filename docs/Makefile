# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for documentation development
clean-all: clean
	@echo "Removing all build files..."
	rm -rf $(BUILDDIR)
	rm -rf _static/coverage/
	@echo "Clean complete."

livehtml:
	@echo "Starting live reload server..."
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--host 0.0.0.0 --port 8000 --open-browser

linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

coverage:
	@echo "Generating documentation coverage report..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

spelling:
	@echo "Checking spelling..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest:
	@echo "Running doctests..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Development targets
dev-install:
	@echo "Installing documentation dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed."

dev-build: clean html
	@echo "Development build complete."

dev-serve: livehtml

# Quality targets
check: linkcheck doctest spelling
	@echo "All documentation checks complete."

validate: html
	@echo "Validating HTML output..."
	python -m webbrowser "$(BUILDDIR)/html/index.html"

# Release targets
release-html: clean html
	@echo "Production HTML build complete."
	@echo "Output in $(BUILDDIR)/html/"

release-pdf: clean latexpdf
	@echo "PDF build complete."
	@echo "Output in $(BUILDDIR)/latex/XRayLabTool.pdf"
