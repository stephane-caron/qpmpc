# Makefile for Sphinx documentation
#
# Copyright 2022 Stéphane Caron
# SPDX-License-Identifier: Apache-2.0

SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SOURCEDIR     = src
BUILDDIR      = build
ONLINEDIR     = scaron.info:public_html/doc/qpmpc

.PHONY: help Makefile upload

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

upload: clean html
	rsync -auvz --delete-after $(BUILDDIR)/html/ $(ONLINEDIR)/
