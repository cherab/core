.PHONY: help doc doc-view doc-clean test sdist setupclean

PYTHON      = python3
SPHINXBUILD = sphinx-build

help:
	@echo "Check Makefile source"

sdist: doc
	$(PYTHON) setup.py sdist

test:
	cd test && PYTHONPAH=.. $(PYTHON) testall.py

setupclean:
	$(PYTHON) setup.py clean --al

doc:
	cd doc && SPHINXBUILD=$(SPHINXBUILD) make html

doc-view:
	firefox doc/_build/html/index.html

doc-clean:
	cd doc && make clean

