# Renal-Net test runner
#
#   make smoke   — fast synthetic tests, no model needed
#   make full    — smoke + integration (requires model + CT data)
#   make clean   — delete generated reports and figures

REPORTS_DIR = reports
PYTEST = python -m pytest

.PHONY: install install-dev smoke full clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

smoke:
	mkdir -p $(REPORTS_DIR)
	$(PYTEST) tests/test_smoke.py \
		-v \
		--html=$(REPORTS_DIR)/report_smoke.html \
		--self-contained-html

full:
	mkdir -p $(REPORTS_DIR)
	$(PYTEST) tests/test_smoke.py tests/test_integration.py \
		-v \
		--html=$(REPORTS_DIR)/report_full.html \
		--self-contained-html

clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.mypy_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ruff_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('build')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('dist')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('htmlcov')]"
	rm -rf $(REPORTS_DIR)
	@echo "Clean complete."
	