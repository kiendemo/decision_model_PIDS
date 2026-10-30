# Thin wrappers over the CLI. PYTHON may point at any interpreter with the pinned environment
# (requirements-lock.txt); the package is used from src/ without installation.
PYTHON ?= python
THREADS ?= 8
JOBS ?= 5
ENV = PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$(THREADS) \
      MKL_NUM_THREADS=$(THREADS) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HABITUS = $(ENV) $(PYTHON) -m habitus.cli

.PHONY: help test test-assets gates reproduce reproduce-all release status

help:
	@echo "make test | test-assets | gates | reproduce | reproduce-all | release | status"

test:            ## unit tests on CPU (bundle / tokenizer / weight / catalog tests skip when absent)
	$(ENV) $(PYTHON) -m pytest -q -p no:cacheprovider tests/ -m "not provenance"

test-assets:     ## G6: a skip caused by a missing release asset fails (local_assets tests still skip)
	$(ENV) HABITUS_REQUIRE_ASSETS=1 $(PYTHON) -m pytest -q -p no:cacheprovider tests/ -m "not provenance"

gates:           ## G1-G6, G8-G11 (G7 is an optional GPU smoke: habitus gates --only G7 --gpu N)
	$(HABITUS) gates --jobs $(JOBS)

reproduce:       ## the tables from the shipped score files (about 35 s)
	$(HABITUS) reproduce --level tables --jobs $(JOBS)

reproduce-all:   ## tables + exact cache-only replay of every score file + operating tables (about 15 min)
	$(HABITUS) reproduce --level all --jobs $(JOBS)

release:         ## server only: write release/habitus-cache-v1 from configs/local.yaml sources
	$(ENV) $(PYTHON) scripts/make_release.py --hosts CADETS_E3,THEIA_E3,CLEARSCOPE_E3,optc_h501,ATLASV2_EDR --jobs $(JOBS)

status:          ## print the generated STATUS MATRIX block
	$(ENV) $(PYTHON) -m habitus.dm.status --render
