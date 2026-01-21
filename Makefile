.PHONY: install frontend-install frontend-build start lint clean unified-backtest discover-semantic-regimes bundle-study

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PIP_FLAGS ?= --no-cache-dir
PIP_BREAK_FLAG ?= --break-system-packages
LINT_PATHS ?= scripts/trading scripts/research scripts/tools scripts/trading_service.py

CONFIG ?= configs/research/semantic_pilot.json

install:
	$(PIP) install $(PIP_FLAGS) -r requirements.txt || \
		$(PIP) install $(PIP_FLAGS) $(PIP_BREAK_FLAG) -r requirements.txt

frontend-install:
	cd apps/frontend && npm install

frontend-build:
	cd apps/frontend && npm run build

start:
	$(PYTHON) scripts/trading_service.py

lint:
	$(PYTHON) -m compileall $(LINT_PATHS)

clean:
	rm -rf __pycache__ */**/__pycache__ apps/frontend/node_modules apps/frontend/dist

unified-backtest:
	@$(PYTHON) scripts/run_unified_backtest.py $(ARGS)

discover-semantic-regimes:
	@$(PYTHON) scripts/research/semantic_regime_discovery.py --config $(CONFIG) $(DISCOVER_ARGS)

bundle-study:
	@$(PYTHON) scripts/research/build_bundle_activation_tape.py --gates docs/evidence/roc_history --bundle-config config/bundle_strategy.yaml --output output/strand_bundles/bundle_activation_tape.jsonl
	@$(PYTHON) scripts/tools/bundle_outcome_study.py --gates docs/evidence/roc_history --bundle-config config/bundle_strategy.yaml --output docs/evidence/bundle_outcomes.json
	@$(PYTHON) scripts/research/bundle_activation_simulator.py --gates docs/evidence/roc_history --bundle-config config/bundle_strategy.yaml --output output/strand_bundles/bundle_trades.csv
