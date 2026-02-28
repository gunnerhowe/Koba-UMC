.PHONY: test test-fast benchmark train-financial demo lint clean

test:
	python -m pytest tests/ -v --tb=short --cov=umc --cov-report=term-missing

test-fast:
	python -m pytest tests/ -v --tb=short -k "not benchmark and not integration and not large"

benchmark:
	python scripts/benchmark.py \
		--data-dir ./data/test/ \
		--codec-path ./checkpoints/financial_v1/ \
		--output ./results/benchmark.json

train-financial:
	python scripts/train.py \
		--symbols SPY,AAPL,MSFT,GOOGL,AMZN,BTC-USD,ETH-USD,GC=F,CL=F \
		--period 5y \
		--interval 1d \
		--config configs/financial_v1.yaml

demo:
	python scripts/demo.py --interactive

lint:
	python -m py_compile umc/__init__.py
	python -m pytest tests/ --collect-only -q

clean:
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
