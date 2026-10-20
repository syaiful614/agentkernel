.PHONY: install lint type test redteam all
install:
	pip install -e ".[dev]"
lint:
	ruff check . && ruff format --check .
type:
	mypy src
test:
	pytest -q --cov=agentkernel --cov-report=term-missing
redteam:
	python -m agentkernel redteam
all: lint type test redteam
