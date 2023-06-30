.PHONY: lint tests

tests :
	pytest -v --cov=minivb --cov-report=term-missing --cov-fail-under=100

lint:
	flake8
	mypy minivb
