.PHONY: docs doctests lint tests

tests :
	pytest -v --cov=minivb --cov-report=term-missing --cov-fail-under=100

lint:
	flake8
	mypy minivb

requirements.txt : requirements.in setup.py
	pip-compile -v

docs :
	rm -rf docs/_build
	sphinx-build -vn . docs/_build

doctests :
	sphinx-build -b doctest . docs/_build
