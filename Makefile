.PHONY: all docs doctests lint tests

all : tests lint docs doctests

tests :
	pytest -v --cov=mininf --cov-report=term-missing --cov-fail-under=100

lint:
	flake8
	mypy mininf

requirements.txt : requirements.in setup.py
	pip-compile -v

docs :
	rm -rf docs/_build
	sphinx-build -vnW . docs/_build

doctests :
	sphinx-build -b doctest . docs/_build
