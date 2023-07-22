.PHONY: all docs doctests lint package publish tests

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

clean :
	rm -rf docs/_build docs/.jupyter_cache .coverage* .pytest_cache

package :
	python setup.py sdist
	twine check dist/*.tar.gz

publish : package
	twine upload --skip-existing --username=__token__ dist/*.tar.gz
