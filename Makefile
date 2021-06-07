.PHONY: lint format test pylint mypy black isort

FOLDERS=rl_rpsr/ scripts/ tests/
FILES=$(shell find rl_rpsr scripts tests -name '*.py')

lint: pylint mypy

format: black isort

test:
	python -m unittest discover

pylint:
	pylint --reports n --disable similarities $(FILES)

mypy: 
	mypy --ignore-missing-imports $(FOLDERS)

black:
	black --skip-string-normalization --line-length 80 $(FOLDERS)

isort:
	isort $(FILES)
