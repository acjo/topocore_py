.PHONY: help
CONDA_ENV:=requirements/environment.yml
ENV_NAME:=thinkpose

## help: Prints this list of commands.
## development-environment: creates development environment
## install: installs the python package in editable mode
## install-with-doc-dependencies: installs the python package in editable mode with the dependencies to build documentation
## install-with-tests: installs the python package in editable mode with the dependencies to test the package.

development-environment:
	conda env create --file ${CONDA_ENV}

install:
	uv pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

install-with-doc-dependencies:
	pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .[docs]

install-with-tests:
	pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .[test]


help:
	@echo "\nUsage: \n"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/-/'
