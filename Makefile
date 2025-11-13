#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ecg-svd
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ecg_svd
	isort --check --diff ecg_svd
	black --check ecg_svd

## Format source code with black
.PHONY: format
format:
	isort ecg_svd
	black ecg_svd


## Launch all the experiments
.PHONY: experiments
experiments:
	python ecg_svd/experiments/1_svd_single_segment.py 
	python ecg_svd/experiments/2_svd_multi_segment.py 
	python ecg_svd/experiments/3_fastica_multi_segment.py 
	python ecg_svd/experiments/4_svd_unfolded_tensor.py 
	python ecg_svd/experiments/5_tucker_tensor_segments.py 
	python ecg_svd/experiments/6_svd_single_signal.py 
	python ecg_svd/experiments/7_parafac_tensor_segments.py 
	python ecg_svd/experiments/8_tucker_tensor_gpu.py 
	python ecg_svd/experiments/9_parafac_tensor_gpu.py 
	python ecg_svd/src/viz.py


.PHONY: analytics
analytics:
	python ecg_svd/src/viz.py


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) ecg_svd/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
