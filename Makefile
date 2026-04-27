# set default shell
SHELL := $(shell which bash)
GROUP_ID = $(shell id -g)
USER_ID = $(shell id -u)
FOLDER=$$(pwd)
# default shell options
.SHELLFLAGS = -c
PORT=8884
.SILENT: ;
default: help;   # default target

TESTS_DIR = tests
NOTEBOOK_DIR = notebooks
SRC_DIR = fairdream
DATA_DIR = data

IMAGE_NAME = fairdream:latest
DOCKER_NAME = fairdream
DOCKER_NAME_GPU = fairdream_gpu
DOCKER_RUN = docker run --rm -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c

build:
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} .
.PHONY: build

configure-pre-commit:
	$(DOCKER_RUN) "poetry run pre-commit install -f"
	make chown
	echo "Copy pre-commit configuration to .git/hooks"
	cp -a pre-commit/* .git/hooks/
.PHONY: configure-pre-commit

install: build ## First time: Build image, and install all the dependencies, including jupyter
	echo "Installing dependencies"
	docker run --rm     -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c 'poetry install'
	echo "Activating notebook extension"
	echo "Configuring pre-commit"
	make configure-pre-commit
	echo "Changing current folder rights"
.PHONY: install


up-notebook-extension: ## Activate useful notebook extensions
	$(DOCKER_RUN) "poetry run jupyter contrib nbextension install --sys-prefix --symlink"
	$(DOCKER_RUN) "poetry run jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --sys-prefix"
	$(DOCKER_RUN) "poetry run jupyter nbextension enable autosavetime/main --sys-prefix && poetry run jupyter nbextension enable tree-filter/index --sys-prefix && poetry run jupyter nbextension enable splitcell/splitcell --sys-prefix && poetry run jupyter nbextension enable toc2/main --sys-prefix && poetry run jupyter nbextension enable toggle_all_line_numbers/main --sys-prefix && poetry run jupyter nbextension enable cell_filter/cell_filter --sys-prefix && poetry run jupyter nbextension enable code_prettify/autopep8 --sys-prefix && poetry run jupyter nbextension enable jupyter-black-master/jupyter-black --sys-prefix"
.PHONY: up-notebook-extension

build-gpu:
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} . -f Dockerfile_gpu
.PHONY: build-gpu

install-gpu: build-gpu ## First time: Build image gpu, and install all the dependencies, including jupyter
	echo "Installing dependencies"
	docker run --gpus all --rm -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c 'poetry install'
	echo "Activating notebook extension"
	make up-notebook-extension
	echo "Configuring pre-commit"
	make configure-pre-commit
	echo "Changing current folder rights"
	sudo chmod -R 777 .cache
.PHONY: install-gpu

start-gpu: chown ## To get inside the gpu container (can launch "poetry shell" from inside or "poetry add <package>")
	echo "Starting container ${IMAGE_NAME}"
	docker run --name $(DOCKER_NAME_GPU) --gpus all --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start-gpu

start: chown ## To get inside the container (can launch "poetry shell" from inside or "poetry add <package>")
	echo "Starting container ${IMAGE_NAME}"
	docker run --name $(DOCKER_NAME) --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start

notebook: ## Start the Jupyter notebook (must be run from inside the container)
	poetry run jupyter notebook --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
	# &> /dev/null &
.PHONY: notebook

lab: ## Start the Jupyter notebook (must be run from inside the container)
	poetry run jupyter lab --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
.PHONY: notebook

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

ps: ## see docker running
	docker ps
.PHONY: ps

bash: ## Bash
	docker exec -it $(DOCKER_NAME) bash
.PHONY: bash

bash-gpu: ## Bash gpu
	docker exec -it $(DOCKER_NAME_GPU) bash
.PHONY: bash-gpu

tests: ## To run tests
	poetry run coverage run -m pytest -p no:cacheprovider tests/
.PHONY: tests

coverage: tests  ## To see tests coverage
	poetry run coverage report
	poetry run coverage html
.PHONY: coverage

docker_make: ## Running command inside docker
	@:$(call check_defined, cmd, command)
	echo "Running command inside docker, with home dir $$(echo ~): ${cmd}"
	docker run -it -v $$(pwd):/work/ -v $$(echo ~):/root/ -w /work ubuntu ${cmd}
.PHONY: docker_make

_ubuntu:
	docker pull ubuntu
.PHONY: _ubuntu

chown: _ubuntu ## Own your dir ! Don't let root get you !
	$(MAKE) docker_make cmd="chown -R ${USER_ID}:${GROUP_ID} ${DATA_DIR} ${SRC_DIR} ${TESTS_DIR} ${NOTEBOOK_DIR} .git"
#	$(MAKE) docker_make cmd="chmod -R 777 ${DATA_DIR} ${SRC_DIR} ${TESTS_DIR} ${NOTEBOOK_DIR}"
.PHONY: chown

reformat_file: ## pre commit
	echo "executing command inside docker..."
	$(DOCKER_RUN) "poetry run .git/hooks/pre-commit.py"
	echo "done"
.PHONY: reformat_file
