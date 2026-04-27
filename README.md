# How to reuse this template:
1. change folder name `template_package_name`
2. change `template_package_name` in `pyproject.toml` file
3. change variables in Makefile:
   - `IMAGE_NAME`
   - `DOCKER_NAME`
   - `DOCKER_NAME_GPU`
4. Change path in loggers config in the following files:
   - `template_package_name/utils/logger_config.py`
   - `template_package_name/utils/logging_dev.json`
   - `template_package_name/utils/logging.json`
   - `template_package_name/__init__.py`
5. Change `INSTALL_PYTHON` in pre-commit/pre-commit.py
6. Generate a beautiful readme with https://github.com/pedroermarinho/markdown-readme-generator like https://github.com/dreamquark-ai/nlp package
7. Create `develop` branch and configure it as default branch in Settings/Branches/Default branch


# How to install
cpu mode :
```shell
make install
```
gpu mode :
```shell
make install-gpu
```

# How to use
cpu mode:
launch container:
```shell
make start
```
gpu mode:
```shell
make start-gpu
```
launch jupyter notebook inside container:
```shell
make notebook
```
launch jupyter lab inside container:
```shell
make lab
```

# How to create new repository from scratch:
1. poetry new name_package
2. add to github repository
3. setup Dockerfile and Dockerfile_gpu properly to match de python version in pyproject.toml
4. Generate a beautiful readme with https://github.com/pedroermarinho/markdown-readme-generator like 

