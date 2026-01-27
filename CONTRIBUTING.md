# 🚀 Contributing to SpikingJelly

SpikingJelly is an open-source framework for spiking neural networks, and we welcome contributions from the community. This document outlines the recommended ways to contribute and the standards we expect contributors to follow.

## Ways to Contribute

You can contribute to SpikingJelly in several ways:

- [reporting bugs or requesting features](#reporting-issues)
- [submitting code changes or new features](#contributing-code)
- [improving or translating documentation and tutorials](#contributing-docs-and-tutorials)

Each contribution type is described in more detail below.

## Reporting Issues

If you encounter a bug or would like to request a new feature, please open an issue on GitHub.

Before creating a new issue:

1. Search the existing issues to **avoid duplicates**.
2. Ensure that the issue is **reproducible** with a supported version of SpikingJelly.

When creating an issue, please include:

- SpikingJelly version
- Python version
- Operating system
- Relevant error messages or logs
- Minimal code snippet to reproduce the issue (if applicable)

Please use appropriate labels (e.g., `bug`, `feature request`) to help maintainers categorize the issue.

## Contributing Code

### Overall Workflow

1. Fork the repository and clone it locally.
2. Create a new branch for your work: `git checkout -b <descriptive-branch-name>` .
3. Make your changes following the coding and style guidelines.
4. Add or update tests if your changes affect functionality.
5. Submit a Pull Request (PR) with a clear and concise description of the changes.

### Development Environment

We highly suggest contributors to use [uv](https://docs.astral.sh/uv/) for environment management.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) to your machine.
2. Deactivate other Python environments.
3. Create a new virtual environment at `.venv` inside the project directory: `uv venv --python 3.11`. `python>=3.11` is recommended.
4. Install PyTorch from wheels following the [official instructions](https://pytorch.org/get-started/previous-versions/). We suggest using [`uv pip install`](https://docs.astral.sh/uv/pip/packages/) instead of `pip install`.
5. Install SpikingJelly according to our `pyproject.toml`: `uv pip install --editable . --group dev` .

    - This will install SpikingJelly in **editable mode**, which means you can make changes to the code and they will be reflected in your environment immediately.
    - All dependencies will be installed automatically.
    - The argument `--group dev` installs all development and doc tools (e.g. sphinx).
    - To install optional dependencies, use the syntax `uv pip install --editable ".[triton]"`. See the `project.optional-dependencies` table in `pyproject.toml` for a list of optional dependencies.

6. Manually install other development tools: `uv pip install pyclean pytest ...`
7. Activate the virtual environment: `source .venv/bin/activate` .

You may want to directly sync your virtual environment through `uv sync --extra ...` . However, we discourage this because it locks your PyTorch version.

### Coding Standards

New code should be readable, **well-documented**, and maintainable. Public APIs should include clear docstrings. Please follow the docstring style of other modules in SpikingJelly.

Please format your code before committing. 

1. Make sure that [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed.
2. Run `uv format` at the project root.

### Pull Request Guidelines

- Keep PRs focused on a single issue or feature.
- Avoid unrelated refactoring unless necessary.
- Clearly explain what was changed and why.
- Reference related issues when applicable.

All PRs will be reviewed by maintainers. Feedback and revision requests are a normal part of the review process.

## Contributing Docs and Tutorials

We welcome improvements to [documentation and tutorials](https://spikingjelly.readthedocs.io/), including:

- Fixing inaccuracies or typos
- Improving clarity or examples
- Translating documentation between English and Chinese

### Workflow

1. Fork the repository and clone it locally.
2. Create a new branch for your work: `git checkout -b <descriptive-branch-name>` .
3. Make your changes.

    - Docs contents are placed in the docstrings of the corresponding modules. Docs are organized according to `docs/source/APIs`.
    - Tutorials are located in `docs/source/tutorials`.
    - Assets (e.g., images) should be placed in `docs/source/_static`.
    - Make the style and structure of your changes consistent with the rest of the docs!

4. Submit a Pull Request (PR) with a clear and concise description of the changes.

### Build the Docs Locally

You may want to build the docs locally to preview your changes. Follow these steps:

1. Prepare a Python virtual environment (through conda, uv, etc.). Install the required dependencies listed in `docs/requirements.txt`.
2. Go into the `docs` directory, and build the docs:

    ```bash
    cd docs
    make html
    ```

3. A new directory `docs/build` will then be available. Open `docs/build/html/index.html` in your browser to preview the docs.
