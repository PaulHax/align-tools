# align-tools

A monorepo containing utility packages and tools to support the align-system codebase. Built with `uv` workspaces for dependency management and automated semantic versioning for PyPI publishing.

## Overview

The `align-tools` repository is organized as a monorepo using `uv`'s workspace feature.


## Packages

### align-utils
Utilities for parsing and processing align-system experiment data.
- Pydantic models for align-system input_output.json data structures
- YAML/JSON parsing utilities
- Data export utilities (CSV)

## Quick Start

### Prerequisites
- Python 3.9+
- uv (Python package manager)

### Installation

Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and set up the development environment:
```bash
git clone https://github.com/paulhax/align-tools.git
cd align-tools
uv sync --all-extras --dev
```

### Using Published Packages

Individual packages are published to PyPI and can be installed directly:
```bash
pip install align-utils
```

## Development

### Running Tests

Test all packages:
```bash
uv run pytest packages/
```

Test a specific package:
```bash
uv run pytest packages/align-utils/
```

### Code Quality

Format code:
```bash
uv run ruff format .
```

Lint code:
```bash
uv run ruff check . --fix
```

Type checking:
```bash
uv run mypy packages/
```

### Adding Dependencies

Add a dependency to a specific package:
```bash
cd packages/align-utils
uv add requests
```

Add a development dependency to the workspace:
```bash
uv add --dev pytest-mock
```

### Building Packages

Build a specific package:
```bash
cd packages/align-utils
uv build
```

## Release Process

This repository uses semantic versioning and automated releases:

1. Commit messages follow the Angular convention:
   - `feat:` - New features (minor version bump)
   - `fix:` - Bug fixes (patch version bump)
   - `BREAKING CHANGE:` - Breaking changes (major version bump)

2. When changes are merged to `main`, the release workflow:
   - Analyzes commits since the last release
   - Bumps version according to semantic versioning
   - Updates changelog
   - Creates git tag
   - Builds and publishes to PyPI
   - Creates GitHub release

## Contributing

1. Create a feature branch from `main`
2. Make your changes following the commit message convention
3. Ensure tests pass and code is formatted
4. Create a pull request
5. Once merged, automated release will handle versioning and publishing

## Links

- [Repository](https://github.com/paulhax/align-tools)
- [Issues](https://github.com/paulhax/align-tools/issues)
- [PyPI - align-utils](https://pypi.org/project/align-utils/)