# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup

```bash
# Install dependencies for all packages
uv sync --dev
```

### Testing

```bash
# Run all tests
uv run pytest packages/

# Test specific package
uv run pytest packages/align-utils/
uv run pytest packages/align-track/

# Run tests
uv run pytest packages/

# Run a single test file
uv run pytest packages/align-utils/tests/test_parsing.py

# Run a specific test
uv run pytest packages/align-utils/tests/test_parsing.py::test_specific_function
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check . --fix

# Type checking
uv run mypy packages/
```

### Building

```bash
# Build a specific package
cd packages/align-utils
uv build
```

### Dependency Management

```bash
# Add dependency to specific package
cd packages/align-utils
uv add <package-name>

# Add dev dependency to workspace
uv add --dev <package-name>
```

## Code Style

Follow functional programming principles:

- Prefer pure functions without side effects
- Use immutable data structures where possible
- Avoid classes when functions suffice
- Use function composition and higher-order functions
- Minimize state mutation

## Architecture

This is a Python monorepo using `uv` workspaces containing utility packages for the align-system:

- **align-utils**: Core utilities for parsing align-system data

  - Pydantic models for input_output.json structures
  - YAML/JSON parsing utilities
  - CSV export functionality
  - KDMA (Key Decision Making Attributes) parsing

- **align-track**: Experiment tracking and organization utilities
  - Depends on align-utils
  - Provides experiment tracking capabilities

The packages are published independently to PyPI with semantic versioning handled by automated GitHub Actions workflows. Commit messages follow Angular convention (feat:, fix:, etc.) to trigger appropriate version bumps.

## Testing Approach

Tests are located in `packages/*/tests/` directories. The test suite uses pytest and includes unit tests for parsing, KDMA extraction, and experiment data handling. Run tests from the repository root using `uv run pytest`.
