# Sim2Real Train

A framework for simulation to reality training.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for Python package management and virtual environment creation. Ensure you have `uv` installed before proceeding.

## Installation

Clone the repository and create a virtual environment using uv:

```bash
git clone <repository-url>
cd sim2real-train
uv venv
uv sync
```

### Development Installation

To install with development and testing dependencies:

```bash
uv sync --extra dev
```

## Requirements

- Python >= 3.12.3

## Testing

Run tests using pytest:

```bash
pytest
```

## Project Structure

- `src/` - Main package source code
- `tests/` - Test suite
- `docs/` - Documentation

## License

See [LICENSE](LICENSE) file for details.
