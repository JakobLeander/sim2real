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

## Robot Hardware
DRV8825 stepper driver
Stepper motor mecury sm24byg011-2s, 2 phase, 1.8 grad step
Model as velocity motor
in real robot convert velocity to steps
kv=0.4: how stiff motor is
forcerange = -0.23 0.23 the same as the nm on motor
ctrlrange -20 20: rad/second 

For real robot calculate drift by counting stepper pulses
for mujoco drift is in meters