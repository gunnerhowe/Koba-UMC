# Contributing to UMC

Thanks for your interest in contributing to UMC.

## Quick Start

```bash
git clone https://github.com/umccodec/umc.git
cd umc
pip install -e ".[dev]"
pytest tests/ -v
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests for any new functionality
4. Run `pytest tests/ -v` and make sure all 405+ tests pass
5. Submit a pull request

## Code Style

- Follow existing patterns in the codebase
- Keep functions focused and under 50 lines where possible
- Use type hints for public API functions
- No external dependencies for core compress/decompress (numpy only)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only (skip benchmarks and large tests)
pytest tests/ -v -m "not benchmark and not slow"

# Run with coverage
pytest tests/ --cov=umc --cov-report=term-missing
```

## Adding a Compression Mode

1. Add the compressor function to `umc/codec/tiered.py`
2. Register the mode tag in `_compress_storage()` and `_decompress_storage()`
3. Add the mode name to `_ALL_STORAGE_MODES` in `umc/__init__.py`
4. Add round-trip tests in `tests/`
5. Update the mode table in README.md

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, UMC version, minimal reproducible example
- For compression bugs: include the data shape and dtype

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
