---
summary: Documents how micromojograd verifies scalar autodiff behavior using Mojo TestSuite tests run through Pixi.
read_when:
  - Adding or updating engine behavior or derivative rules
  - Running, debugging, or extending Mojo unit tests
---

# Testing in Mojo

## What

The scalar autodiff engine is tested in `tests/test_engine.mojo` with Mojo's standard-library `TestSuite` runner and floating-point assertions. The test suite verifies forward values and hand-computed gradients for the supported graph operations.

## Why

The Python scratchgrad implementation could compare derivatives against PyTorch. The initial Mojo suite instead uses small expressions with explicit expected derivatives, so each assertion directly explains the calculus and tests the engine without adding a Python/PyTorch interop dependency.

## Key Files

- `tests/test_engine.mojo`: executable Mojo test module for `Value` arithmetic, backward propagation, and `relu()`.
- `pixi.toml`: defines the reproducible `pixi run test` command.
- `micromojograd/engine.mojo`: implementation under test.

## How It Works

Current Mojo no longer provides a `mojo test` command. A test module imports `TestSuite` and provides an executable test runner:

```mojo
from std.testing import assert_almost_equal, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Each function whose name begins with `test_` is discovered automatically. Because autodiff uses `Float64`, tests use `assert_almost_equal(..., atol=1e-12)` for forward outputs and gradients.

Run all current engine tests with:

```sh
pixi run test
```

The Pixi task expands to:

```sh
mojo run -I . tests/test_engine.mojo
```

`-I .` adds the project root to Mojo's import search path. This is required because a file compiled from `tests/` does not otherwise locate the sibling `micromojograd/` package.

The initial suite covers:

- Accumulation through a shared intermediate (`product + product`).
- Scalar-left and scalar-right operator overloads.
- Power derivatives.
- Positive and negative ReLU branches.
- A compound expression combining arithmetic and ReLU.

## Gotchas

- Run tests with `pixi run test`; do not use the removed `mojo test` command.
- Keep `-I .` in the test task while tests live outside the source package directory.
- Prefer hand-computed expectations for small learning steps; add reference-framework comparisons only when they clarify rather than obscure behavior.
