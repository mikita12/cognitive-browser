# Contributing to Cognitive Browser

Thank you for your interest in contributing to Cognitive Browser! We welcome contributions, bug reports, and suggestions from the community.

Please take a moment to review this document to ensure a smooth collaboration process.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1).

---

## Development Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Playwright browsers: `playwright install chromium`
- Optional for local DeBERTa model: CUDA or Apple Silicon (MPS). CPU execution is also supported.

### 2. Clone and Install
```bash
git clone https://github.com/mikita12/cognitive-browser.git
cd cognitive-browser

python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development tooling
pip install -e ".[dev]"
```

---

## Quality and Verification Standards

All code contributions must meet our quality standards before being merged:

### 1. Run the Test Suite
The unit tests run completely offline without GPU or network dependencies:
```bash
pytest
```

### 2. Lint and Format (Ruff)
We use `ruff` for fast linting and formatting. Ensure there are no warnings:
```bash
# Check code style and common bugs
ruff check cognitive_browser tests

# Check code formatting
ruff format --check cognitive_browser tests

# Auto-format files if needed
ruff format cognitive_browser tests
```

### 3. Static Type Checking (Mypy)
We maintain strict type annotations on library code:
```bash
mypy cognitive_browser tests
```

---

## Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for our commit messages:

- `feat:` A new user-facing feature or enhancement.
- `fix:` A bug fix.
- `docs:` Documentation additions or updates.
- `test:` Adding or improving test cases.
- `refactor:` Code refactoring that neither adds a feature nor fixes a bug.
- `tooling:` CI, build system, or linter configuration updates.
- `perf:` Performance improvements.

**Example:**
```bash
git commit -m "feat(security): allow custom private network CIDR exemptions"
```

---

## Benchmarks & Testing Ethics

- **Never invent or guess numbers**: If you contribute benchmark comparisons or latency claims, explain the exact measurement setup and token counting rules.
- **Do not run aggressive benchmark loops on live third-party websites**: Use the offline fixture mode (`--use-fixtures`) in `benchmarks/run_15_cases_benchmark.py` whenever testing workflow logic.
- **Respect robots.txt and website Terms of Service** during testing and development.

---

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feat/my-new-feature`).
2. Make your changes in small, logical commits adhering to Conventional Commits.
3. Ensure all tests, linter, and type checks pass cleanly (`pytest && ruff check cognitive_browser tests && mypy cognitive_browser tests`).
4. Update relevant documentation (`README.md`, `docs/`, `CHANGELOG.md`) if behavior changed.
5. Open a Pull Request referencing any related issues using the provided PR template.
