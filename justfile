set fallback

[private]
help:
  just --list

# installs pre-commit hooks (and pre-commit if it is not installed)
install-hooks: install-pre-commit
  #!/usr/bin/env bash
  set -euo pipefail
  if [ ! -f .git/hooks/pre-commit ]; then
    pre-commit install
  fi

[private]
install-pre-commit:
  #!/usr/bin/env bash
  set -euo pipefail
  # Check if command pre-commit is available. If yes - exists
  if command -v pre-commit &> /dev/null; then
    exit 0
  fi
  # Check if pipx exists. If it does not, asks
  if ! command -v pipx &> /dev/null; then
    echo "pipx is not installed. It is recommended to install pre-commit using pipx rather than pip. Please install pipx using `pip install pipx` and try again."
    exit 1
  fi
  pipx install pre-commit

setup: install-hooks

run:
  #!/usr/bin/env bash
  set -euo pipefail
  poetry run python -m run_server