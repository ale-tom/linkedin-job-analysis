# Makefile for automating Python virtual environment and common tasks.
#
# This Makefile assumes:
#   - A specified Python version (default: python3.8)
#   - A virtual environment will be created in the ".venv" directory
#   - Dependencies are listed in "requirements.txt"
#   - The main application script is "src/main.py" (adjust as needed)
#   - Tests are executed via pytest
#
# To override the Python version, run for example:
#   make venv PYTHON_VERSION=python3.9

# Default python version and virtual environment directory
PYTHON_BIN ?= "/opt/homebrew/opt/python@3.13/libexec/bin/python"
VENV_DIR ?= .venv

# List of phony targets
.PHONY: help venv install clean run test

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  venv      - Create a virtual environment using $(PYTHON_VERSION)"
	@echo "  install   - Install dependencies from requirements.txt"
	@echo "  clean     - Remove the virtual environment directory"
	@echo "  run       - Run the application (assumes main.py in src/)"
	@echo "  test      - Run tests using pytest"
	@echo ""
	@echo "Example:"
	@echo "  make venv                  # creates the venv with default Python ($(PYTHON_VERSION))"
	@echo "  make venv PYTHON_BIN=/my/path/to/python/version  # creates the venv with the specified Python version"

# Create a virtual environment using the specified Python version.
venv:
	@echo "Creating virtual environment using $(PYTHON_VERSION)..."
	$(PYTHON_BIN) -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)."
	@echo "Activate it with:"
	@echo "  source $(VENV_DIR)/bin/activate   (on Unix/macOS)"

# Install project dependencies from requirements.txt.
install: venv
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV_DIR)/bin/activate && pip install -r requirements.txt && pip install -e .

# Remove the virtual environment folder.
clean:
	@echo "Removing virtual environment directory ($(VENV_DIR))..."
	rm -rf $(VENV_DIR)
	@echo "Clean complete."

# Run the main application.
run: install
	@echo "Running the application..."
	. $(VENV_DIR)/bin/activate && python src/main.py

# Run tests using pytest.
test: install
	@echo "Running tests..."
	. $(VENV_DIR)/bin/activate && pytest
