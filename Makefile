# Makefile for Blob Tracker 4K

.PHONY: all build clean distclean

EXE_NAME = BlobTracker4K
SCRIPT_MAIN = blob_tracker_menu.py
SCRIPT_APP = blob_tracker_app.py
ICON = logo.ico

# PyInstaller flags
PYINSTALLER_FLAGS = --onefile --console --add-data "$(SCRIPT_APP);." --collect-all numpy --collect-all cv2 --icon=$(ICON) --name "$(EXE_NAME)"

all: build

# Build the executable
build:
	@echo "Cleaning old builds..."
	-rmdir /s /q build 2>nul
	-rmdir /s /q dist 2>nul
	-del $(EXE_NAME).spec 2>nul
	@echo "Running PyInstaller..."
	python -m PyInstaller $(PYINSTALLER_FLAGS) $(SCRIPT_MAIN)
	@echo "Build complete! Executable in dist/$(EXE_NAME).exe"

# Clean build files only
clean:
	@echo "Cleaning build files..."
	-rmdir /s /q build 2>nul
	-rmdir /s /q __pycache__ 2>nul
	-del *.spec 2>nul
	@echo "Clean done!"

# Clean everything including executable
distclean: clean
	@echo "Removing executable..."
	-rmdir /s /q dist 2>nul
	@echo "Distclean done!"

# Help
help:
	@echo "Available commands:"
	@echo "  make build      - Build executable"
	@echo "  make clean      - Remove build files"
	@echo "  make distclean  - Remove build files and executable"