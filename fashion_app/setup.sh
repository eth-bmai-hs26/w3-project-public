#!/bin/bash
# ============================================
# Python Installer Script for macOS
# ============================================

PYTHON_VERSION="3.12.9"
INSTALLER_URL="https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-macos11.pkg"
INSTALLER_PATH="/tmp/python-installer.pkg"

echo "=============================="
echo "  Python $PYTHON_VERSION Installer"
echo "=============================="

# Check if Python is already installed
echo ""
echo "Checking if Python is already installed..."
if command -v python3 &>/dev/null; then
    EXISTING_VERSION=$(python3 --version)
    echo "Python is already installed: $EXISTING_VERSION"
else
    echo "Python not found. Proceeding with installation..."

    # Download Python installer
    echo ""
    echo "Downloading Python $PYTHON_VERSION..."
    curl -o "$INSTALLER_PATH" "$INSTALLER_URL" --progress-bar
    echo "Download complete!"

    # Install Python
    echo ""
    echo "Installing Python (you may be prompted for your password)..."
    sudo installer -pkg "$INSTALLER_PATH" -target /

    # Cleanup
    rm -f "$INSTALLER_PATH"
    echo "Cleaned up installer."

    # Verify installation
    echo ""
    echo "Verifying installation..."
    if command -v python3 &>/dev/null; then
        echo "SUCCESS: $(python3 --version) is installed!"
        echo "Location: $(which python3)"
    else
        echo "Python installed. Please restart your terminal to use it."
    fi
fi

if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found"
else
    echo "Creating virtual enviorment..."
    python3 -m venv venv
    source venv/bin/activate    
    echo "Downloading libraries..."
    pip install -r requirements.txt
    deactivate
fi

