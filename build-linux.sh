#!/bin/bash
# Orchestra Quick Build Script for Linux

set -e  # Exit on error

echo "╔════════════════════════════════════════╗"
echo "║  Orchestra v2.9 - Linux Build Script  ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install with: sudo apt install nodejs npm"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Install with: sudo apt install npm"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install with: sudo apt install python3 python3-pip"
    exit 1
fi

echo "✅ All prerequisites found"
echo ""

# Install Node dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Build frontend
echo "🔨 Building React frontend..."
npm run build

# Build Linux packages
echo "📦 Building Linux distribution packages..."
echo ""
echo "Select build type:"
echo "1) AppImage (recommended - universal)"
echo "2) DEB package (Ubuntu/Debian)"
echo "3) RPM package (Fedora/RHEL)"
echo "4) All formats"
read -p "Choice (1-4): " choice

case $choice in
    1)
        echo "Building AppImage..."
        npm run electron:build:appimage
        ;;
    2)
        echo "Building DEB package..."
        npm run electron:build:deb
        ;;
    3)
        echo "Building RPM package..."
        npm run electron:build:rpm
        ;;
    4)
        echo "Building all formats..."
        npm run electron:build:linux
        ;;
    *)
        echo "Invalid choice. Building AppImage by default..."
        npm run electron:build:appimage
        ;;
esac

echo ""
echo "╔════════════════════════════════════════╗"
echo "║         BUILD COMPLETE! 🎉             ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "📁 Output files are in: dist/"
ls -lh dist/

echo ""
echo "📝 Next steps:"
echo "1. Test your build:"
echo "   chmod +x dist/Orchestra-*.AppImage"
echo "   ./dist/Orchestra-*.AppImage"
echo ""
echo "2. Distribute:"
echo "   - Upload to GitHub Releases"
echo "   - Share AppImage (works on all Linux distros)"
echo "   - Provide DEB for Ubuntu/Debian users"
echo ""
echo "3. End users need:"
echo "   - Ollama installed: curl https://ollama.ai/install.sh | sh"
echo "   - Python deps: pip3 install -r requirements.txt --break-system-packages"
echo "   - At least one model: ollama pull granite3.3:8b"
echo ""
echo "Happy distributing! 🚀"
