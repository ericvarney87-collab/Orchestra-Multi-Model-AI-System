#!/bin/bash
echo "Orchestra post-install setup..."
mkdir -p ~/.orchestra/users
chmod 755 ~/.orchestra

if ! command -v python3 &> /dev/null; then
    echo "WARNING: Python 3 required. Install: sudo apt install python3 python3-pip"
fi

if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama required. Install: curl https://ollama.ai/install.sh | sh"
fi

echo "Orchestra installation complete!"
