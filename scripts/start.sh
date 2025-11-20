#!/bin/bash
# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

echo "Starting AI Document OCR Application..."
echo ""
python3 app.py


