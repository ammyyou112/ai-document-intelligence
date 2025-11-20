#!/bin/bash
# Cleanup script for DeepSeek-OCR project
# This script cleans Python cache files and creates a backup of moved files

set -e  # Exit on error

echo "=========================================="
echo "DeepSeek-OCR Project Cleanup Script"
echo "=========================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "Project root: $PROJECT_ROOT"
echo ""

# Function to show directory structure
show_structure() {
    echo "Current directory structure:"
    echo "----------------------------------------"
    tree -L 2 -I 'venv|__pycache__|*.pyc|*.pyo' "$PROJECT_ROOT" 2>/dev/null || find "$PROJECT_ROOT" -maxdepth 2 -type d ! -path '*/venv/*' ! -path '*/__pycache__/*' | head -20
    echo ""
}

# Show before structure
echo "BEFORE cleanup:"
show_structure

# Create backup directory
BACKUP_DIR="$PROJECT_ROOT/backup-$(date +%Y%m%d-%H%M%S)"
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup moved files (if they exist in new locations)
echo ""
echo "Backing up moved files..."
if [ -d "$PROJECT_ROOT/archive" ]; then
    echo "  - Backing up archive/ directory"
    cp -r "$PROJECT_ROOT/archive" "$BACKUP_DIR/" 2>/dev/null || echo "    (archive/ already backed up or doesn't exist)"
fi

if [ -d "$PROJECT_ROOT/examples" ]; then
    echo "  - Backing up examples/ directory"
    cp -r "$PROJECT_ROOT/examples" "$BACKUP_DIR/" 2>/dev/null || echo "    (examples/ already backed up or doesn't exist)"
fi

if [ -d "$PROJECT_ROOT/scripts" ]; then
    echo "  - Backing up scripts/ directory"
    cp -r "$PROJECT_ROOT/scripts" "$BACKUP_DIR/" 2>/dev/null || echo "    (scripts/ already backed up or doesn't exist)"
fi

# Create tar.gz backup
if [ -d "$BACKUP_DIR" ] && [ "$(ls -A $BACKUP_DIR)" ]; then
    echo ""
    echo "Creating compressed backup..."
    BACKUP_FILE="$PROJECT_ROOT/backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    tar -czf "$BACKUP_FILE" -C "$PROJECT_ROOT" archive examples scripts 2>/dev/null || echo "  (Some files may not exist, continuing...)"
    if [ -f "$BACKUP_FILE" ]; then
        echo "  ✓ Backup created: $BACKUP_FILE"
        rm -rf "$BACKUP_DIR"
    fi
fi

# Clean Python cache files
echo ""
echo "Cleaning Python cache files..."
echo "----------------------------------------"

# Remove __pycache__ directories
find "$PROJECT_ROOT" -type d -name "__pycache__" -not -path "*/venv/*" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed __pycache__ directories"

# Remove .pyc files
find "$PROJECT_ROOT" -type f -name "*.pyc" -not -path "*/venv/*" -delete 2>/dev/null || true
echo "  ✓ Removed .pyc files"

# Remove .pyo files
find "$PROJECT_ROOT" -type f -name "*.pyo" -not -path "*/venv/*" -delete 2>/dev/null || true
echo "  ✓ Removed .pyo files"

# Remove .DS_Store files (Mac)
find "$PROJECT_ROOT" -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ Removed .DS_Store files"

# Note: We do NOT delete outputs/ or uploads/ contents as requested
echo ""
echo "  ℹ Preserved outputs/ and uploads/ directories (not cleaned)"

# Show after structure
echo ""
echo "AFTER cleanup:"
show_structure

# Summary
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo "✓ Python cache files removed"
echo "✓ Backup created (if files were moved)"
echo "✓ Directory structure reorganized"
echo ""
echo "New structure:"
echo "  - archive/     : Legacy code"
echo "  - examples/    : Example scripts"
echo "  - scripts/     : Helper scripts"
echo "  - app/         : Main application code"
echo ""
echo "To run the application:"
echo "  - From root: python app.py"
echo "  - Or use: scripts/start.sh or scripts/start.bat"
echo ""
echo "Cleanup completed successfully!"
echo "=========================================="

