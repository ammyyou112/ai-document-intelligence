# Scripts Directory

This directory contains utility scripts and helpers for the DeepSeek-OCR application.

## Scripts Overview

### Application Scripts

#### `run_app.py`
**Purpose**: Python launcher for the Flask application  
**Usage**: `python scripts/run_app.py`  
**Description**: Starts the Flask web server with proper path configuration.

#### `start.sh` (Linux/Mac)
**Purpose**: Bash script to start the application  
**Usage**: `./scripts/start.sh`  
**Description**: Changes to project root and starts the Flask app.

#### `start.bat` (Windows)
**Purpose**: Batch script to start the application  
**Usage**: `scripts\start.bat`  
**Description**: Changes to project root and starts the Flask app.

#### `setup_check.py`
**Purpose**: Dependency and environment checker  
**Usage**: `python scripts/setup_check.py`  
**Description**: Verifies system dependencies, CUDA availability, and configuration.

### Utility Scripts

#### `fix_tokenizer.py`
**Purpose**: Fix corrupted tokenizer cache  
**Usage**: `python scripts/fix_tokenizer.py`  
**Description**: 
- Deletes corrupted tokenizer files from HuggingFace cache
- Forces fresh download on next model load
- Use when encountering tokenizer-related errors

**When to use:**
- "tokenizer.json is corrupted" errors
- "Failed to load tokenizer" errors
- Tokenizer file not found errors

**What it does:**
1. Finds DeepSeek-OCR model cache
2. Deletes all tokenizer files
3. Deletes snapshots to force re-download
4. Next model load downloads fresh files

#### `patch_deepseek_model.py`
**Purpose**: Compatibility patch for DeepSeek-OCR model  
**Usage**: Automatically called by `deepseek_ocr_wrapper.py` (or run manually)  
**Description**:
- Patches model code to fix `LlamaFlashAttention2` import errors
- Works around compatibility issues with newer transformers versions
- Modifies cached model files in HuggingFace cache

**Status**: 
- ✅ **Automatically used** - Called during model initialization
- ✅ **Permanent utility** - Required for compatibility
- ✅ **Safe** - Only modifies local cache, not source code

**What it does:**
1. Finds cached model in HuggingFace cache
2. Patches Python files to add try/except for LlamaFlashAttention2 import
3. Falls back to LlamaAttention if LlamaFlashAttention2 unavailable

## Running Scripts

All scripts should be run from the project root directory:

```bash
# From project root
python scripts/fix_tokenizer.py
python scripts/setup_check.py
python scripts/run_app.py
```

Or use the shell scripts:

```bash
# Linux/Mac
./scripts/start.sh

# Windows
scripts\start.bat
```

## Script Dependencies

Most scripts add the project root to `sys.path` automatically, so imports work correctly.

## Troubleshooting

If a script fails to import modules:
1. Ensure you're running from project root
2. Check that virtual environment is activated
3. Verify dependencies are installed: `pip install -r requirements.txt`

