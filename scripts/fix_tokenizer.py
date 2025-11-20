"""
Utility script to fix corrupted tokenizer cache

PURPOSE:
    When the HuggingFace tokenizer cache becomes corrupted (often due to
    incomplete downloads or file system issues), this script cleans the cache
    and forces a fresh download of the tokenizer files.

USAGE:
    Run this script when you encounter tokenizer-related errors:
    
        python scripts/fix_tokenizer.py
    
    Common error messages that indicate you need this:
        - "tokenizer.json is corrupted"
        - "Failed to load tokenizer"
        - "Tokenizer file not found"
    
    After running, the next time you load the model, it will download
    fresh tokenizer files from HuggingFace.

WHAT IT DOES:
    1. Finds the DeepSeek-OCR model cache directory
    2. Deletes all tokenizer-related files (*tokenizer*)
    3. Deletes the snapshots directory to force complete re-download
    4. Next model load will download fresh files

LOCATION:
    Cache directory: ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/

STATUS:
    Permanent utility - Keep this for troubleshooting tokenizer issues.

WARNING:
    This will delete your cached tokenizer files. You'll need internet
    connection to re-download them on next model load.
"""
import os
import shutil
from pathlib import Path

def fix_tokenizer():
    """Delete corrupted tokenizer files and force fresh download"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / "models--deepseek-ai--DeepSeek-OCR"
    
    deleted_count = 0
    
    if model_cache.exists():
        print(f"Found model cache at: {model_cache}")
        
        # Delete all tokenizer-related files
        for root, dirs, files in os.walk(model_cache):
            for file in files:
                if 'tokenizer' in file.lower():
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"  Deleted: {file}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  Failed to delete {file}: {e}")
    else:
        print(f"Model cache not found at: {model_cache}")
    
    # Also delete snapshots to force complete re-download
    if model_cache.exists():
        snapshots_dir = model_cache / "snapshots"
        if snapshots_dir.exists():
            try:
                shutil.rmtree(snapshots_dir)
                print(f"  Deleted snapshots directory")
                deleted_count += 1
            except Exception as e:
                print(f"  Failed to delete snapshots: {e}")
    
    print(f"\nDeleted {deleted_count} files/directories")
    print("Next time you load the tokenizer, it will download fresh from HuggingFace")

if __name__ == "__main__":
    fix_tokenizer()

