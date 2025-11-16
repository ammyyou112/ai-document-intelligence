"""
Script to fix corrupted tokenizer by deleting cache and forcing fresh download
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

