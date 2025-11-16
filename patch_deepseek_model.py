"""
Patch script to fix LlamaFlashAttention2 import error in DeepSeek-OCR model
This patches the downloaded model code to work with newer transformers versions
"""
import os
import sys
import re
from pathlib import Path

def patch_model_code():
    """Patch the DeepSeek-OCR model code to fix import errors"""
    # Find the cached model directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / "models--deepseek-ai--DeepSeek-OCR"
    
    if not model_dir.exists():
        print("Model not downloaded yet. Will patch after first download.")
        return False
    
    # Find the actual model directory (with hash)
    model_dirs = list(model_dir.glob("snapshots/*"))
    if not model_dirs:
        print("Model snapshots not found.")
        return False
    
    # Use the latest snapshot
    latest_snapshot = max(model_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Found model at: {latest_snapshot}")
    
    # Find Python files that might have the import
    python_files = list(latest_snapshot.rglob("*.py"))
    
    patched = False
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix LlamaFlashAttention2 import
            # Replace: from transformers.models.llama.modeling_llama import LlamaFlashAttention2
            # With: Try to import, fallback to LlamaAttention if not available
            if 'LlamaFlashAttention2' in content and 'from transformers.models.llama.modeling_llama import' in content:
                # Create a more compatible import
                content = re.sub(
                    r'from transformers\.models\.llama\.modeling_llama import LlamaFlashAttention2',
                    '''try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention as LlamaFlashAttention2''',
                    content
                )
                patched = True
            
            # Also fix if it's imported differently
            if 'LlamaFlashAttention2' in content and 'import LlamaFlashAttention2' in content:
                content = re.sub(
                    r'import LlamaFlashAttention2',
                    '''try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention as LlamaFlashAttention2''',
                    content
                )
                patched = True
            
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Patched: {py_file.relative_to(latest_snapshot)}")
                
        except Exception as e:
            print(f"Error patching {py_file}: {e}")
            continue
    
    if patched:
        print("Model code patched successfully!")
        return True
    else:
        print("No patches needed or model code structure different.")
        return False

if __name__ == '__main__':
    patch_model_code()


