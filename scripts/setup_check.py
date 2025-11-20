"""
Setup check script to verify all dependencies are installed correctly
"""
import sys
import platform
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_cuda():
    """Check CUDA availability and versions via torch if present"""
    try:
        import torch
    except Exception as e:
        print("ℹ️  Skipping CUDA check: torch is not installed")
        return False
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        try:
            # torch.version.cuda gives the CUDA toolkit version used to build torch
            torch_cuda_ver = getattr(torch.version, "cuda", None)
        except Exception:
            torch_cuda_ver = None
        print(f"✅ CUDA available: {device_count} device(s) detected")
        print(f"   GPU[0]: {name}")
        if torch_cuda_ver:
            print(f"   PyTorch CUDA build version: {torch_cuda_ver}")
        else:
            print("   PyTorch CUDA build version: Unknown")
        return True
    else:
        print("⚠️  CUDA not available (torch.cuda.is_available() == False)")
        print("   The model can run on CPU but will be much slower.")
        return False

def check_flash_attention():
    """Check FlashAttention presence and compatibility"""
    try:
        import flash_attn  # noqa: F401
        print("✅ flash-attn is installed")
        return True
    except Exception as e:
        print(f"⚠️  flash-attn not available: {e}")
        print("   Tip: pip install flash-attn==2.7.3 --no-build-isolation")
        return False

def check_transformers_and_torch():
    """Check transformers and torch versions and basic GPU dtype support"""
    ok = True
    try:
        import transformers
        print(f"✅ transformers {transformers.__version__}")
    except Exception as e:
        print(f"❌ transformers missing: {e}")
        ok = False
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
        # Check BF16/FP16 availability on GPU if CUDA exists
        if torch.cuda.is_available():
            bf16_ok = torch.cuda.is_bf16_supported()
            print(f"   GPU bfloat16 support: {'Yes' if bf16_ok else 'No'}")
    except Exception as e:
        print(f"❌ torch missing/broken: {e}")
        ok = False
    return ok

def check_vllm_optional():
    """Check vLLM presence (optional)"""
    try:
        import vllm  # noqa: F401
        print("ℹ️  vLLM installed (optional) – good for high-throughput inference")
        return True
    except Exception:
        print("ℹ️  vLLM not installed (optional)")
        return False

def check_poppler():
    """Check if Poppler is installed"""
    try:
        from pdf2image import convert_from_path
        print("✅ Poppler is installed (pdf2image can import)")
        return True
    except Exception as e:
        print(f"⚠️  Poppler: {str(e)}")
        print("   Note: Required for PDF processing. Image files work without it.")
        return False

def check_deepseek_api():
    """Check if DeepSeek API key is configured"""
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    if api_key:
        print("✅ DeepSeek API key is configured")
        return True
    else:
        print("⚠️  DeepSeek API key is not configured")
        print("   Note: Set DEEPSEEK_API_KEY in .env file or environment")
        print("   Get your API key from: https://platform.deepseek.com/")
        return False

def check_deepseek_local():
    """Check if DeepSeek OCR local model is available"""
    try:
        from deepseek_ocr import DeepSeekOCR
        print("✅ DeepSeek OCR local model is available")
        try:
            model = DeepSeekOCR()
            print("✅ DeepSeek OCR local model can initialize")
            return True
        except Exception as e:
            print(f"⚠️  DeepSeek OCR local model initialization: {str(e)}")
            return False
    except ImportError:
        print("ℹ️  DeepSeek OCR local model is not installed")
        print("   Note: Optional if using API. Install from: https://github.com/deepseek-ai/DeepSeek-OCR")
        return False

def main():
    """Main setup check"""
    print("=" * 60)
    print("DeepSeek OCR - Setup Check")
    print("=" * 60)
    print(f"\nPlatform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}\n")
    
    print("Checking Python version...")
    python_ok = check_python_version()
    print()
    
    print("Checking core Python packages...")
    packages = [
        ('Flask', 'flask'),
        ('Flask-CORS', 'flask_cors'),
        ('OpenCV', 'cv2'),
        ('Pillow', 'PIL'),
        ('NumPy', 'numpy'),
        ('Werkzeug', 'werkzeug'),
        ('pdf2image', 'pdf2image'),
        ('Requests', 'requests'),
        ('python-dotenv', 'dotenv'),
    ]
    
    packages_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            packages_ok = False
    print()
    
    print("Checking ML stack (torch/transformers/flash-attn/vLLM)...")
    ml_ok = check_transformers_and_torch()
    cuda_ok = check_cuda()
    fa_ok = check_flash_attention()
    vllm_ok = check_vllm_optional()
    print()

    print("Checking DeepSeek OCR...")
    local_ok = check_deepseek_local()
    api_ok = check_deepseek_api()
    print()
    
    print("Checking PDF processing...")
    poppler_ok = check_poppler()
    print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    if python_ok and packages_ok:
        if local_ok:
            print("✅ Core requirements are met!")
            print("✅ DeepSeek OCR local model is available (RECOMMENDED)")
        elif api_ok:
            print("⚠️  Core requirements are met, but local model is not available")
            print("⚠️  DeepSeek OCR API is configured (fallback only)")
            print("⚠️  RECOMMENDED: Install DeepSeek-OCR local model from GitHub")
        else:
            print("❌ DeepSeek OCR is not configured")
            print("❌ Please install DeepSeek-OCR local model from GitHub")
            print()
            print("Installation steps:")
            print("  1. git clone https://github.com/deepseek-ai/DeepSeek-OCR.git")
            print("  2. cd DeepSeek-OCR")
            print("  3. Follow installation instructions in the repository")
            print("  4. Add to Python path or install as package")
        
        if poppler_ok:
            print("✅ PDF processing is available")
        else:
            print("⚠️  PDF processing is not available (image files still work)")
        
        if local_ok or api_ok:
            print("\n🚀 You can start the application with: python app.py")
    else:
        print("❌ Some requirements are missing")
        print("\nInstall missing packages with: pip install -r requirements.txt")
        if not local_ok:
            print("\nREQUIRED: Install DeepSeek-OCR local model:")
            print("  1. git clone https://github.com/deepseek-ai/DeepSeek-OCR.git")
            print("  2. cd DeepSeek-OCR")
            print("  3. Follow installation instructions")
            print("  4. Add to Python path or install as package")
        if not poppler_ok:
            print("\nInstall Poppler for PDF support (optional)")
        if not ml_ok:
            print("\nInstall ML stack:")
            print("  pip install transformers torch")
        if not fa_ok:
            print("\nOptional performance package:")
            print("  pip install flash-attn==2.7.3 --no-build-isolation")
        if not cuda_ok:
            print("\nCUDA not detected. To use GPU, install a CUDA-enabled PyTorch build and NVIDIA drivers.")
    
    print("=" * 60)

if __name__ == '__main__':
    main()

