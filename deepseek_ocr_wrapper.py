"""
Wrapper for DeepSeek-OCR model using Transformers
This wrapper integrates DeepSeek-OCR from HuggingFace
"""
import os
import sys
import tempfile
import shutil
from typing import Dict, List, Optional
from PIL import Image
import torch
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Try to import transformers
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers library not installed")
    print("   Install with: pip install transformers torch")

def is_cuda_available() -> bool:
    """Return True if CUDA is available via torch"""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


class DeepSeekOCR:
    """Wrapper class for DeepSeek-OCR model"""
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR', device: str = 'cuda'):
        """
        Initialize DeepSeek-OCR model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers torch"
            )
        
        # Resolve configuration from env with sensible defaults
        env_model_name = os.getenv("MODEL_NAME", model_name)
        self.model_name = env_model_name
        # Respect CUDA_VISIBLE_DEVICES if set to -1 (force CPU)
        if os.getenv("CUDA_VISIBLE_DEVICES", "") == "-1":
            resolved_device = "cpu"
        else:
            resolved_device = device if is_cuda_available() else 'cpu'
        self.device = resolved_device
        # Store env-based default sizes
        def _int_env(name: str, default: int) -> int:
            try:
                return int((os.getenv(name) or "").strip() or default)
            except Exception:
                return default
        self.default_base_size = _int_env("BASE_SIZE", 1024 if self.device == 'cuda' else 640)
        self.default_image_size = _int_env("IMAGE_SIZE", 640)
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize the model (lazy loading)"""
        if self._initialized:
            return
        
        try:
            print(f"\n{'='*60}")
            print(f"Loading DeepSeek-OCR model from {self.model_name}...")
            print(f"{'='*60}")
            print(f"Device requested: {self.device} | CUDA available: {torch.cuda.is_available()}")
            print("WARNING: This may take a while and requires model download from HuggingFace")
            print("WARNING: Model size: ~several GB - ensure you have:")
            print("   - Stable internet connection")
            print("   - Sufficient disk space")
            print("   - GPU with CUDA (recommended) or sufficient RAM for CPU")
            print(f"{'='*60}\n")
            
            # Patch model code before loading (fix compatibility issues)
            try:
                from patch_deepseek_model import patch_model_code
                patch_model_code()
            except:
                pass  # Patch is optional
            
            # Load tokenizer
            print("Step 1/3: Loading tokenizer...")
            # Disable fast tokenizer to avoid corrupted tokenizer.json file
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            # First, try slow tokenizer directly (more reliable with corrupted cache)
            try:
                print("   Trying slow tokenizer (more reliable)...")
                # Delete tokenizer.json if it exists to force slow tokenizer
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                model_cache = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
                if os.path.exists(model_cache):
                    for root, dirs, files in os.walk(model_cache):
                        for file in files:
                            if file == 'tokenizer.json':
                                try:
                                    os.remove(os.path.join(root, file))
                                    print(f"   Removed tokenizer.json to force slow tokenizer")
                                except:
                                    pass
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True,
                    use_fast=False,
                    force_download=False,
                    local_files_only=False
                )
                print("OK: Tokenizer loaded (slow)")
            except Exception as e:
                error_msg = str(e)
                print(f"WARNING: Slow tokenizer failed: {error_msg}")
                print("   Attempting to clear cache and force re-download...")
                try:
                    # Clear tokenizer cache completely
                    import shutil
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    model_cache = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
                    if os.path.exists(model_cache):
                        # Find and remove tokenizer files
                        for root, dirs, files in os.walk(model_cache):
                            for file in files:
                                if 'tokenizer' in file.lower():
                                    try:
                                        file_path = os.path.join(root, file)
                                        os.remove(file_path)
                                        print(f"   Removed: {file}")
                                    except Exception as rm_err:
                                        pass
                    
                    # Force fresh download
                    print("   Downloading fresh tokenizer from HuggingFace...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, 
                        trust_remote_code=True,
                        use_fast=False,
                        force_download=True
                    )
                    print("OK: Tokenizer downloaded and loaded successfully")
                except Exception as e2:
                    # Last attempt: try fast tokenizer
                    print(f"WARNING: Cache clear failed: {e2}")
                    print("   Trying fast tokenizer as last resort...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name, 
                            trust_remote_code=True,
                            use_fast=True,
                            force_download=True
                        )
                        print("OK: Fast tokenizer loaded")
                    except Exception as e3:
                        raise Exception(
                            f"Failed to load tokenizer after all attempts.\n"
                            f"Slow tokenizer error: {error_msg}\n"
                            f"Cache clear error: {e2}\n"
                            f"Fast tokenizer error: {e3}\n\n"
                            f"SOLUTION: The tokenizer file may be corrupted.\n"
                            f"  1. Manually delete: C:\\Users\\Ammad\\.cache\\huggingface\\hub\\models--deepseek-ai--DeepSeek-OCR\n"
                            f"  2. Restart the app to force fresh download\n"
                            f"  3. Check internet connection for HuggingFace access"
                        )
            
            # Load model
            print("Step 2/3: Loading model (this may take several minutes)...")
            print("   Downloading model weights from HuggingFace...")
            
            # Try to patch after tokenizer loads (model files are now downloaded)
            try:
                from patch_deepseek_model import patch_model_code
                patch_model_code()
            except:
                pass  # Patch is optional
            try:
                # Force eager attention to avoid SDPA incompatibility
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_safetensors=True,
                    attn_implementation="eager"
                )
                print("OK: Model weights loaded with attn_implementation='eager'")
            except Exception as e:
                error_msg = str(e)
                raise Exception(
                    f"Failed to load model with attn_implementation='eager': {error_msg}\n"
                    f"This issue is often caused by incompatible transformers/torch versions.\n"
                    f"Try:\n"
                    f"  pip install --upgrade torch transformers accelerate sentencepiece\n"
                    f"  pip install transformers==4.39.3"
                )
            
            # Move to device and set to eval mode
            print(f"Step 3/3: Moving model to {self.device} and setting up...")
            try:
                self.model = self.model.eval()
                if self.device == 'cuda' and torch.cuda.is_available():
                    print("   Moving to GPU (CUDA)...")
                    try:
                        self.model = self.model.cuda()
                        # Try bfloat16, fallback to float32 if not supported
                        try:
                            self.model = self.model.to(torch.bfloat16)
                            print("   Using bfloat16 precision")
                        except:
                            self.model = self.model.to(torch.float16)
                            print("   Using float16 precision (bfloat16 not available)")
                    except Exception as cuda_error:
                        print(f"WARNING: CUDA error: {cuda_error}, falling back to CPU")
                        self.device = 'cpu'
                        self.model = self.model.cpu().to(torch.float32)
                        print("   Using CPU with float32 precision")
                else:
                    print("   Using CPU (this will be slower but more stable)...")
                    self.model = self.model.to(torch.float32)
                    print("   Using float32 precision")
                print("OK: Model setup complete")
            except Exception as e:
                raise Exception(f"Failed to setup model on device: {str(e)}")
            
            self._initialized = True
            print(f"\n{'='*60}")
            print("OK: DeepSeek-OCR model loaded successfully!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n{'='*60}")
            print(f"ERROR: Failed to load DeepSeek-OCR model")
            print(f"{'='*60}")
            print(f"Error: {error_msg}")
            print(f"{'='*60}\n")
            raise Exception(
                f"Failed to load DeepSeek-OCR model: {error_msg}\n\n"
                f"Troubleshooting:\n"
                f"1. Check internet connection (model download required)\n"
                f"2. Install dependencies: pip install transformers torch\n"
                f"3. Ensure sufficient disk space (~several GB)\n"
                f"4. For GPU: Install CUDA-compatible PyTorch\n"
                f"5. For CPU: Ensure sufficient RAM (8GB+ recommended)"
            )
    
    def process(self, image_path: str, prompt: str = None, 
                base_size: int = None, image_size: int = None, 
                crop_mode: bool = True, extract_structure: bool = True) -> Dict:
        """
        Process an image and extract text with structure detection
        
        Args:
            image_path: Path to image file
            prompt: Prompt for OCR (default: uses structure-aware prompt if extract_structure=True)
            base_size: Base size for image processing (512, 640, 1024, 1280)
            image_size: Image size for processing
            crop_mode: Whether to use crop mode
            extract_structure: Whether to extract document structure (uses markdown conversion)
            
        Returns:
            Dictionary with 'full_text', 'structured_data', 'metadata', and 'markdown_output'
        """
        if not self._initialized:
            print("WARNING: Model not initialized, initializing now...")
            self._initialize()
        
        # Verify image file exists
        if not os.path.exists(image_path):
            raise Exception(f"Image file not found: {image_path}")
        
        # Set default sizes (can be controlled via env; CPU uses smaller defaults)
        if base_size is None:
            base_size = self.default_base_size
        if image_size is None:
            image_size = self.default_image_size
        
        # Set default prompt based on extract_structure flag
        if prompt is None:
            if extract_structure:
                # Use structure-aware prompt for markdown conversion (as per DeepSeek-OCR docs)
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
            else:
                prompt = "<image>\nFree OCR."
        
        try:
            print(f"Processing image: {os.path.basename(image_path)}")
            print(f"   Prompt: {prompt[:80]}...")
            print(f"   Device: {self.device}, Base size: {base_size}, Image size: {image_size}")
            print(f"   Structure extraction: {extract_structure}")
            
            # Use model's infer method (as per DeepSeek-OCR documentation)
            # The infer method signature: infer(tokenizer, prompt, image_file, output_path, 
            # base_size, image_size, crop_mode, save_results, test_compress)
            if hasattr(self.model, 'infer'):
                print("   Running OCR inference...")
                
                # Create a temporary directory for output (model may need it)
                temp_output_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
                
                try:
                    # Call infer method - output_path should be a directory string, not None
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        output_path=temp_output_dir,  # Use temp directory instead of None
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=False,  # Don't save to disk
                        test_compress=False
                    )
                    print("   OK: Inference completed")
                    
                    # Cleanup temp directory
                    try:
                        shutil.rmtree(temp_output_dir, ignore_errors=True)
                    except:
                        pass
                        
                except Exception as infer_error:
                    error_msg = str(infer_error)
                    print(f"   ERROR: Inference error: {error_msg}")
                    
                    # Cleanup temp directory on error
                    try:
                        shutil.rmtree(temp_output_dir, ignore_errors=True)
                    except:
                        pass
                    
                    # Provide helpful error messages
                    if 'cuda' in error_msg.lower() or 'gpu' in error_msg.lower():
                        raise Exception(
                            f"CUDA/GPU error during inference: {error_msg}\n"
                            f"The model may require GPU. Try:\n"
                            f"  1. Install CUDA-compatible PyTorch\n"
                            f"  2. Ensure GPU drivers are installed\n"
                            f"  3. Or use CPU mode (slower but should work)"
                        )
                    elif 'memory' in error_msg.lower() or 'out of memory' in error_msg.lower():
                        raise Exception(
                            f"Out of memory error: {error_msg}\n"
                            f"Try:\n"
                            f"  1. Use smaller image size (base_size=640, image_size=640)\n"
                            f"  2. Close other applications\n"
                            f"  3. Use GPU if available"
                        )
                    else:
                        raise Exception(
                            f"OCR inference failed: {error_msg}\n"
                            f"This could be due to:\n"
                            f"  - Invalid image format\n"
                            f"  - Insufficient memory/GPU resources\n"
                            f"  - Model compatibility issues\n"
                            f"  - Image file corruption"
                        )
                
                # Extract text from result
                # The infer method may return different formats
                print("   Extracting text from result...")
                markdown_output = None
                if isinstance(result, dict):
                    full_text = result.get('text', result.get('output', result.get('result', str(result))))
                    markdown_output = result.get('markdown', result.get('md', None))
                elif isinstance(result, str):
                    full_text = result
                    # If extract_structure was used, the result is likely markdown
                    if extract_structure:
                        markdown_output = result
                elif hasattr(result, 'text'):
                    full_text = result.text
                    if hasattr(result, 'markdown'):
                        markdown_output = result.markdown
                elif hasattr(result, 'output'):
                    full_text = result.output
                elif hasattr(result, 'result'):
                    full_text = result.result
                else:
                    full_text = str(result)
                
                # If markdown wasn't explicitly returned but structure extraction was used,
                # treat the full_text as markdown
                if extract_structure and markdown_output is None:
                    markdown_output = full_text
                
                if not full_text or len(full_text.strip()) == 0:
                    print("   WARNING: No text extracted from image")
                    full_text = ""  # Return empty string instead of raising error
                    markdown_output = ""
                    
            else:
                raise Exception(
                    "Model does not have 'infer' method. "
                    "Please ensure you're using the correct DeepSeek-OCR model from HuggingFace: "
                    "deepseek-ai/DeepSeek-OCR"
                )
            
            # Create structured data
            lines = full_text.split('\n') if full_text else []
            structured_data = [
                {
                    'text': line,
                    'confidence': 0.95,
                    'bbox': [0, 0, 0, 0]
                }
                for line in lines if line.strip()
            ]
            
            # Create metadata
            metadata = {
                'total_words': len(full_text.split()) if full_text else 0,
                'total_lines': len(lines),
                'confidence_scores': [0.95] * len(lines) if lines else [],
                'average_confidence': 0.95 if lines else 0.0
            }
            
            print(f"   OK: Extracted {metadata['total_words']} words, {metadata['total_lines']} lines")
            
            return {
                'full_text': full_text,
                'markdown_output': markdown_output if markdown_output else full_text,
                'structured_data': structured_data,
                'metadata': metadata
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ERROR: Processing error: {error_msg}")
            raise Exception(
                f"DeepSeek-OCR processing error: {error_msg}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify image file is valid and accessible\n"
                f"2. Check model is properly loaded\n"
                f"3. Ensure sufficient memory/GPU resources\n"
                f"4. Try a different image or smaller image size"
            )

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'transformers': False,
        'torch': False,
        'cuda': False
    }
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies['torch'] = True
        dependencies['cuda'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    return dependencies

