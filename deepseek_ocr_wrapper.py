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
                # Try importing from scripts directory first, then root (for backward compatibility)
                try:
                    from scripts.patch_deepseek_model import patch_model_code
                except ImportError:
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
                # Try importing from scripts directory first, then root (for backward compatibility)
                try:
                    from scripts.patch_deepseek_model import patch_model_code
                except ImportError:
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
                # Note: Adding trailing space as shown in example
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = "<image>\nFree OCR. "
        
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
                original_result = None  # Initialize for bbox extraction
                saved_json_data = None  # Initialize for JSON bbox data
                
                try:
                    # Call infer method - output_path should be a directory string, not None
                    # Note: Even with save_results=False, the model might save files, so we check for them
                    print(f"   Calling model.infer() with output_path: {temp_output_dir}")
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        output_path=temp_output_dir,  # Use temp directory instead of None
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=True,  # Set to True to ensure we can read results from files if needed
                        test_compress=False
                    )
                    print(f"   DEBUG: Inference result type: {type(result)}")
                    print(f"   DEBUG: Inference result value: {str(result)[:200] if result is not None else 'None'}")
                    
                    # 🔍 DEBUG: DeepSeek Model Output Structure
                    print("\n" + "="*60)
                    print("🔍 DEBUG: DeepSeek Model Output Structure")
                    print("="*60)
                    print(f"Type: {type(result)}")
                    
                    if isinstance(result, dict):
                        print(f"Keys: {list(result.keys())}")
                        # Show first item structure
                        for key, value in list(result.items())[:5]:
                            value_str = str(value)
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                            print(f"  {key}: {type(value)} - {value_str}")
                            
                    elif isinstance(result, list) and len(result) > 0:
                        print(f"List length: {len(result)}")
                        print(f"First item type: {type(result[0])}")
                        if isinstance(result[0], dict):
                            print(f"First item keys: {list(result[0].keys())}")
                            for key, value in list(result[0].items())[:5]:
                                value_str = str(value)
                                if len(value_str) > 200:
                                    value_str = value_str[:200] + "..."
                                print(f"  {key}: {type(value)} - {value_str}")
                        else:
                            print(f"First item: {result[0]}")
                    elif isinstance(result, str):
                        print(f"String length: {len(result)}")
                        print(f"First 500 chars: {result[:500]}")
                    
                    print("="*60 + "\n")
                    
                    # Check if output files were created (even if result is None or "None")
                    output_files = []
                    if os.path.exists(temp_output_dir):
                        print(f"   DEBUG: Checking output directory: {temp_output_dir}")
                        # List all files in the directory
                        all_files = []
                        for root, dirs, files in os.walk(temp_output_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                all_files.append(file_path)
                        print(f"   DEBUG: Found {len(all_files)} total files in output directory")
                        
                        # Check for common output file patterns
                        for file_path in all_files:
                            file_ext = os.path.splitext(file_path)[1].lower()
                            file_name = os.path.basename(file_path).lower()
                            # Accept text files, markdown, json, or any file that might contain output
                            if (file_ext in ('.txt', '.md', '.json', '.html', '.xml') or 
                                'output' in file_name or 'result' in file_name or 
                                'ocr' in file_name or 'text' in file_name):
                                output_files.append(file_path)
                    
                    if output_files:
                        print(f"   DEBUG: Found {len(output_files)} potential output files")
                        # 🔍 DEBUG: Check for JSON files that might contain bbox data
                        json_files = [f for f in output_files if f.endswith('.json')]
                        if json_files:
                            print(f"   🔍 DEBUG: Found {len(json_files)} JSON files - checking for bbox data...")
                            for json_file in json_files:
                                try:
                                    import json
                                    with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                                        json_data = json.load(f)
                                        print(f"   🔍 DEBUG: JSON file structure: {type(json_data)}")
                                        if isinstance(json_data, dict):
                                            print(f"   🔍 DEBUG: JSON keys: {list(json_data.keys())}")
                                            # Look for bbox-related keys
                                            bbox_keys = [k for k in json_data.keys() if 'bbox' in k.lower() or 'box' in k.lower() or 'coord' in k.lower()]
                                            if bbox_keys:
                                                print(f"   🔍 DEBUG: Found bbox-related keys: {bbox_keys}")
                                        elif isinstance(json_data, list) and len(json_data) > 0:
                                            print(f"   🔍 DEBUG: JSON is list with {len(json_data)} items")
                                            if isinstance(json_data[0], dict):
                                                print(f"   🔍 DEBUG: First item keys: {list(json_data[0].keys())}")
                                                bbox_keys = [k for k in json_data[0].keys() if 'bbox' in k.lower() or 'box' in k.lower() or 'coord' in k.lower()]
                                                if bbox_keys:
                                                    print(f"   🔍 DEBUG: Found bbox-related keys: {bbox_keys}")
                                except Exception as json_err:
                                    print(f"   DEBUG: Could not parse JSON file {json_file}: {json_err}")
                        
                        # Try to read text from output files (prioritize .md and .txt)
                        output_files.sort(key=lambda x: (x.endswith('.md'), x.endswith('.txt'), x.endswith('.json')))
                        for output_file in output_files:
                            try:
                                print(f"   DEBUG: Attempting to read: {output_file}")
                                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    file_content = f.read()
                                    if file_content and file_content.strip() and file_content.strip().lower() != 'none':
                                        print(f"   DEBUG: Successfully read {len(file_content)} chars from {output_file}")
                                        # Use file content if result is None, empty, or "None"
                                        if (result is None or 
                                            (isinstance(result, str) and (result.strip() == '' or result.strip().lower() == 'none'))):
                                            result = file_content
                                            print(f"   DEBUG: Using content from file as result")
                                            break
                                        elif isinstance(result, str) and len(file_content) > len(result):
                                            # File content is longer, prefer it
                                            result = file_content
                                            print(f"   DEBUG: File content is longer, using it")
                                            break
                            except Exception as read_err:
                                print(f"   DEBUG: Could not read {output_file}: {read_err}")
                    else:
                        print(f"   DEBUG: No output files found in {temp_output_dir}")
                        if os.path.exists(temp_output_dir):
                            print(f"   DEBUG: Directory contents: {os.listdir(temp_output_dir)}")
                    
                    print("   OK: Inference completed")
                    
                    # Store original result before text extraction
                    original_result = result
                    
                    # Check for JSON files with bbox data before cleanup
                    if os.path.exists(temp_output_dir):
                        json_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.json')]
                        for json_file in json_files:
                            try:
                                import json
                                json_path = os.path.join(temp_output_dir, json_file)
                                with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    saved_json_data = json.load(f)
                                    print(f"   🔍 DEBUG: Loaded JSON from {json_file} for bbox extraction")
                                    break
                            except Exception as json_err:
                                print(f"   DEBUG: Could not load JSON {json_file}: {json_err}")
                    
                    # Cleanup temp directory after reading JSON (if any)
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
                full_text = None
                
                # Preserve original result for bbox extraction
                original_result = result
                
                # Try to decode if result is token IDs (tensor or list)
                if result is not None:
                    try:
                        import torch
                        if isinstance(result, torch.Tensor):
                            # Decode token IDs to text
                            print("   DEBUG: Result is a tensor, attempting to decode...")
                            decoded = self.tokenizer.decode(result, skip_special_tokens=False)
                            if decoded and decoded.strip() and decoded.strip().lower() != 'none':
                                result = decoded
                                print(f"   DEBUG: Decoded {len(decoded)} characters from tensor")
                        elif isinstance(result, (list, tuple)) and len(result) > 0:
                            # Might be token IDs as list
                            if isinstance(result[0], int) or (hasattr(result[0], 'item') and callable(getattr(result[0], 'item', None))):
                                print("   DEBUG: Result appears to be token IDs list, attempting to decode...")
                                decoded = self.tokenizer.decode(result, skip_special_tokens=False)
                                if decoded and decoded.strip() and decoded.strip().lower() != 'none':
                                    result = decoded
                                    print(f"   DEBUG: Decoded {len(decoded)} characters from list")
                    except Exception as decode_err:
                        print(f"   DEBUG: Could not decode result (might not be token IDs): {decode_err}")
                
                if result is None:
                    print("   WARNING: Model infer() returned None - checking for output files...")
                    full_text = ""  # Will be handled below
                elif isinstance(result, dict):
                    full_text = result.get('text', result.get('output', result.get('result', '')))
                    markdown_output = result.get('markdown', result.get('md', None))
                    if not full_text:
                        full_text = str(result) if result else ""
                elif isinstance(result, str):
                    # Handle the case where result is the string "None" (not Python None)
                    if result.strip().lower() == 'none':
                        print("   WARNING: Model returned string 'None' - treating as empty")
                        full_text = ""
                    else:
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
                    # Try to convert to string, but handle None properly
                    if result is not None:
                        full_text = str(result)
                        # Check if it's the string "None"
                        if full_text.strip().lower() == 'none':
                            print("   WARNING: Result converted to string 'None' - treating as empty")
                            full_text = ""
                    else:
                        full_text = ""
                
                # Ensure full_text is not None
                if full_text is None:
                    full_text = ""
                
                # Final check: if full_text is "None" (string), treat as empty
                if isinstance(full_text, str) and full_text.strip().lower() == 'none':
                    print("   WARNING: Final check: full_text is 'None' string - treating as empty")
                    full_text = ""
                
                # If markdown wasn't explicitly returned but structure extraction was used,
                # treat the full_text as markdown
                if extract_structure and markdown_output is None and full_text:
                    markdown_output = full_text
                
                if not full_text or len(full_text.strip()) == 0:
                    print("   WARNING: No text extracted from image")
                    print(f"   DEBUG: full_text value: {repr(full_text)}")
                    print(f"   DEBUG: full_text type: {type(full_text)}")
                    full_text = ""  # Return empty string instead of raising error
                    markdown_output = ""
                    
            else:
                raise Exception(
                    "Model does not have 'infer' method. "
                    "Please ensure you're using the correct DeepSeek-OCR model from HuggingFace: "
                    "deepseek-ai/DeepSeek-OCR"
                )
            
            # Helper function to extract bbox from various formats
            def extract_bbox(item, default_bbox=[0, 0, 0, 0]):
                """
                Extract [x1, y1, x2, y2] from model output item
                Supports multiple bbox formats
                """
                if item is None:
                    return default_bbox
                
                # Try common field names
                bbox_data = None
                if isinstance(item, dict):
                    bbox_data = (item.get('bbox') or item.get('box') or 
                                item.get('quad_box') or item.get('geometry') or
                                item.get('coordinates') or item.get('bounding_box') or
                                item.get('rect') or item.get('region'))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Might be [text, bbox] or [x1, y1, x2, y2]
                    if len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
                        return [int(x) for x in item]
                    elif len(item) == 2 and isinstance(item[1], (list, tuple)):
                        bbox_data = item[1]
                
                if bbox_data is None:
                    return default_bbox
                
                # Handle different bbox formats
                if isinstance(bbox_data, (list, tuple)):
                    if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                        # Format: [x1, y1, x2, y2]
                        return [int(x) for x in bbox_data]
                    elif len(bbox_data) == 4 and isinstance(bbox_data[0], (list, tuple)):
                        # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (quadrilateral)
                        xs = [p[0] for p in bbox_data if isinstance(p, (list, tuple)) and len(p) >= 2]
                        ys = [p[1] for p in bbox_data if isinstance(p, (list, tuple)) and len(p) >= 2]
                        if xs and ys:
                            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    elif len(bbox_data) == 2:
                        # Format: [width, height] - not enough info, return default
                        return default_bbox
                
                # If bbox_data is a dict with x, y, width, height
                if isinstance(bbox_data, dict):
                    x = bbox_data.get('x', bbox_data.get('left', bbox_data.get('x1', 0)))
                    y = bbox_data.get('y', bbox_data.get('top', bbox_data.get('y1', 0)))
                    w = bbox_data.get('width', bbox_data.get('w', 0))
                    h = bbox_data.get('height', bbox_data.get('h', 0))
                    if w > 0 and h > 0:
                        return [int(x), int(y), int(x + w), int(y + h)]
                
                return default_bbox
            
            # Try to extract structured data with bboxes from model output
            structured_data = []
            raw_result_for_bbox = original_result  # Use original result before text extraction
            
            # saved_json_data is already loaded in the try block above (if JSON files exist)
            # Try to extract from saved JSON first (most likely to have bboxes)
            if saved_json_data:
                if isinstance(saved_json_data, dict):
                    # Check for structured_data, blocks, regions
                    for key in ['structured_data', 'blocks', 'regions', 'text_blocks', 'ocr_results']:
                        if key in saved_json_data:
                            items = saved_json_data[key]
                            if isinstance(items, list):
                                for item in items:
                                    if isinstance(item, dict):
                                        text = item.get('text', item.get('content', item.get('label', '')))
                                        if text:
                                            structured_data.append({
                                                'text': text,
                                                'confidence': float(item.get('confidence', item.get('score', 0.95))),
                                                'bbox': extract_bbox(item)
                                            })
                                if structured_data:
                                    print(f"   ✅ Extracted {len(structured_data)} blocks from JSON {key}")
                                    break
                elif isinstance(saved_json_data, list):
                    # JSON is a list of items
                    for item in saved_json_data:
                        if isinstance(item, dict):
                            text = item.get('text', item.get('content', item.get('label', '')))
                            if text:
                                structured_data.append({
                                    'text': text,
                                    'confidence': float(item.get('confidence', item.get('score', 0.95))),
                                    'bbox': extract_bbox(item)
                                })
                    if structured_data:
                        print(f"   ✅ Extracted {len(structured_data)} blocks from JSON list")
            
            # Check if result contains structured data
            if isinstance(raw_result_for_bbox, dict):
                # Check for structured_data key
                if 'structured_data' in raw_result_for_bbox:
                    structured_items = raw_result_for_bbox['structured_data']
                    if isinstance(structured_items, list):
                        for item in structured_items:
                            if isinstance(item, dict):
                                text = item.get('text', item.get('content', ''))
                                if text:
                                    structured_data.append({
                                        'text': text,
                                        'confidence': float(item.get('confidence', item.get('score', 0.95))),
                                        'bbox': extract_bbox(item)
                                    })
                # Check for blocks, regions, or text_blocks
                elif 'blocks' in raw_result_for_bbox:
                    for block in raw_result_for_bbox['blocks']:
                        if isinstance(block, dict):
                            text = block.get('text', block.get('content', ''))
                            if text:
                                structured_data.append({
                                    'text': text,
                                    'confidence': float(block.get('confidence', block.get('score', 0.95))),
                                    'bbox': extract_bbox(block)
                                })
                elif 'regions' in raw_result_for_bbox:
                    for region in raw_result_for_bbox['regions']:
                        if isinstance(region, dict):
                            text = region.get('text', region.get('content', ''))
                            if text:
                                structured_data.append({
                                    'text': text,
                                    'confidence': float(region.get('confidence', region.get('score', 0.95))),
                                    'bbox': extract_bbox(region)
                                })
            
            # If no structured data found, fall back to line-based extraction
            if not structured_data:
                lines = full_text.split('\n') if full_text else []
                structured_data = [
                    {
                        'text': line,
                        'confidence': 0.95,
                        'bbox': [0, 0, 0, 0]  # No bbox available from text-only output
                    }
                    for line in lines if line.strip()
                ]
                print("   ⚠️  WARNING: No structured data with bboxes found - using text-only extraction")
            
            # Validate bboxes and log statistics
            valid_bboxes = sum(1 for item in structured_data if item['bbox'] != [0, 0, 0, 0])
            total_blocks = len(structured_data)
            if total_blocks > 0:
                bbox_percentage = (valid_bboxes / total_blocks) * 100
                print(f"   📊 Bbox Statistics: {valid_bboxes}/{total_blocks} valid bounding boxes ({bbox_percentage:.1f}%)")
                if valid_bboxes > 0:
                    # Show sample bbox
                    sample_bbox = next((item['bbox'] for item in structured_data if item['bbox'] != [0, 0, 0, 0]), None)
                    if sample_bbox:
                        print(f"   📊 Sample bbox: {sample_bbox}")
            else:
                print("   ⚠️  WARNING: No structured data blocks extracted")
            
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

