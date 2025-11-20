from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import sys
import json
import base64
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import tempfile
from datetime import datetime
import uuid
import platform
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
LOG_FOLDER = 'logs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Configure logging
import logging
log_file_path = os.path.join(LOG_FOLDER, 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# DeepSeek OCR Configuration
# Priority: 1. Local Model (recommended), 2. API (optional, if available)
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_OCR_API_URL = os.getenv('DEEPSEEK_OCR_API_URL', 'https://api.deepseek-ocr.ai/v1/ocr')
DEEPSEEK_CHAT_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')
OCR_ENGINE = 'none'
deepseek_ocr_model = None

# Try to initialize DeepSeek OCR model locally (PRIORITY: Local Model)
print("=" * 60)
print("Initializing DeepSeek OCR...")
print("=" * 60)

try:
    # Try importing DeepSeek OCR wrapper
    # First, try our custom wrapper that uses Transformers
    try:
        from deepseek_ocr_wrapper import DeepSeekOCR, check_dependencies
        
        # Check dependencies
        deps = check_dependencies()
        if not deps.get('transformers'):
            raise ImportError("transformers library not installed")
        if not deps.get('torch'):
            raise ImportError("torch library not installed")
        
        # Try to initialize model (will download from HuggingFace on first use)
        try:
            deepseek_ocr_model = DeepSeekOCR()
            OCR_ENGINE = 'deepseek-local'
            print("OK: DeepSeek OCR wrapper initialized")
            print("WARNING: Note: Model will be downloaded from HuggingFace on first use")
            print("WARNING: This requires internet connection and may take time")
        except Exception as e:
            print(f"WARNING: Failed to initialize DeepSeek OCR model: {e}")
            print("WARNING: Model will be loaded on first OCR request")
            OCR_ENGINE = 'deepseek-local'
            deepseek_ocr_model = None  # Will be initialized lazily
            
    except ImportError as e:
        # Try importing from DeepSeek-OCR repository directly
        deepseek_ocr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DeepSeek-OCR', 'DeepSeek-OCR', 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm')
        if os.path.exists(deepseek_ocr_path):
            if deepseek_ocr_path not in sys.path:
                sys.path.insert(0, deepseek_ocr_path)
                print(f"OK: Added DeepSeek-OCR path: {deepseek_ocr_path}")
        
        # Try importing from repository
        try:
            from deepseek_ocr import DeepSeekOCRForCausalLM
            OCR_ENGINE = 'deepseek-local-vllm'
            print("OK: DeepSeek-OCR vLLM model found")
            print("WARNING: Note: vLLM setup required for full functionality")
            deepseek_ocr_model = None  # vLLM requires different initialization
        except ImportError:
            raise ImportError("DeepSeek-OCR not available")
        
except ImportError:
    print("WARNING: DeepSeek OCR local model not found")
    print("WARNING: Please install DeepSeek-OCR from: https://github.com/deepseek-ai/DeepSeek-OCR")
    print()
    print("Installation steps:")
    print("  1. git clone https://github.com/deepseek-ai/DeepSeek-OCR.git")
    print("  2. cd DeepSeek-OCR")
    print("  3. Follow the installation instructions in the repository")
    print("  4. Add DeepSeek-OCR to your Python path or install as package")
    print()
    
    # Check if API key is available as fallback
    if DEEPSEEK_API_KEY:
        OCR_ENGINE = 'deepseek-api'
        print("INFO: API key found - will use API as fallback (if available)")
    else:
        OCR_ENGINE = 'none'
        print("WARNING: No API key found - API access is optional")
        print("WARNING: Local model installation is recommended")
except Exception as e:
    print(f"WARNING: DeepSeek OCR initialization error: {e}")
    OCR_ENGINE = 'none'
    deepseek_ocr_model = None

print("=" * 60)
print(f"OCR Engine: {OCR_ENGINE}")
print(f"Local Model: {'OK: Available' if deepseek_ocr_model else 'ERROR: Not Available'}")
print(f"API Key: {'OK: Configured' if DEEPSEEK_API_KEY else 'WARNING: Not Configured (optional)'}")
print("=" * 60)
print()

# Initialize Enhanced OCR Pipeline
enhanced_pipeline = None
try:
    from app.processors.enhanced_ocr_pipeline import EnhancedOCRPipeline
    print("Initializing Enhanced OCR Pipeline...")
    
    # Create processor function for DeepSeek
    def deepseek_processor(image_path: str):
        """Wrapper for DeepSeek OCR processing"""
        global deepseek_ocr_model
        if deepseek_ocr_model is None:
            from deepseek_ocr_wrapper import DeepSeekOCR
            deepseek_ocr_model = DeepSeekOCR()
        return deepseek_ocr_model.process(image_path)
    
    enhanced_pipeline = EnhancedOCRPipeline(
        deepseek_processor=deepseek_processor if OCR_ENGINE in ['deepseek-local', 'deepseek-local-vllm'] else None,
        complexity_threshold=0.7,
        pdf_dpi=300
    )
    print("OK: Enhanced OCR Pipeline initialized")
except ImportError as e:
    print(f"WARNING: Enhanced OCR Pipeline not available: {e}")
    print("WARNING: Falling back to basic OCR processing")
    enhanced_pipeline = None
except Exception as e:
    print(f"WARNING: Failed to initialize Enhanced OCR Pipeline: {e}")
    enhanced_pipeline = None

print()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to base64 string for API"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_deepseek_api(image_path):
    """Extract text using DeepSeek OCR API"""
    if not DEEPSEEK_API_KEY:
        raise Exception(
            "DeepSeek API key not configured. "
            "Please set DEEPSEEK_API_KEY environment variable or in .env file. "
            "Get your API key from: https://platform.deepseek.com/ or https://www.deepseek-ocr.ai/"
        )
    
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }
    
    # Try OCR-specific API endpoint first (file upload format)
    ocr_api_available = True
    try:
        with open(image_path, 'rb') as image_file:
            files = {
                'file': (os.path.basename(image_path), image_file, 'image/png')
            }
            response = requests.post(
                DEEPSEEK_OCR_API_URL, 
                headers=headers, 
                files=files, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # Parse OCR API response
                return parse_ocr_api_response(result)
            elif response.status_code == 404:
                # OCR API endpoint doesn't exist, try chat API
                print("WARNING: OCR API endpoint not found, trying chat API...")
                ocr_api_available = False
            else:
                error_msg = response.text
                raise Exception(f"DeepSeek OCR API error ({response.status_code}): {error_msg}")
    except requests.exceptions.RequestException as e:
        # Network error or connection issue
        print(f"WARNING: OCR API request failed: {e}, trying chat API...")
        ocr_api_available = False
    except Exception as e:
        # Re-raise other exceptions
        if "DeepSeek OCR API error" in str(e):
            raise
        print(f"WARNING: OCR API error: {e}, trying chat API...")
        ocr_api_available = False
    
    # Fallback to chat API if OCR API is not available
    if not ocr_api_available:
        # Fallback: Try chat API with vision capabilities
        try:
            image_base64 = image_to_base64(image_path)
            
            payload = {
                'model': 'deepseek-chat',  # Use chat model that supports vision
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Extract all text from this image. Return the text exactly as it appears, preserving line breaks and structure.'
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/png;base64,{image_base64}'
                                }
                            }
                        ]
                    }
                ],
                'max_tokens': 4000
            }
            
            headers['Content-Type'] = 'application/json'
            response = requests.post(DEEPSEEK_CHAT_API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                error_msg = response.text
                raise Exception(f"DeepSeek Chat API error ({response.status_code}): {error_msg}")
            
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Parse chat API response (text only)
            return parse_chat_api_response(content)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek Chat API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"DeepSeek Chat API error: {str(e)}")
    
    # If we get here, both APIs failed
    raise Exception(
        "DeepSeek OCR API is not available. "
        "Please check your API key and network connection, "
        "or install DeepSeek-OCR locally from: https://github.com/deepseek-ai/DeepSeek-OCR"
    )

def parse_ocr_api_response(result):
    """Parse DeepSeek OCR API response"""
    # Handle different response formats
    if isinstance(result, dict):
        # If response has structured data
        if 'text' in result:
            full_text = result['text']
        elif 'full_text' in result:
            full_text = result['full_text']
        elif 'content' in result:
            full_text = result['content']
        else:
            # Try to get text from nested structure
            full_text = json.dumps(result)
        
        structured_data = result.get('structured_data', [])
        metadata = result.get('metadata', {})
    else:
        # If response is string
        full_text = str(result)
        structured_data = []
        metadata = {}
    
    # Create structured data if not provided
    if not structured_data and full_text:
        lines = full_text.split('\n')
        structured_data = [
            {
                'text': line,
                'confidence': 0.95,
                'bbox': [0, 0, 0, 0]
            }
            for line in lines if line.strip()
        ]
    
    # Calculate metadata
    if not metadata:
        metadata = {
            'total_words': len(full_text.split()),
            'total_lines': len(full_text.split('\n')),
            'confidence_scores': [0.95] * len(full_text.split('\n')) if full_text.split('\n') else [],
            'average_confidence': 0.95
        }
    
    return {
        'full_text': full_text,
        'structured_data': structured_data,
        'metadata': metadata
    }

def parse_chat_api_response(content):
    """Parse DeepSeek Chat API response (text only)"""
    # Chat API returns plain text, so we structure it
    full_text = content.strip()
    
    lines = full_text.split('\n')
    structured_data = [
        {
            'text': line,
            'confidence': 0.95,
            'bbox': [0, 0, 0, 0]
        }
        for line in lines if line.strip()
    ]
    
    metadata = {
        'total_words': len(full_text.split()),
        'total_lines': len(lines),
        'confidence_scores': [0.95] * len(lines) if lines else [],
        'average_confidence': 0.95
    }
    
    return {
        'full_text': full_text,
        'structured_data': structured_data,
        'metadata': metadata
    }

def extract_text_deepseek_local(image_path):
    """Extract text using local DeepSeek OCR model"""
    try:
        # Import wrapper (lazy import to handle initialization)
        from deepseek_ocr_wrapper import DeepSeekOCR
        
        # Initialize model if not already initialized (lazy loading)
        global deepseek_ocr_model
        if deepseek_ocr_model is None:
            print("Initializing DeepSeek-OCR model (this may take a while on first use)...")
            print("WARNING: Model will be downloaded from HuggingFace (~several GB)")
            deepseek_ocr_model = DeepSeekOCR()
        
        # Process image with DeepSeek OCR with structure extraction enabled
        # Use smaller sizes for CPU to avoid memory issues
        use_cpu = deepseek_ocr_model.device == 'cpu' if hasattr(deepseek_ocr_model, 'device') else True
        if use_cpu:
            # CPU mode - use smaller sizes
            result = deepseek_ocr_model.process(
                image_path, 
                base_size=640, 
                image_size=640, 
                crop_mode=False,
                extract_structure=True  # Enable structure extraction
            )
        else:
            # GPU mode - can use larger sizes
            result = deepseek_ocr_model.process(
                image_path, 
                base_size=1024, 
                image_size=640, 
                crop_mode=True,
                extract_structure=True  # Enable structure extraction
            )
        
        # Convert result to standardized format
        # The wrapper returns 'full_text', 'markdown_output', 'structured_data', and 'metadata'
        extracted_data = {
            'full_text': result.get('full_text', result.get('text', '')),
            'markdown_output': result.get('markdown_output', result.get('full_text', '')),
            'structured_data': result.get('structured_data', []),
            'metadata': result.get('metadata', {
                'total_words': 0,
                'total_lines': 0,
                'confidence_scores': [],
                'average_confidence': 0.95
            })
        }
        
        # If structured_data is not in expected format, create it
        if not extracted_data['structured_data'] and extracted_data['full_text']:
            lines = extracted_data['full_text'].split('\n')
            extracted_data['structured_data'] = [
                {
                    'text': line,
                    'confidence': 0.95,
                    'bbox': [0, 0, 0, 0]
                }
                for line in lines if line.strip()
            ]
        
        return extracted_data
        
    except ImportError as e:
        raise Exception(
            f"DeepSeek-OCR wrapper not available: {str(e)}\n"
            f"Please install dependencies:\n"
            f"  pip install transformers torch\n"
            f"Or install DeepSeek-OCR from: https://github.com/deepseek-ai/DeepSeek-OCR"
        )
    except Exception as e:
        raise Exception(
            f"DeepSeek OCR local model error: {str(e)}\n"
            f"Make sure you have:\n"
            f"1. transformers and torch installed: pip install transformers torch\n"
            f"2. GPU with CUDA (recommended) or sufficient RAM for CPU\n"
            f"3. Internet connection for model download from HuggingFace\n"
            f"4. Sufficient disk space (~several GB for model weights)"
        )

def extract_text_deepseek(image_path):
    """Extract text using DeepSeek OCR (prioritizes local model)"""
    # Priority 1: Try local model first (recommended)
    if OCR_ENGINE in ['deepseek-local', 'deepseek-local-vllm']:
        try:
            return extract_text_deepseek_local(image_path)
        except Exception as e:
            error_msg = str(e)
            print(f"WARNING: Local model failed: {error_msg}")
            
            # Only fallback to API if local model fails and API is available
            if DEEPSEEK_API_KEY:
                print("WARNING: Attempting to use API as fallback...")
                try:
                    return extract_text_deepseek_api(image_path)
                except Exception as api_error:
                    raise Exception(
                        f"Both local model and API failed.\n"
                        f"Local model error: {error_msg}\n"
                        f"API error: {str(api_error)}\n"
                        f"Please check your DeepSeek-OCR installation or API configuration."
                    )
            else:
                raise Exception(
                    f"DeepSeek OCR local model failed: {error_msg}\n"
                    f"Please check your DeepSeek-OCR installation from: "
                    f"https://github.com/deepseek-ai/DeepSeek-OCR"
                )
    
    # Priority 2: Try API only if local model is not available
    if OCR_ENGINE == 'deepseek-api' and DEEPSEEK_API_KEY:
        try:
            return extract_text_deepseek_api(image_path)
        except Exception as e:
            raise Exception(
                f"DeepSeek OCR API failed: {str(e)}\n"
                f"Please install DeepSeek-OCR locally from: "
                f"https://github.com/deepseek-ai/DeepSeek-OCR (recommended)"
            )
    
    # No OCR available
    raise Exception(
        "DeepSeek OCR is not configured.\n\n"
        "RECOMMENDED: Install DeepSeek-OCR locally:\n"
        "  1. git clone https://github.com/deepseek-ai/DeepSeek-OCR.git\n"
        "  2. cd DeepSeek-OCR\n"
        "  3. Follow installation instructions in the repository\n"
        "  4. Add to Python path or install as package\n\n"
        "OPTIONAL: Set DEEPSEEK_API_KEY in .env file if you have API access"
    )

def detect_document_structure(structured_data, markdown_text=None):
    """Detect document structure using DocumentAnalyzer"""
    from document_analyzer import DocumentAnalyzer
    
    analyzer = DocumentAnalyzer()
    
    # Use markdown if available, otherwise use structured data
    text_to_analyze = markdown_text if markdown_text else ' '.join([item.get('text', '') for item in structured_data])
    
    # Extract hierarchical structure
    hierarchical_structure = analyzer.extract_hierarchical_structure(text_to_analyze)
    
    # Extract tables
    tables = analyzer.extract_tables(text_to_analyze)
    
    # Extract figures
    figures = analyzer.extract_figures(text_to_analyze, markdown_text)
    
    # Legacy format for backward compatibility
    structure = {
        'headers': [{'text': s['heading'], 'level': s['level']} for s in hierarchical_structure.get('sections', [])],
        'paragraphs': [{'text': ' '.join(s.get('content', [])), 'word_count': len(' '.join(s.get('content', [])).split())} 
                      for s in hierarchical_structure.get('sections', []) if s.get('content')],
        'tables': tables,
        'figures': figures,
        'lists': []
    }
    
    # Add hierarchical structure
    structure['hierarchical'] = hierarchical_structure
    
    return structure

def pdf_to_images(pdf_path):
    """Convert PDF to images"""
    try:
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        for i, image in enumerate(images):
            image_path = pdf_path.replace('.pdf', f'_page_{i+1}.png')
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        error_msg = str(e)
        if 'poppler' in error_msg.lower() or 'pdfinfo' in error_msg.lower():
            raise Exception(
                "PDF processing requires Poppler. "
                "Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/ "
                "and add to PATH. Or use image files instead."
            )
        raise Exception(f"PDF conversion error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if OCR_ENGINE != 'none' else 'not_configured',
        'ocr_engine': OCR_ENGINE,
        'has_local_model': deepseek_ocr_model is not None,
        'has_api_key': bool(DEEPSEEK_API_KEY),
        'has_enhanced_pipeline': enhanced_pipeline is not None,
        'recommended_method': 'local' if deepseek_ocr_model else 'api' if DEEPSEEK_API_KEY else 'none',
        'message': 'Ready' if OCR_ENGINE != 'none' else 'Please install DeepSeek-OCR locally',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    """Enhanced OCR endpoint using Enhanced OCR Pipeline"""
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("New OCR request received")
        logger.info("=" * 60)
        
        if 'file' not in request.files:
            logger.error("No file provided in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        logger.info(f"Saving uploaded file: {filename} ({file.content_length / 1024 / 1024:.2f} MB)")
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Use Enhanced Pipeline if available, otherwise fallback to basic processing
        if enhanced_pipeline is not None:
            logger.info("Using Enhanced OCR Pipeline...")
            try:
                # Process document with enhanced pipeline
                result = enhanced_pipeline.process_document(file_path, filename)
                
                # Generate output files
                output_id = result['document_metadata']['document_id']
                
                # Save comprehensive JSON
                json_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_id}.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"JSON saved: {json_path}")
                
                # Save TXT file
                txt_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_id}.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Document: {filename}\n")
                    f.write(f"Processing Date: {result['document_metadata']['processed_date']}\n")
                    f.write(f"Document Type: {result['document_metadata']['type']}\n")
                    f.write(f"Engines Used: {', '.join(result['document_metadata']['engines_used'])}\n")
                    f.write(f"Total Pages: {result['document_metadata']['total_pages']}\n")
                    f.write(f"Total Words: {result['content_metadata']['words']}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Write page texts
                    for page in result['pages']:
                        f.write(f"\n--- Page {page['page_number']} ---\n\n")
                        f.write(page['text'])
                        f.write("\n")
                
                logger.info(f"TXT saved: {txt_path}")
                
                # Cleanup uploaded file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
                
                logger.info("OCR processing completed successfully!")
                logger.info("=" * 60)
                
                return jsonify({
                    'success': True,
                    'document_id': output_id,
                    'result': result,
                    'download_json': f'/api/download/json/{output_id}',
                    'download_txt': f'/api/download/txt/{output_id}'
                })
                
            except Exception as pipeline_error:
                logger.error(f"Enhanced pipeline error: {pipeline_error}", exc_info=True)
                # Fall through to basic processing
                logger.warning("Falling back to basic OCR processing...")
        
        # Fallback to basic processing
        logger.info("Using basic OCR processing (fallback)...")
        
        # Process file
        image_paths = []
        if filename.lower().endswith('.pdf'):
            logger.info("Converting PDF to images...")
            image_paths = pdf_to_images(file_path)
            logger.info(f"Converted to {len(image_paths)} image(s)")
        else:
            image_paths = [file_path]
            logger.info("Processing image file...")
        
        # Process each image with basic OCR
        all_results = {
            'document_id': str(uuid.uuid4()),
            'filename': filename,
            'pages': [],
            'full_document_text': '',
            'full_document_structure': {
                'headers': [],
                'paragraphs': [],
                'tables': [],
                'lists': []
            },
            'metadata': {
                'total_pages': len(image_paths),
                'processing_timestamp': datetime.now().isoformat(),
                'ocr_engine': OCR_ENGINE
            }
        }
        
        for page_num, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing page {page_num}/{len(image_paths)}...")
            
            try:
                page_data = extract_text_deepseek(image_path)
                logger.info(f"Text extracted: {len(page_data.get('full_text', ''))} characters")
                
                markdown_text = page_data.get('markdown_output', page_data.get('full_text', ''))
                page_structure = detect_document_structure(page_data['structured_data'], markdown_text=markdown_text)
                
                page_result = {
                    'page_number': page_num,
                    'text': page_data['full_text'],
                    'markdown': page_data.get('markdown_output', page_data['full_text']),
                    'structured_data': page_data['structured_data'],
                    'structure': page_structure,
                    'metadata': page_data['metadata']
                }
                
                all_results['pages'].append(page_result)
                all_results['full_document_text'] += f"\n\n--- Page {page_num} ---\n\n" + page_data['full_text']
                
            except Exception as page_error:
                logger.error(f"Error processing page {page_num}: {page_error}", exc_info=True)
                raise Exception(f"Error processing page {page_num}: {str(page_error)}")
        
        # Generate output files
        output_id = all_results['document_id']
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_id}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        txt_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_id}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Document: {filename}\n")
            f.write(f"Processing Date: {all_results['metadata']['processing_timestamp']}\n")
            f.write(f"OCR Engine: {all_results['metadata']['ocr_engine']}\n")
            f.write(f"Total Pages: {all_results['metadata']['total_pages']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(all_results['full_document_text'])
        
        # Cleanup
        for path in image_paths:
            if path != file_path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        logger.info("Basic OCR processing completed")
        
        return jsonify({
            'success': True,
            'document_id': output_id,
            'result': all_results,
            'download_json': f'/api/download/json/{output_id}',
            'download_txt': f'/api/download/txt/{output_id}'
        })
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"OCR Error: {error_msg}", exc_info=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/download/json/<document_id>', methods=['GET'])
def download_json(document_id):
    json_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{document_id}.json')
    if os.path.exists(json_path):
        return send_file(json_path, as_attachment=True, download_name=f'{document_id}.json')
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/download/txt/<document_id>', methods=['GET'])
def download_txt(document_id):
    txt_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{document_id}.txt')
    if os.path.exists(txt_path):
        return send_file(txt_path, as_attachment=True, download_name=f'{document_id}.txt')
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("DeepSeek OCR Application")
    print("=" * 60)
    print(f"OCR Engine: {OCR_ENGINE}")
    print(f"Local Model: {'OK: Available' if deepseek_ocr_model else 'ERROR: Not Available'}")
    print(f"API Key: {'OK: Configured' if DEEPSEEK_API_KEY else 'WARNING: Not Configured (optional)'}")
    
    if OCR_ENGINE == 'none':
        print()
        print("WARNING: DeepSeek OCR is not configured!")
        print("WARNING: Please install DeepSeek-OCR locally:")
        print("   1. git clone https://github.com/deepseek-ai/DeepSeek-OCR.git")
        print("   2. cd DeepSeek-OCR")
        print("   3. Follow installation instructions")
        print("   4. Add to Python path or install as package")
        print()
    elif OCR_ENGINE == 'deepseek-local':
        print("OK: Using local model (recommended)")
    elif OCR_ENGINE == 'deepseek-api':
        print("WARNING: Using API (local model recommended)")
    
    print("=" * 60)
    print("\nServer will be available at:")
    print("  http://localhost:5000")
    print("  http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
