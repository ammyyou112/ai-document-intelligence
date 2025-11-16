# DeepSeek OCR - Intelligent Document Processing

A web application for intelligent AI document OCR using DeepSeek OCR model that converts scanned documents into structured JSON and TXT formats for AI training.

## Features

- 🚀 **DeepSeek OCR Processing**: Uses DeepSeek OCR model (API or local) for accurate text extraction
- 📄 **Multi-format Support**: Supports PDF, PNG, JPG, JPEG, TIFF, and BMP files
- 📊 **Structured Output**: Extracts text with bounding boxes, confidence scores, and document structure
- 📥 **Multiple Export Formats**: Export results as JSON and TXT files
- 🎨 **Modern UI**: Beautiful, responsive web interface with drag-and-drop support
- 🔍 **Document Structure Detection**: Automatically detects headers, paragraphs, lists, and other document elements
- 📈 **Processing Statistics**: Provides detailed statistics about extracted content
- 🌐 **API & Local Support**: Works with DeepSeek OCR API or local model installation

## Installation

### Prerequisites

- Python 3.8 or higher
- **DeepSeek-OCR local model** (REQUIRED - Recommended method)
- DeepSeek API Key (OPTIONAL - Only if you have API access)

#### Setting Up DeepSeek OCR

**RECOMMENDED: Install DeepSeek-OCR Locally**

DeepSeek OCR does not provide public API keys. The recommended way is to install the local model from GitHub:

1. **Clone the DeepSeek-OCR repository:**
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd DeepSeek-OCR
   ```

2. **Follow the installation instructions in the repository:**
   - Check the README.md in the DeepSeek-OCR repository
   - Install required dependencies
   - Download model weights if needed
   - Set up the environment

3. **Install DeepSeek-OCR as a package or add to Python path:**
   
   **Option A: Install as package (recommended)**
   ```bash
   cd DeepSeek-OCR
   pip install -e .
   ```
   
   **Option B: Add to Python path**
   ```python
   # In your app.py or add to sys.path
   import sys
   sys.path.append('/path/to/DeepSeek-OCR')
   ```

4. **Verify installation:**
   ```bash
   python -c "from deepseek_ocr import DeepSeekOCR; print('✅ DeepSeek-OCR installed successfully')"
   ```

5. **The application will automatically detect and use the local model**

**OPTIONAL: Using DeepSeek OCR API (if you have access)**

1. Get your API key (if available)
2. Create a `.env` file in the project root:
   ```bash
   DEEPSEEK_API_KEY=your_api_key_here
   ```
3. Copy `config.example.env` to `.env` and add your API key
4. **Note:** API is used as fallback only if local model is not available

#### Installing Poppler (Required for PDF processing)

**Windows:**
1. Download Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract the ZIP file
3. Add the `bin` folder to your system PATH
4. Or place poppler in a folder and add it to PATH in your system environment variables

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**Note:** If you don't install Poppler, PDF processing will not work. You can still use image files (PNG, JPG, etc.) without Poppler.

### Setup

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. **Install DeepSeek-OCR locally (REQUIRED):**
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd DeepSeek-OCR
   # Follow installation instructions in the repository
   pip install -e .  # or add to Python path
   ```
   
   **Note:** The application prioritizes the local model. API is optional and used only as fallback.

4. Verify installation:
```bash
python setup_check.py
```

This script will check if all dependencies are installed correctly and provide guidance on any missing components.

5. Start the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Start the Flask application:**
```bash
python app.py
```

2. **Open your web browser and navigate to:**
```
http://localhost:5000
```

3. **Upload a document:**
   - Click "Choose File" or drag and drop a document
   - Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP

4. **Process the document:**
   - Click "Process Document" button
   - Wait for processing to complete (processing time depends on document size and OCR method)

5. **Download results:**
   - Download JSON file with structured data
   - Download TXT file with extracted text

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# DeepSeek OCR API Key (required for API usage)
DEEPSEEK_API_KEY=your_api_key_here

# Optional: Custom API URL
# DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
```

### Local Model vs API

- **Local Model (RECOMMENDED)**: Requires DeepSeek-OCR installation from GitHub. 
  - ✅ Works offline
  - ✅ No API costs
  - ✅ Full control over the model
  - ✅ Better for production/private use
  - ✅ No rate limits
  - ✅ Better for high-volume processing

- **API Mode (OPTIONAL)**: Requires `DEEPSEEK_API_KEY` in `.env` file (if you have API access).
  - ⚠️  Requires internet connection
  - ⚠️  May have rate limits
  - ⚠️  May incur costs
  - ⚠️  Used only as fallback if local model is not available

## API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and OCR engine information.

### OCR Processing
```
POST /api/ocr
```
Upload a document file for OCR processing.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (document file)

**Response:**
```json
{
  "success": true,
  "document_id": "uuid",
  "result": {
    "document_id": "uuid",
    "filename": "document.pdf",
    "pages": [...],
    "full_document_text": "...",
    "full_document_structure": {...},
    "metadata": {...}
  },
  "download_json": "/api/download/json/{document_id}",
  "download_txt": "/api/download/txt/{document_id}"
}
```

### Download JSON
```
GET /api/download/json/<document_id>
```
Download the JSON file with structured OCR results.

### Download TXT
```
GET /api/download/txt/<document_id>
```
Download the TXT file with extracted text.

## Output Formats

### JSON Format
The JSON output contains:
- Full document text
- Structured data with bounding boxes and confidence scores
- Document structure (headers, paragraphs, tables, lists)
- Metadata (page count, processing timestamp, OCR engine)

### TXT Format
The TXT output contains:
- Plain text extraction from all pages
- Document metadata
- Formatted text for easy reading

## Advanced Configuration

You can modify the following settings in `app.py`:

- `UPLOAD_FOLDER`: Directory for uploaded files
- `OUTPUT_FOLDER`: Directory for output files
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 50MB)
- `ALLOWED_EXTENSIONS`: Supported file types
- `DEEPSEEK_API_URL`: DeepSeek API endpoint (default: https://api.deepseek.com/v1/chat/completions)

## Project Structure

```
Deepseek-OCR/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Uploaded files (created automatically)
└── outputs/              # Output files (created automatically)
```

## Technologies Used

- **Flask**: Web framework
- **DeepSeek OCR**: OCR engine (API or local model)
- **OpenCV**: Image preprocessing
- **Pillow**: Image processing
- **pdf2image**: PDF to image conversion
- **Flask-CORS**: Cross-origin resource sharing
- **Requests**: HTTP client for API calls
- **python-dotenv**: Environment variable management

## Performance Tips

1. **API Usage**: Using DeepSeek OCR API is recommended for most users as it doesn't require local GPU resources

2. **Local Model**: For high-volume processing or offline use, install the local DeepSeek-OCR model with GPU support

3. **Image Preprocessing**: The application automatically handles image preprocessing for optimal OCR results

4. **Batch Processing**: For multiple documents, consider implementing batch processing or using the API with rate limiting

## Troubleshooting

### DeepSeek API Key not configured
- Make sure you have created a `.env` file with your API key
- Get your API key from: https://platform.deepseek.com/
- Check that the `.env` file is in the project root directory

### DeepSeek API errors
- Verify your API key is correct and has sufficient credits
- Check your internet connection
- Review the API response in the application logs

### Local model not found
- If using local model, make sure DeepSeek-OCR is properly installed
- Check that the DeepSeek-OCR package is in your Python path
- Refer to the DeepSeek-OCR repository for installation instructions

### Memory issues with large PDFs
- Reduce the DPI in `pdf_to_images()` function
- Process pages individually for very large documents
- Consider using the API instead of local model for large documents

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.

