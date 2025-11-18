"""
Simple OCR Engine using EasyOCR

Fast OCR processing for simple documents using EasyOCR library.
Returns text with bounding boxes in [x1, y1, x2, y2] format.
"""
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
import time

# Configure logging
logger = logging.getLogger(__name__)

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")


class SimpleOCREngine:
    """Fast OCR engine using EasyOCR for simple documents"""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize EasyOCR reader
        
        Args:
            languages: List of language codes (e.g., ['en', 'es'])
            gpu: Whether to use GPU if available
        """
        self.languages = languages
        self.gpu = gpu
        self.reader: Optional[object] = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of EasyOCR reader"""
        if self._initialized:
            return
        
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR is not installed. "
                "Install with: pip install easyocr"
            )
        
        try:
            logger.info(f"Initializing EasyOCR reader (languages: {self.languages}, GPU: {self.gpu})...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            self._initialized = True
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise Exception(f"EasyOCR initialization failed: {str(e)}")
    
    def process(self, image: Image.Image) -> Dict:
        """
        Process image and extract text with bounding boxes
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with format:
            {
                'blocks': [
                    {
                        'text': str,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float
                    }
                ],
                'engine': 'simple',
                'processing_time': float
            }
        """
        start_time = time.time()
        
        try:
            # Initialize if needed
            if not self._initialized:
                self._initialize()
            
            # Convert PIL Image to numpy array
            np_image = np.array(image)
            
            # EasyOCR expects RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
                np_image = np.array(image)
            
            logger.debug(f"Processing image: {np_image.shape}")
            
            # Run OCR
            results = self.reader.readtext(np_image)
            
            # Process results into standardized format
            blocks = []
            for detection in results:
                try:
                    # EasyOCR returns: (bbox, text, confidence)
                    bbox_points, text, confidence = detection
                    
                    # Convert bbox from 4 points to [x1, y1, x2, y2]
                    # EasyOCR bbox is list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if len(bbox_points) == 4:
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        x1 = int(min(x_coords))
                        y1 = int(min(y_coords))
                        x2 = int(max(x_coords))
                        y2 = int(max(y_coords))
                        
                        bbox = [x1, y1, x2, y2]
                    else:
                        # Fallback: use first and last point
                        x1 = int(bbox_points[0][0])
                        y1 = int(bbox_points[0][1])
                        x2 = int(bbox_points[-1][0])
                        y2 = int(bbox_points[-1][1])
                        bbox = [x1, y1, x2, y2]
                    
                    # Ensure confidence is float
                    confidence = float(confidence) if confidence is not None else 0.0
                    
                    blocks.append({
                        'text': text.strip() if text else '',
                        'bbox': bbox,
                        'confidence': confidence
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing detection: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            logger.info(f"EasyOCR processed {len(blocks)} text blocks in {processing_time:.2f}s")
            
            return {
                'blocks': blocks,
                'engine': 'simple',
                'processing_time': processing_time,
                'total_blocks': len(blocks)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in SimpleOCR processing: {e}", exc_info=True)
            raise Exception(f"SimpleOCR processing failed: {str(e)}")
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract plain text from image (convenience method)
        
        Args:
            image: PIL Image object
            
        Returns:
            Plain text string
        """
        result = self.process(image)
        text_lines = [block['text'] for block in result['blocks']]
        return '\n'.join(text_lines)
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available"""
        return EASYOCR_AVAILABLE

