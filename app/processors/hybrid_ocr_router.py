"""
Hybrid OCR Router

Intelligently routes documents to appropriate OCR engine based on complexity analysis.
- Simple documents → EasyOCR (fast)
- Complex documents → DeepSeek-OCR (accurate)
"""
import logging
from typing import Dict, Optional, List
from PIL import Image
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import analyzers and engines
try:
    from app.analyzers.document_complexity_analyzer import DocumentComplexityAnalyzer, ComplexityResult
except ImportError:
    # Fallback for different import paths
    try:
        import sys
        import os
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from app.analyzers.document_complexity_analyzer import DocumentComplexityAnalyzer, ComplexityResult
    except ImportError:
        try:
            from analyzers.document_complexity_analyzer import DocumentComplexityAnalyzer, ComplexityResult
        except ImportError:
            logger.error("Could not import DocumentComplexityAnalyzer")
            DocumentComplexityAnalyzer = None
            ComplexityResult = None

try:
    from app.processors.simple_ocr_engine import SimpleOCREngine
except ImportError:
    try:
        import sys
        import os
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from app.processors.simple_ocr_engine import SimpleOCREngine
    except ImportError:
        try:
            from processors.simple_ocr_engine import SimpleOCREngine
        except ImportError:
            logger.error("Could not import SimpleOCREngine")
            SimpleOCREngine = None


class HybridOCRRouter:
    """Routes documents to appropriate OCR engine based on complexity"""
    
    def __init__(self, 
                 complexity_threshold: float = 0.5,  # Lowered from 0.7 - allow routing to simple with lower confidence
                 force_engine: Optional[str] = None):
        """
        Initialize the hybrid router
        
        Args:
            complexity_threshold: Confidence threshold for routing decision (0.0-1.0)
                                 Lowered to 0.5 to allow simple pages with moderate confidence
            force_engine: Force use of specific engine ('simple' or 'deepseek'), None for auto
        """
        self.complexity_threshold = complexity_threshold
        self.force_engine = force_engine
        
        # Initialize analyzer
        if DocumentComplexityAnalyzer is None:
            raise ImportError("DocumentComplexityAnalyzer is not available")
        self.analyzer = DocumentComplexityAnalyzer()
        
        # Initialize simple OCR engine
        if SimpleOCREngine is None:
            logger.warning("⚠️  SimpleOCREngine not available - will only use DeepSeek-OCR")
            self.simple_engine = None
        else:
            try:
                logger.info("Initializing SimpleOCR engine (EasyOCR)...")
                self.simple_engine = SimpleOCREngine()
                logger.info("✅ SimpleOCR engine initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize SimpleOCREngine: {e}")
                logger.warning("⚠️  All pages will be routed to DeepSeek-OCR")
                self.simple_engine = None
        
        # Verify initialization
        if not hasattr(self, 'simple_engine') or self.simple_engine is None:
            logger.warning("⚠️  SimpleOCR not initialized! All pages will use DeepSeek")
        
        # Statistics tracking
        self.stats = {
            'total_calls': 0,
            'simple_calls': 0,
            'deepseek_calls': 0,
            'total_processing_time': 0.0,
            'simple_processing_time': 0.0,
            'deepseek_processing_time': 0.0,
            'complexity_analyses': []
        }
    
    def process(self, 
                image: Image.Image,
                deepseek_processor: Optional[object] = None) -> Dict:
        """
        Process image with appropriate OCR engine
        
        Args:
            image: PIL Image object
            deepseek_processor: DeepSeekOCR instance or callable that processes images
                              Should accept image_path (str) and return dict with 'full_text' and 'structured_data'
        
        Returns:
            Dictionary with format:
            {
                'blocks': List[Dict],  # Text blocks with bbox and confidence
                'engine_used': str,  # 'simple' or 'deepseek'
                'complexity_analysis': ComplexityResult,
                'processing_time': float,
                'stats': Dict  # Current statistics
            }
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        try:
            # Step 1: Analyze complexity (unless engine is forced)
            complexity_result = None
            use_simple = False
            
            if self.force_engine == 'simple':
                use_simple = True
                logger.info("Forced to use Simple OCR engine")
            elif self.force_engine == 'deepseek':
                use_simple = False
                logger.info("Forced to use DeepSeek OCR engine")
            else:
                # Analyze complexity
                logger.info("Analyzing document complexity...")
                complexity_result = self.analyzer.analyze(image)
                self.stats['complexity_analyses'].append({
                    'complexity': complexity_result.complexity,
                    'confidence': complexity_result.confidence,
                    'timestamp': datetime.now().isoformat()
                })
                
                # DETAILED LOGGING BEFORE DECISION
                logger.info(f"   📊 Complexity Analysis Results:")
                logger.info(f"      Complexity: {complexity_result.complexity}")
                logger.info(f"      Confidence: {complexity_result.confidence:.2f}")
                logger.info(f"      Recommended engine: {complexity_result.recommended_engine}")
                if complexity_result.reasons:
                    logger.info(f"      Reasons: {', '.join(complexity_result.reasons)}")
                
                # Decision logic: use simple if complexity is simple AND (confidence is high OR recommended_engine is simple)
                # Also check if simple_engine is available
                should_use_simple = (
                    complexity_result.complexity == 'simple' and
                    self.simple_engine is not None and
                    (complexity_result.confidence >= self.complexity_threshold or 
                     complexity_result.recommended_engine == 'simple')
                )
                
                if should_use_simple:
                    use_simple = True
                    logger.info(f"      → Routing to SIMPLE OCR (Fast) - complexity: {complexity_result.complexity}, confidence: {complexity_result.confidence:.2f}")
                else:
                    use_simple = False
                    if self.simple_engine is None:
                        logger.warning(f"      ⚠️  SimpleOCR not available! Routing to DeepSeek")
                    else:
                        logger.info(f"      → Routing to DeepSeek OCR (Complex) - complexity: {complexity_result.complexity}, confidence: {complexity_result.confidence:.2f}")
            
            # Step 2: Process with selected engine
            if use_simple and self.simple_engine is not None:
                # Use Simple OCR
                logger.info("Processing with Simple OCR (EasyOCR)...")
                result = self.simple_engine.process(image)
                engine_used = 'simple'
                self.stats['simple_calls'] += 1
                self.stats['simple_processing_time'] += result.get('processing_time', 0.0)
                
                # Convert SimpleOCR format to standard format
                blocks = result.get('blocks', [])
                
            else:
                # Use DeepSeek OCR
                if deepseek_processor is None:
                    raise ValueError(
                        "DeepSeek processor not provided. "
                        "Either provide deepseek_processor or ensure SimpleOCR is available."
                    )
                
                logger.info("Processing with DeepSeek OCR...")
                
                # DeepSeek expects file path, so save image temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image_path = tmp_file.name
                    image.save(image_path, 'PNG')
                
                try:
                    # Call DeepSeek processor
                    if callable(deepseek_processor):
                        deepseek_result = deepseek_processor(image_path)
                    else:
                        # Assume it's a DeepSeekOCR instance
                        deepseek_result = deepseek_processor.process(image_path)
                    
                    # Convert DeepSeek format to standard format
                    blocks = self._convert_deepseek_to_blocks(deepseek_result)
                    
                    processing_time = time.time() - start_time
                    engine_used = 'deepseek'
                    self.stats['deepseek_calls'] += 1
                    self.stats['deepseek_processing_time'] += processing_time
                    
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(image_path)
                    except:
                        pass
            
            total_processing_time = time.time() - start_time
            self.stats['total_processing_time'] += total_processing_time
            
            # Prepare result
            result = {
                'blocks': blocks,
                'engine_used': engine_used,
                'complexity_analysis': complexity_result,
                'processing_time': total_processing_time,
                'stats': self.get_stats()
            }
            
            logger.info(f"Processing completed in {total_processing_time:.2f}s using {engine_used} engine")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in HybridOCR processing: {e}", exc_info=True)
            raise Exception(f"HybridOCR processing failed: {str(e)}")
    
    def _convert_deepseek_to_blocks(self, deepseek_result: Dict) -> List[Dict]:
        """
        Convert DeepSeek OCR result to standard block format
        
        Args:
            deepseek_result: Result from DeepSeekOCR.process()
            
        Returns:
            List of blocks in standard format
        """
        blocks = []
        
        try:
            # DeepSeek returns structured_data with text and bbox
            structured_data = deepseek_result.get('structured_data', [])
            
            if structured_data:
                for item in structured_data:
                    text = item.get('text', '')
                    bbox = item.get('bbox', [0, 0, 0, 0])
                    confidence = item.get('confidence', 0.95)
                    
                    # Ensure bbox is [x1, y1, x2, y2] format
                    if len(bbox) == 4:
                        blocks.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': float(confidence)
                        })
            else:
                # Fallback: use full_text and create a single block
                full_text = deepseek_result.get('full_text', '')
                if full_text:
                    blocks.append({
                        'text': full_text,
                        'bbox': [0, 0, 0, 0],  # Unknown bbox
                        'confidence': 0.95
                    })
        
        except Exception as e:
            logger.warning(f"Error converting DeepSeek result: {e}")
            # Fallback: create block from full_text
            full_text = deepseek_result.get('full_text', '')
            if full_text:
                blocks.append({
                    'text': full_text,
                    'bbox': [0, 0, 0, 0],
                    'confidence': 0.95
                })
        
        return blocks
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['total_calls'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_calls']
        else:
            stats['avg_processing_time'] = 0.0
        
        if stats['simple_calls'] > 0:
            stats['avg_simple_time'] = stats['simple_processing_time'] / stats['simple_calls']
        else:
            stats['avg_simple_time'] = 0.0
        
        if stats['deepseek_calls'] > 0:
            stats['avg_deepseek_time'] = stats['deepseek_processing_time'] / stats['deepseek_calls']
        else:
            stats['avg_deepseek_time'] = 0.0
        
        # Calculate percentages
        if stats['total_calls'] > 0:
            stats['simple_percentage'] = (stats['simple_calls'] / stats['total_calls']) * 100
            stats['deepseek_percentage'] = (stats['deepseek_calls'] / stats['total_calls']) * 100
        else:
            stats['simple_percentage'] = 0.0
            stats['deepseek_percentage'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_calls': 0,
            'simple_calls': 0,
            'deepseek_calls': 0,
            'total_processing_time': 0.0,
            'simple_processing_time': 0.0,
            'deepseek_processing_time': 0.0,
            'complexity_analyses': []
        }
        logger.info("Statistics reset")

