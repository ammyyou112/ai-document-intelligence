"""
Enhanced OCR Pipeline

Complete document processing pipeline integrating:
- Hybrid OCR routing
- Document classification
- Metadata extraction
- Research paper structuring
"""
import logging
import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
import tempfile

# Configure logging
logger = logging.getLogger(__name__)

# Import components
try:
    from app.processors.hybrid_ocr_router import HybridOCRRouter
    from app.processors.document_classifier import DocumentClassifier
    from app.processors.metadata_extractor import MetadataExtractor
    from app.processors.research_paper_structurer import ResearchPaperStructurer
except ImportError:
    try:
        from processors.hybrid_ocr_router import HybridOCRRouter
        from processors.document_classifier import DocumentClassifier
        from processors.metadata_extractor import MetadataExtractor
        from processors.research_paper_structurer import ResearchPaperStructurer
    except ImportError:
        logger.error("Could not import pipeline components")
        raise


class EnhancedOCRPipeline:
    """Enhanced OCR pipeline with document intelligence"""
    
    def __init__(self, 
                 deepseek_processor: Optional[Any] = None,
                 complexity_threshold: float = 0.7,
                 pdf_dpi: int = 300):
        """
        Initialize enhanced OCR pipeline
        
        Args:
            deepseek_processor: DeepSeekOCR instance or callable for processing
            complexity_threshold: Threshold for routing decision (0.0-1.0)
            pdf_dpi: DPI for PDF to image conversion
        """
        logger.info("Initializing Enhanced OCR Pipeline...")
        
        # Initialize components
        self.router = HybridOCRRouter(complexity_threshold=complexity_threshold)
        self.classifier = DocumentClassifier()
        self.extractor = MetadataExtractor()
        self.structurer = ResearchPaperStructurer()
        self.deepseek_processor = deepseek_processor
        
        self.pdf_dpi = pdf_dpi
        
        logger.info("Enhanced OCR Pipeline initialized successfully")
    
    def process_document(self, 
                        file_path: str,
                        filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process complete document (PDF or image)
        
        Args:
            file_path: Path to document file
            filename: Original filename (optional)
            
        Returns:
            Comprehensive JSON result with all metadata and structure
        """
        document_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Processing document: {filename or file_path}")
        logger.info(f"Document ID: {document_id}")
        
        try:
            # Determine file type
            is_pdf = file_path.lower().endswith('.pdf')
            image_paths = []
            pdf_images_to_cleanup = []  # Track PDF images for cleanup
            
            # Convert PDF to images or load image
            if is_pdf:
                logger.info(f"Converting PDF to images at {self.pdf_dpi} DPI...")
                image_paths = self._pdf_to_images(file_path)
                pdf_images_to_cleanup = image_paths.copy()  # Track for cleanup
                logger.info(f"Converted PDF to {len(image_paths)} page(s)")
            else:
                image_paths = [file_path]
                logger.info("Processing single image file")
            
            # Process first page for analysis
            first_page_image = Image.open(image_paths[0])
            logger.info("Analyzing first page...")
            
            # Step 1: Process first page with hybrid router
            first_page_result = self.router.process(
                image=first_page_image,
                deepseek_processor=self.deepseek_processor
            )
            
            first_page_blocks = first_page_result['blocks']
            first_page_text = '\n'.join([block.get('text', '') for block in first_page_blocks])
            
            logger.info(f"First page processed: {len(first_page_blocks)} blocks, {len(first_page_text)} chars")
            logger.info(f"First page engine: {first_page_result['engine_used']}")
            
            # Step 2: Classify document type
            logger.info("Classifying document type...")
            classification = self.classifier.classify(first_page_text, first_page_blocks)
            logger.info(f"Document classified as: {classification.type} (confidence: {classification.confidence:.2f})")
            
            # Step 3: Extract metadata from first page
            logger.info("Extracting metadata from first page...")
            metadata = self.extractor.extract(first_page_text, first_page_blocks)
            logger.info(f"Metadata extracted: title={bool(metadata.title)}, authors={len(metadata.authors)}, date={bool(metadata.date)}")
            
            # Step 4: Process all pages
            logger.info(f"Processing all {len(image_paths)} page(s)...")
            all_pages = []
            all_blocks = []
            engines_used = set()
            total_words = 0
            has_tables = False
            
            for page_num, image_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {page_num}/{len(image_paths)}...")
                
                page_image = Image.open(image_path)
                page_result = self.router.process(
                    image=page_image,
                    deepseek_processor=self.deepseek_processor
                )
                
                page_blocks = page_result['blocks']
                page_text = '\n'.join([block.get('text', '') for block in page_blocks])
                page_words = len(page_text.split())
                total_words += page_words
                
                engines_used.add(page_result['engine_used'])
                
                # Check for tables (simple heuristic: look for table-related keywords)
                if any('table' in block.get('text', '').lower() for block in page_blocks):
                    has_tables = True
                
                # Get complexity analysis if available
                complexity_analysis = page_result.get('complexity_analysis')
                complexity_info = None
                if complexity_analysis:
                    complexity_info = {
                        'complexity': complexity_analysis.complexity,
                        'confidence': complexity_analysis.confidence,
                        'reasons': complexity_analysis.reasons[:3]  # Top 3 reasons
                    }
                
                page_data = {
                    'page_number': page_num,
                    'blocks': page_blocks,
                    'text': page_text,
                    'word_count': page_words,
                    'engine_used': page_result['engine_used'],
                    'processing_time': page_result['processing_time'],
                    'complexity': complexity_info
                }
                
                all_pages.append(page_data)
                all_blocks.extend(page_blocks)
                
                logger.info(f"Page {page_num} processed: {len(page_blocks)} blocks, {page_words} words, engine: {page_result['engine_used']}")
            
            # Step 5: Structure research paper if applicable
            document_structure = None
            if classification.type == 'research_paper':
                logger.info("Structuring research paper...")
                full_text = '\n'.join([page['text'] for page in all_pages])
                structure_result = self.structurer.structure(full_text)
                document_structure = self.structurer.to_dict(structure_result)
                logger.info(f"Research paper structured: {structure_result.total_sections} sections, {structure_result.total_words} words")
            else:
                logger.info(f"Skipping structuring (document type: {classification.type})")
            
            # Step 6: Calculate quality score
            quality_score = self._calculate_quality_score(
                all_blocks,
                classification,
                metadata,
                has_tables
            )
            
            # Step 7: Get router statistics
            router_stats = self.router.get_stats()
            
            # Step 8: Build final result
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                'document_metadata': {
                    'document_id': document_id,
                    'filename': filename or os.path.basename(file_path),
                    'type': classification.type,
                    'engines_used': list(engines_used),
                    'processed_date': end_time.isoformat(),
                    'processing_time_seconds': processing_time,
                    'total_pages': len(image_paths)
                },
                'content_metadata': {
                    'title': metadata.title,
                    'authors': metadata.authors,
                    'date': metadata.date,
                    'organization': metadata.organization,
                    'document_number': metadata.document_number,
                    'pages': len(image_paths),
                    'words': total_words,
                    'classification_confidence': classification.confidence
                },
                'document_structure': document_structure,
                'pages': all_pages,
                'training_annotations': {
                    'quality_score': quality_score,
                    'has_tables': has_tables,
                    'engines_distribution': {
                        'simple': router_stats.get('simple_calls', 0),
                        'deepseek': router_stats.get('deepseek_calls', 0),
                        'simple_percentage': router_stats.get('simple_percentage', 0.0),
                        'deepseek_percentage': router_stats.get('deepseek_percentage', 0.0)
                    },
                    'average_processing_time': router_stats.get('avg_processing_time', 0.0),
                    'total_blocks': len(all_blocks),
                    'document_type': classification.type
                },
                'classification_details': {
                    'type': classification.type,
                    'confidence': classification.confidence,
                    'keywords_found': classification.keywords_found[:10],  # Top 10
                    'patterns_matched': len(classification.patterns_matched)
                },
                'router_statistics': router_stats
            }
            
            logger.info(f"Document processing completed in {processing_time:.2f}s")
            logger.info(f"Result: {len(all_pages)} pages, {total_words} words, quality_score: {quality_score:.2f}")
            
            # Cleanup PDF images
            if pdf_images_to_cleanup:
                logger.info("Cleaning up PDF images...")
                for img_path in pdf_images_to_cleanup:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except Exception as cleanup_err:
                        logger.warning(f"Could not cleanup image {img_path}: {cleanup_err}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            # Cleanup on error
            if 'pdf_images_to_cleanup' in locals():
                for img_path in pdf_images_to_cleanup:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except:
                        pass
            raise Exception(f"Document processing failed: {str(e)}")
    
    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF to images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of image file paths
        """
        try:
            images = convert_from_path(pdf_path, dpi=self.pdf_dpi)
            image_paths = []
            
            # Save images in same directory as PDF (will be cleaned up later)
            pdf_dir = os.path.dirname(pdf_path)
            pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for i, image in enumerate(images):
                image_path = os.path.join(pdf_dir, f'{pdf_basename}_page_{i+1}.png')
                image.save(image_path, 'PNG')
                image_paths.append(image_path)
            
            logger.info(f"PDF converted to {len(image_paths)} image(s) at {self.pdf_dpi} DPI")
            return image_paths
            
        except Exception as e:
            error_msg = str(e)
            if 'poppler' in error_msg.lower() or 'pdfinfo' in error_msg.lower():
                raise Exception(
                    "PDF processing requires Poppler. "
                    "Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/ "
                    "and add to PATH."
                )
            raise Exception(f"PDF conversion error: {str(e)}")
    
    def _calculate_quality_score(self,
                                 blocks: List[Dict],
                                 classification: Any,
                                 metadata: Any,
                                 has_tables: bool) -> float:
        """
        Calculate overall quality score (0.0-1.0)
        
        Args:
            blocks: All text blocks
            classification: Classification result
            metadata: Metadata result
            has_tables: Whether document has tables
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Factor 1: Text extraction quality (30%)
        if blocks:
            avg_confidence = sum(block.get('confidence', 0.0) for block in blocks) / len(blocks)
            score += avg_confidence * 0.3
        else:
            score += 0.0
        
        # Factor 2: Classification confidence (20%)
        score += classification.confidence * 0.2
        
        # Factor 3: Metadata completeness (25%)
        metadata_score = 0.0
        if metadata.title:
            metadata_score += 0.3
        if metadata.authors:
            metadata_score += 0.3
        if metadata.date:
            metadata_score += 0.2
        if metadata.organization:
            metadata_score += 0.1
        if metadata.document_number:
            metadata_score += 0.1
        score += metadata_score * 0.25
        
        # Factor 4: Document structure (15%)
        if classification.type == 'research_paper':
            # Research papers should have structure
            score += 0.15
        else:
            # Other documents get partial credit
            score += 0.05
        
        # Factor 5: Content richness (10%)
        if has_tables:
            score += 0.05
        if len(blocks) > 50:
            score += 0.05
        
        return min(1.0, score)

