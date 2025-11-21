"""
Document Complexity Analyzer

Robust, general-purpose document complexity analyzer using objective criteria.
Works on any document type without tuning to specific examples.

Design principles:
- Uses objective, measurable criteria (not tuned to specific documents)
- Multiple independent signals (image quality, tables, diagrams, layout)
- Clear decision boundaries
- Conservative approach: when in doubt, mark as complex (use DeepSeek)
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ComplexityResult:
    """Result of complexity analysis"""
    complexity: str  # 'simple' or 'complex'
    confidence: float  # 0.0 to 1.0
    reasons: List[str]  # List of reasons for the decision
    recommended_engine: str  # 'simple' or 'deepseek'
    metrics: Dict[str, float]  # Detailed metrics


class DocumentComplexityAnalyzer:
    """
    Robust document complexity analyzer using objective criteria.
    
    Works on any document type without tuning to specific examples.
    Uses multiple independent signals to make conservative decisions.
    """
    
    def __init__(self):
        """
        Initialize the analyzer with objective, general-purpose thresholds.
        No tuning needed - works on any document type.
        """
        pass
        
    def analyze(self, image: Image.Image) -> ComplexityResult:
        """
        Analyze document complexity using multiple independent signals.
        
        Args:
            image: PIL Image object
            
        Returns:
            ComplexityResult with analysis results
        """
        try:
            # Convert to grayscale numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
            else:
                img_array = image
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Multiple independent complexity signals
            signals = {
                'image_quality': self._check_image_quality(gray),
                'has_tables': self._detect_tables(gray),
                'has_diagrams': self._detect_diagrams(gray),
                'layout_complexity': self._analyze_layout(gray)
            }
            
            # Make decision
            decision = self._make_decision(signals)
            
            # Convert to ComplexityResult format
            metrics = {
                'sharpness': signals['image_quality']['sharpness'],
                'line_count': signals['has_tables']['line_count'],
                'large_regions': signals['has_diagrams']['large_regions'],
                'density': signals['layout_complexity']['density'],
                'columns': signals['layout_complexity']['columns'],
                'has_tables': 1.0 if signals['has_tables']['has_tables'] else 0.0,
                'has_diagrams': 1.0 if signals['has_diagrams']['has_diagrams'] else 0.0,
                'is_multi_column': 1.0 if signals['layout_complexity']['is_multi_column'] else 0.0,
                'is_poor_quality': 1.0 if signals['image_quality']['is_poor'] else 0.0
            }
            
            return ComplexityResult(
                complexity=decision['complexity'],
                confidence=decision['confidence'],
                reasons=decision['reasons'],
                recommended_engine=decision['recommended_engine'],
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing document complexity: {e}", exc_info=True)
            # Default to complex on error (safer)
            return ComplexityResult(
                complexity='complex',
                confidence=0.5,
                reasons=[f"Analysis error: {str(e)}"],
                recommended_engine='deepseek',
                metrics={}
            )
    
    def _check_image_quality(self, gray: np.ndarray) -> Dict:
        """
        Check if image quality is poor (needs advanced OCR).
        
        Uses objective measure: Laplacian variance for sharpness.
        """
        # Sharpness via Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Poor quality if very blurry (objective threshold)
        is_poor = sharpness < 50
        
        return {
            'is_poor': is_poor,
            'sharpness': float(sharpness),
            'reason': 'Poor image quality' if is_poor else None
        }
    
    def _detect_tables(self, gray: np.ndarray) -> Dict:
        """
        Detect tables via line detection - OBJECTIVE measure.
        
        Tables have many straight lines. Uses Hough line detection.
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=50, minLineLength=50, maxLineGap=10)
        
        line_count = 0 if lines is None else len(lines)
        
        # OBJECTIVE: Tables have many straight lines (>30)
        has_tables = line_count > 30
        
        return {
            'has_tables': has_tables,
            'line_count': line_count,
            'reason': f'Contains tables ({line_count} lines)' if has_tables else None
        }
    
    def _detect_diagrams(self, gray: np.ndarray) -> Dict:
        """
        Detect diagrams/images via large contiguous regions.
        
        Uses connected components to find large regions (not individual characters).
        """
        _, binary = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Count LARGE regions (diagrams, not individual characters)
        large_regions = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 2000:  # OBJECTIVE: Large = likely diagram/image
                large_regions += 1
        
        # OBJECTIVE: 3+ large regions = has diagrams
        has_diagrams = large_regions >= 3
        
        return {
            'has_diagrams': has_diagrams,
            'large_regions': large_regions,
            'reason': f'Contains diagrams ({large_regions} large regions)' if has_diagrams else None
        }
    
    def _analyze_layout(self, gray: np.ndarray) -> Dict:
        """
        Analyze layout complexity via pixel distribution.
        
        Detects multi-column layouts and dense content.
        """
        # Calculate darkness density
        total_pixels = gray.size
        dark_pixels = np.sum(gray < 128)
        density = dark_pixels / total_pixels
        
        # OBJECTIVE: Very dense (>25%) = complex layout
        is_dense = density > 0.25
        
        # Detect multi-column via projection analysis
        h, w = gray.shape
        vertical_projection = np.sum(gray < 128, axis=0)  # Sum along columns
        
        # Find gaps (columns separated by whitespace)
        threshold = h * 0.1  # Gaps with <10% darkness
        gaps = vertical_projection < threshold
        
        # Count transitions (gap → text → gap = column)
        transitions = np.diff(gaps.astype(int))
        column_count = np.sum(transitions > 0)
        
        # OBJECTIVE: 2+ columns = complex layout
        is_multi_column = column_count >= 2
        
        return {
            'is_dense': is_dense,
            'is_multi_column': is_multi_column,
            'density': float(density),
            'columns': int(column_count),
            'reason': f'Multi-column layout ({column_count} columns)' if is_multi_column else None
        }
    
    def _make_decision(self, signals: Dict) -> Dict:
        """
        Make complexity decision based on objective signals.
        
        CONSERVATIVE: If ANY signal indicates complexity → use DeepSeek
        """
        reasons = []
        
        # Check all signals
        if signals['image_quality']['is_poor']:
            reasons.append(signals['image_quality']['reason'])
        
        if signals['has_tables']['has_tables']:
            reasons.append(signals['has_tables']['reason'])
        
        if signals['has_diagrams']['has_diagrams']:
            reasons.append(signals['has_diagrams']['reason'])
        
        if signals['layout_complexity']['is_multi_column']:
            reasons.append(signals['layout_complexity']['reason'])
        
        # Decision: Complex if ANY reason found
        is_complex = len(reasons) > 0
        
        if is_complex:
            complexity = 'complex'
            engine = 'deepseek'
            confidence = min(1.0, 0.5 + len(reasons) * 0.15)  # More reasons = higher confidence
        else:
            complexity = 'simple'
            engine = 'simple'
            confidence = 0.95
            reasons = ['Clean text document', 'No tables', 'No diagrams', 'Simple layout']
        
        # Logging
        logger.info(f"📊 Complexity Analysis:")
        logger.info(f"   Image quality: sharpness={signals['image_quality']['sharpness']:.1f}")
        logger.info(f"   Tables: {signals['has_tables']['line_count']} lines detected")
        logger.info(f"   Diagrams: {signals['has_diagrams']['large_regions']} large regions")
        logger.info(f"   Layout: density={signals['layout_complexity']['density']:.2%}, "
                   f"columns={signals['layout_complexity']['columns']}")
        logger.info(f"   → Decision: {complexity} (confidence: {confidence:.2f})")
        logger.info(f"   → Engine: {engine}")
        logger.info(f"   → Reasons: {', '.join(reasons[:3])}")  # Show first 3
        
        return {
            'complexity': complexity,
            'confidence': confidence,
            'recommended_engine': engine,
            'reasons': reasons
        }
