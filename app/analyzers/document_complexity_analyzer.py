"""
Document Complexity Analyzer

Analyzes document images to determine complexity and recommend appropriate OCR engine.
Uses OpenCV for image quality analysis, layout detection, and content type identification.
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
    """Analyzes document complexity to determine optimal OCR engine"""
    
    def __init__(self, 
                 sharpness_threshold: float = 100.0,
                 noise_threshold: float = 0.1,
                 table_line_threshold: int = 10,
                 column_threshold: float = 0.3):
        """
        Initialize the analyzer with configurable thresholds
        
        Args:
            sharpness_threshold: Minimum Laplacian variance for sharp image
            noise_threshold: Maximum noise ratio for clean image
            table_line_threshold: Minimum lines detected to consider as table
            column_threshold: Minimum column ratio to consider multi-column layout
        """
        self.sharpness_threshold = sharpness_threshold
        self.noise_threshold = noise_threshold
        self.table_line_threshold = table_line_threshold
        self.column_threshold = column_threshold
        
    def analyze(self, image: Image.Image) -> ComplexityResult:
        """
        Analyze document complexity from PIL Image
        
        Args:
            image: PIL Image object
            
        Returns:
            ComplexityResult with analysis results
        """
        try:
            # Convert PIL Image to OpenCV format
            cv_image = self._pil_to_cv2(image)
            
            # Analyze different aspects
            quality_metrics = self._analyze_image_quality(cv_image)
            layout_metrics = self._analyze_layout_complexity(cv_image)
            content_metrics = self._analyze_content_type(cv_image)
            
            # Combine metrics
            all_metrics = {**quality_metrics, **layout_metrics, **content_metrics}
            
            # Determine complexity
            complexity_score = self._calculate_complexity_score(all_metrics)
            complexity = 'complex' if complexity_score > 0.5 else 'simple'
            confidence = abs(complexity_score - 0.5) * 2  # Convert to 0-1 confidence
            
            # Generate reasons
            reasons = self._generate_reasons(all_metrics, complexity_score)
            
            # Recommend engine
            recommended_engine = 'deepseek' if complexity == 'complex' else 'simple'
            
            logger.info(f"Complexity analysis: {complexity} (confidence: {confidence:.2f})")
            logger.debug(f"Metrics: {all_metrics}")
            logger.debug(f"Reasons: {reasons}")
            
            return ComplexityResult(
                complexity=complexity,
                confidence=confidence,
                reasons=reasons,
                recommended_engine=recommended_engine,
                metrics=all_metrics
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
    
    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        try:
            # Convert PIL to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            np_image = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            
            return cv_image
        except Exception as e:
            logger.error(f"Error converting PIL to OpenCV: {e}")
            raise
    
    def _analyze_image_quality(self, cv_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image quality metrics
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # 1. Sharpness analysis using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = laplacian_var
            metrics['is_sharp'] = 1.0 if laplacian_var >= self.sharpness_threshold else 0.0
            
            # 2. DPI estimation (approximate based on image dimensions)
            # Assume standard document size (8.5x11 inches)
            standard_width_inches = 8.5
            standard_height_inches = 11.0
            estimated_dpi_w = width / standard_width_inches
            estimated_dpi_h = height / standard_height_inches
            estimated_dpi = (estimated_dpi_w + estimated_dpi_h) / 2.0
            metrics['estimated_dpi'] = estimated_dpi
            metrics['is_high_dpi'] = 1.0 if estimated_dpi >= 300 else 0.0
            
            # 3. Noise analysis (using standard deviation of Laplacian)
            noise_level = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            normalized_noise = noise_level / 255.0
            metrics['noise_level'] = normalized_noise
            metrics['is_clean'] = 1.0 if normalized_noise <= self.noise_threshold else 0.0
            
            # 4. Contrast analysis
            contrast = gray.std()
            normalized_contrast = contrast / 255.0
            metrics['contrast'] = normalized_contrast
            metrics['has_good_contrast'] = 1.0 if normalized_contrast >= 0.3 else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            metrics = {
                'sharpness': 0.0,
                'is_sharp': 0.0,
                'estimated_dpi': 0.0,
                'is_high_dpi': 0.0,
                'noise_level': 1.0,
                'is_clean': 0.0,
                'contrast': 0.0,
                'has_good_contrast': 0.0
            }
        
        return metrics
    
    def _analyze_layout_complexity(self, cv_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze layout complexity (columns, regions, etc.)
        
        Returns:
            Dictionary with layout metrics
        """
        metrics = {}
        
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # 1. Column detection using horizontal projection
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Horizontal projection (sum of white pixels per column)
            horizontal_projection = np.sum(binary, axis=0)
            
            # Find peaks (columns)
            threshold = np.max(horizontal_projection) * 0.1
            peaks = []
            in_peak = False
            peak_start = 0
            
            for i, value in enumerate(horizontal_projection):
                if value > threshold and not in_peak:
                    in_peak = True
                    peak_start = i
                elif value <= threshold and in_peak:
                    in_peak = False
                    peaks.append((peak_start, i))
            
            if in_peak:
                peaks.append((peak_start, len(horizontal_projection) - 1))
            
            num_columns = len(peaks)
            column_ratio = sum(end - start for start, end in peaks) / width if peaks else 0.0
            
            metrics['num_columns'] = float(num_columns)
            metrics['column_ratio'] = column_ratio
            metrics['is_multi_column'] = 1.0 if num_columns > 1 and column_ratio > self.column_threshold else 0.0
            
            # 2. Vertical projection for row detection
            vertical_projection = np.sum(binary, axis=1)
            threshold_v = np.max(vertical_projection) * 0.1
            peaks_v = []
            in_peak = False
            peak_start = 0
            
            for i, value in enumerate(vertical_projection):
                if value > threshold_v and not in_peak:
                    in_peak = True
                    peak_start = i
                elif value <= threshold_v and in_peak:
                    in_peak = False
                    peaks_v.append((peak_start, i))
            
            if in_peak:
                peaks_v.append((peak_start, len(vertical_projection) - 1))
            
            num_rows = len(peaks_v)
            metrics['num_rows'] = float(num_rows)
            metrics['layout_density'] = (num_columns * num_rows) / (width * height / 10000.0)  # Normalized
            
            # 3. Text region detection
            # Use connected components to find text regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            text_regions = num_labels - 1  # Exclude background
            metrics['num_text_regions'] = float(text_regions)
            metrics['region_density'] = text_regions / (width * height / 10000.0)  # Normalized
            
        except Exception as e:
            logger.error(f"Error analyzing layout complexity: {e}")
            metrics = {
                'num_columns': 1.0,
                'column_ratio': 0.0,
                'is_multi_column': 0.0,
                'num_rows': 0.0,
                'layout_density': 0.0,
                'num_text_regions': 0.0,
                'region_density': 0.0
            }
        
        return metrics
    
    def _analyze_content_type(self, cv_image: np.ndarray) -> Dict[str, float]:
        """
        Detect content types (tables, formulas, diagrams)
        
        Returns:
            Dictionary with content type metrics
        """
        metrics = {}
        
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # 1. Table detection using HoughLines
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect horizontal lines
            horizontal_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(width * 0.3))
            num_h_lines = len(horizontal_lines) if horizontal_lines is not None else 0
            
            # Detect vertical lines
            vertical_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(height * 0.3))
            num_v_lines = len(vertical_lines) if vertical_lines is not None else 0
            
            total_lines = num_h_lines + num_v_lines
            metrics['num_table_lines'] = float(total_lines)
            metrics['has_tables'] = 1.0 if total_lines >= self.table_line_threshold else 0.0
            
            # 2. Formula detection (look for mathematical symbols patterns)
            # This is a simplified heuristic - look for small isolated regions
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Count small regions (potential formula symbols)
            small_regions = 0
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if 5 < area < 100:  # Small but not noise
                    small_regions += 1
            
            metrics['num_small_regions'] = float(small_regions)
            metrics['has_formulas'] = 1.0 if small_regions > 50 else 0.0
            
            # 3. Diagram detection (look for large non-text regions)
            large_regions = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > (width * height * 0.01):  # Large region (>1% of image)
                    large_regions += 1
            
            metrics['num_large_regions'] = float(large_regions)
            metrics['has_diagrams'] = 1.0 if large_regions > 3 else 0.0
            
            # 4. Overall content complexity
            content_complexity = (
                metrics['has_tables'] * 0.4 +
                metrics['has_formulas'] * 0.3 +
                metrics['has_diagrams'] * 0.3
            )
            metrics['content_complexity'] = content_complexity
            
        except Exception as e:
            logger.error(f"Error analyzing content type: {e}")
            metrics = {
                'num_table_lines': 0.0,
                'has_tables': 0.0,
                'num_small_regions': 0.0,
                'has_formulas': 0.0,
                'num_large_regions': 0.0,
                'has_diagrams': 0.0,
                'content_complexity': 0.0
            }
        
        return metrics
    
    def _calculate_complexity_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall complexity score (0.0 = simple, 1.0 = complex)
        
        Args:
            metrics: Dictionary of all metrics
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        score = 0.0
        weights = {
            # Quality factors (lower quality = more complex)
            'is_sharp': -0.1,  # Negative: sharp images are simpler
            'is_clean': -0.1,
            'has_good_contrast': -0.1,
            
            # Layout factors
            'is_multi_column': 0.2,
            'layout_density': 0.15,
            'region_density': 0.1,
            
            # Content factors
            'has_tables': 0.25,
            'has_formulas': 0.15,
            'has_diagrams': 0.15,
            'content_complexity': 0.2
        }
        
        for key, weight in weights.items():
            value = metrics.get(key, 0.0)
            score += value * weight
        
        # Normalize to 0-1 range
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _generate_reasons(self, metrics: Dict[str, float], complexity_score: float) -> List[str]:
        """
        Generate human-readable reasons for complexity decision
        
        Args:
            metrics: Dictionary of all metrics
            complexity_score: Calculated complexity score
            
        Returns:
            List of reason strings
        """
        reasons = []
        
        # Quality reasons
        if metrics.get('is_sharp', 0.0) < 0.5:
            reasons.append("Low image sharpness detected")
        if metrics.get('is_clean', 0.0) < 0.5:
            reasons.append("High noise level detected")
        if metrics.get('has_good_contrast', 0.0) < 0.5:
            reasons.append("Low contrast detected")
        
        # Layout reasons
        if metrics.get('is_multi_column', 0.0) > 0.5:
            reasons.append(f"Multi-column layout detected ({int(metrics.get('num_columns', 0))} columns)")
        if metrics.get('layout_density', 0.0) > 0.5:
            reasons.append("Dense layout structure")
        
        # Content reasons
        if metrics.get('has_tables', 0.0) > 0.5:
            reasons.append(f"Tables detected ({int(metrics.get('num_table_lines', 0))} lines)")
        if metrics.get('has_formulas', 0.0) > 0.5:
            reasons.append("Mathematical formulas detected")
        if metrics.get('has_diagrams', 0.0) > 0.5:
            reasons.append("Diagrams/figures detected")
        
        # Default reason if none found
        if not reasons:
            if complexity_score > 0.5:
                reasons.append("Complex document structure")
            else:
                reasons.append("Simple document structure")
        
        return reasons

