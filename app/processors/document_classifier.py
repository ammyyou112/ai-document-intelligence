"""
Document Classifier

Classifies documents into different types based on text patterns and keywords.
Supports: research_paper, invoice, financial_report, technical_manual, general_document
"""
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of document classification"""
    type: str
    confidence: float
    keywords_found: List[str]
    patterns_matched: List[str]


class DocumentClassifier:
    """Classifies documents into different types"""
    
    def __init__(self):
        """Initialize classifier with pattern definitions"""
        # Define patterns for each document type
        self.patterns = {
            'research_paper': {
                'keywords': [
                    'abstract', 'introduction', 'methodology', 'method', 'results',
                    'discussion', 'conclusion', 'references', 'bibliography',
                    'acknowledgment', 'acknowledgement', 'appendix',
                    'ieee', 'acm', 'springer', 'elsevier', 'arxiv',
                    'doi:', 'doi ', 'issn', 'isbn',
                    'figure', 'table', 'equation', 'theorem', 'lemma',
                    'hypothesis', 'experiment', 'dataset', 'evaluation'
                ],
                'patterns': [
                    r'\babstract\b',
                    r'\bintroduction\b',
                    r'\breferences?\b',
                    r'\bdoi:\s*\d+\.\d+',
                    r'\bissn\s*[:=]\s*\d+',
                    r'figure\s+\d+',
                    r'table\s+\d+',
                    r'equation\s*\(?\d+\)?',
                    r'author\s*[:\-]',
                    r'keywords?\s*[:\-]'
                ],
                'weight': 1.0
            },
            'invoice': {
                'keywords': [
                    'invoice', 'bill', 'payment', 'due date', 'amount due',
                    'subtotal', 'tax', 'total', 'invoice number', 'invoice #',
                    'billing address', 'shipping address', 'purchase order',
                    'po number', 'item', 'quantity', 'unit price', 'line total',
                    'terms', 'net', 'discount', 'balance'
                ],
                'patterns': [
                    r'\binvoice\s*(?:number|#|no\.?)\s*[:=]?\s*\w+',
                    r'\bamount\s+due\b',
                    r'\bdue\s+date\b',
                    r'\btotal\s*[:\-]?\s*\$?\d+',
                    r'\bsubtotal\b',
                    r'\btax\s*[:\-]?\s*\$?\d+',
                    r'\bpurchase\s+order\b',
                    r'\bpo\s*(?:number|#)\s*[:=]?\s*\w+',
                    r'\bquantity\s*[:\-]',
                    r'\bunit\s+price\b'
                ],
                'weight': 1.0
            },
            'financial_report': {
                'keywords': [
                    'financial report', 'annual report', 'quarterly report',
                    'balance sheet', 'income statement', 'cash flow',
                    'revenue', 'expenses', 'profit', 'loss', 'assets',
                    'liabilities', 'equity', 'fiscal year', 'fiscal quarter',
                    'audited', 'auditor', 'sec filing', 'form 10-k', 'form 10-q',
                    'earnings', 'ebitda', 'net income', 'gross margin'
                ],
                'patterns': [
                    r'\bfinancial\s+report\b',
                    r'\bannual\s+report\b',
                    r'\bquarterly\s+report\b',
                    r'\bbalance\s+sheet\b',
                    r'\bincome\s+statement\b',
                    r'\bcash\s+flow\b',
                    r'\bfiscal\s+year\b',
                    r'\bfiscal\s+quarter\b',
                    r'\bform\s+10-[kq]\b',
                    r'\bsec\s+filing\b',
                    r'\baudited\b',
                    r'\bebitda\b'
                ],
                'weight': 1.0
            },
            'technical_manual': {
                'keywords': [
                    'manual', 'user guide', 'installation', 'configuration',
                    'troubleshooting', 'specifications', 'technical specification',
                    'api', 'sdk', 'version', 'release notes', 'changelog',
                    'system requirements', 'hardware requirements', 'software requirements',
                    'chapter', 'section', 'appendix', 'index', 'table of contents',
                    'procedure', 'step', 'warning', 'caution', 'note'
                ],
                'patterns': [
                    r'\b(?:user\s+)?(?:guide|manual)\b',
                    r'\binstallation\s+guide\b',
                    r'\btechnical\s+specification\b',
                    r'\bsystem\s+requirements\b',
                    r'\bhardware\s+requirements\b',
                    r'\bsoftware\s+requirements\b',
                    r'\btable\s+of\s+contents\b',
                    r'\bchapter\s+\d+',
                    r'\bsection\s+\d+',
                    r'\bversion\s+\d+',
                    r'\brelease\s+notes\b',
                    r'\bapi\s+reference\b'
                ],
                'weight': 1.0
            },
            'general_document': {
                'keywords': [],  # Fallback category
                'patterns': [],
                'weight': 0.1  # Lower weight as fallback
            }
        }
    
    def classify(self, text: str, blocks: Optional[List[Dict]] = None) -> ClassificationResult:
        """
        Classify document based on text content
        
        Args:
            text: Full text content of the document
            blocks: Optional list of text blocks with bbox (for spatial analysis)
            
        Returns:
            ClassificationResult with type, confidence, and matched patterns
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided, returning general_document")
                return ClassificationResult(
                    type='general_document',
                    confidence=0.5,
                    keywords_found=[],
                    patterns_matched=[]
                )
            
            # Normalize text for matching
            text_lower = text.lower()
            text_length = len(text)
            
            # Score each document type
            scores = {}
            all_keywords_found = {}
            all_patterns_matched = {}
            
            for doc_type, config in self.patterns.items():
                if doc_type == 'general_document':
                    continue  # Skip fallback in scoring
                
                score = 0.0
                keywords_found = []
                patterns_matched = []
                
                # Check keywords
                for keyword in config['keywords']:
                    keyword_lower = keyword.lower()
                    # Count occurrences (case-insensitive)
                    count = text_lower.count(keyword_lower)
                    if count > 0:
                        keywords_found.append(keyword)
                        # Weight by frequency and length
                        keyword_score = (count * len(keyword)) / max(text_length, 1) * 100
                        score += keyword_score
                
                # Check regex patterns
                for pattern in config['patterns']:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        patterns_matched.append(pattern)
                        # Weight by number of matches
                        pattern_score = len(matches) * 10
                        score += pattern_score
                
                # Apply type weight
                score *= config['weight']
                
                scores[doc_type] = score
                all_keywords_found[doc_type] = keywords_found
                all_patterns_matched[doc_type] = patterns_matched
            
            # Find best match
            if not scores:
                # No matches, return general_document
                return ClassificationResult(
                    type='general_document',
                    confidence=0.5,
                    keywords_found=[],
                    patterns_matched=[]
                )
            
            best_type = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[best_type]
            
            # Calculate confidence (normalize to 0-1)
            total_score = sum(scores.values())
            if total_score > 0:
                confidence = min(1.0, best_score / max(total_score, best_score * 2))
            else:
                confidence = 0.5
            
            # Ensure minimum confidence for clear matches
            if best_score > 50:  # Strong match
                confidence = max(confidence, 0.7)
            elif best_score > 20:  # Moderate match
                confidence = max(confidence, 0.5)
            
            logger.info(f"Document classified as: {best_type} (confidence: {confidence:.2f}, score: {best_score:.2f})")
            
            return ClassificationResult(
                type=best_type,
                confidence=confidence,
                keywords_found=all_keywords_found.get(best_type, []),
                patterns_matched=all_patterns_matched.get(best_type, [])
            )
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}", exc_info=True)
            # Return general_document on error
            return ClassificationResult(
                type='general_document',
                confidence=0.3,
                keywords_found=[],
                patterns_matched=[]
            )
    
    def classify_from_blocks(self, blocks: List[Dict]) -> ClassificationResult:
        """
        Classify document from OCR blocks
        
        Args:
            blocks: List of blocks with 'text' key
            
        Returns:
            ClassificationResult
        """
        try:
            # Combine all text from blocks
            text = '\n'.join([block.get('text', '') for block in blocks])
            return self.classify(text, blocks)
        except Exception as e:
            logger.error(f"Error classifying from blocks: {e}")
            return ClassificationResult(
                type='general_document',
                confidence=0.3,
                keywords_found=[],
                patterns_matched=[]
            )

