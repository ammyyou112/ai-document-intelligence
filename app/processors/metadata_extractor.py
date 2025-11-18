"""
Metadata Extractor

Extracts metadata from document first page:
- Title
- Authors
- Date
- Organization
- Document Number
"""
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MetadataResult:
    """Extracted metadata"""
    title: Optional[str]
    authors: List[str]
    date: Optional[str]
    organization: Optional[str]
    document_number: Optional[str]
    raw_metadata: Dict[str, any]  # Additional extracted fields


class MetadataExtractor:
    """Extracts metadata from document first page"""
    
    def __init__(self):
        """Initialize extractor with regex patterns"""
        # Date patterns
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY, DD/MM/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',  # DD Month YYYY
            r'\b\d{4}\b'  # Just year
        ]
        
        # Author patterns
        self.author_patterns = [
            r'(?:author|authors?|by|written\s+by)\s*[:\-]?\s*(.+?)(?:\n|$)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s*,\s*([A-Z][a-z]+\s+[A-Z][a-z]+))*',  # Name patterns
            r'([A-Z]\.\s*[A-Z][a-z]+\s+[A-Z][a-z]+)',  # Initial. Lastname
            r'([A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+)',  # Firstname M. Lastname
        ]
        
        # Organization patterns
        self.org_patterns = [
            r'\b(?:university|college|institute|institution|corporation|corp|inc|ltd|llc)\b',
            r'\b(?:department|dept|school|faculty|division)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][A-Za-z\s&]+)\s+(?:university|college|institute|institution)',
            r'\b(?:nasa|ieee|acm|springer|elsevier|arxiv|mit|stanford|harvard|mit|caltech)\b',
            r'@\s*([a-z0-9.-]+\.[a-z]{2,})',  # Email domain
        ]
        
        # Document number patterns
        self.doc_number_patterns = [
            r'\b(?:document|doc|report|ref|reference|id|number|#)\s*(?:number|#|no\.?)\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'\b(?:invoice|po|purchase\s+order)\s*(?:number|#|no\.?)\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'\b(?:doi|issn|isbn)\s*[:\-]?\s*([0-9.\-X]+)',
            r'\b(?:form|filing)\s+([0-9]+[A-Z]?-[A-Z]?)',  # Form 10-K, etc.
            r'\b([A-Z]{2,}\d{4,})',  # Alphanumeric codes
        ]
        
        # Title patterns (usually first large text block)
        self.title_indicators = [
            r'^([A-Z][^.!?]{10,200})(?:\n|$)',  # First long sentence
            r'(?:title|subject)\s*[:\-]?\s*(.+?)(?:\n|$)',
        ]
    
    def extract(self, text: str, blocks: Optional[List[Dict]] = None) -> MetadataResult:
        """
        Extract metadata from document text (typically first page)
        
        Args:
            text: Text content (preferably from first page)
            blocks: Optional list of text blocks with bbox for spatial analysis
            
        Returns:
            MetadataResult with extracted fields
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for metadata extraction")
                return MetadataResult(
                    title=None,
                    authors=[],
                    date=None,
                    organization=None,
                    document_number=None,
                    raw_metadata={}
                )
            
            # Extract each metadata field
            title = self._extract_title(text, blocks)
            authors = self._extract_authors(text)
            date = self._extract_date(text)
            organization = self._extract_organization(text)
            document_number = self._extract_document_number(text)
            
            # Additional raw metadata
            raw_metadata = {
                'text_length': len(text),
                'first_100_chars': text[:100],
                'line_count': len(text.split('\n'))
            }
            
            logger.info(f"Extracted metadata: title={bool(title)}, authors={len(authors)}, date={bool(date)}, org={bool(organization)}, doc_num={bool(document_number)}")
            
            return MetadataResult(
                title=title,
                authors=authors,
                date=date,
                organization=organization,
                document_number=document_number,
                raw_metadata=raw_metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}", exc_info=True)
            return MetadataResult(
                title=None,
                authors=[],
                date=None,
                organization=None,
                document_number=None,
                raw_metadata={'error': str(e)}
            )
    
    def _extract_title(self, text: str, blocks: Optional[List[Dict]] = None) -> Optional[str]:
        """Extract document title"""
        try:
            # Method 1: Use spatial analysis if blocks available
            if blocks:
                # Title is usually the largest text block at the top
                top_blocks = sorted(blocks, key=lambda b: b.get('bbox', [0, 0, 0, 0])[1])[:5]  # Top 5 blocks
                if top_blocks:
                    # Find the longest text block near the top
                    top_block = max(top_blocks, key=lambda b: len(b.get('text', '')))
                    title_text = top_block.get('text', '').strip()
                    if len(title_text) > 10 and len(title_text) < 200:
                        return title_text
            
            # Method 2: Use regex patterns
            lines = text.split('\n')
            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    # Check if it looks like a title (starts with capital, no ending punctuation)
                    if line[0].isupper() and not line.endswith(('.', '!', '?')):
                        # Check for title indicators
                        for pattern in self.title_indicators:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                title = match.group(1).strip()
                                if len(title) > 10:
                                    return title
                        # If no pattern match but looks like title, return it
                        if not any(c in line for c in [':', '-', '|']):  # Avoid headers
                            return line
            
            # Method 3: First substantial line
            for line in lines[:10]:
                line = line.strip()
                if 15 < len(line) < 150 and line[0].isupper():
                    return line
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting title: {e}")
            return None
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names"""
        authors = []
        try:
            # Look for author patterns
            for pattern in self.author_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    author_text = match.group(1).strip()
                    # Clean up author text
                    author_text = re.sub(r'^\s*(?:author|authors?|by|written\s+by)\s*[:\-]?\s*', '', author_text, flags=re.IGNORECASE)
                    
                    # Split multiple authors (comma, semicolon, or "and")
                    author_list = re.split(r'[,;]|\s+and\s+', author_text)
                    for author in author_list:
                        author = author.strip()
                        # Validate author name (at least first and last name)
                        if len(author) > 5 and re.match(r'^[A-Z][a-z]+', author):
                            if author not in authors:
                                authors.append(author)
            
            # Also look for common name patterns in first few lines
            lines = text.split('\n')[:15]
            for line in lines:
                # Pattern: Firstname Lastname or F. Lastname
                name_matches = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+|[A-Z]\.\s+[A-Z][a-z]+)\b', line)
                for name in name_matches:
                    name = name.strip()
                    if 5 < len(name) < 50 and name not in authors:
                        authors.append(name)
            
            # Limit to reasonable number
            return authors[:10]
            
        except Exception as e:
            logger.warning(f"Error extracting authors: {e}")
            return []
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date"""
        try:
            dates_found = []
            
            # Try all date patterns
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    date_str = match.group(0)
                    dates_found.append(date_str)
            
            # Prefer dates in first part of document
            lines = text.split('\n')[:30]
            for line in lines:
                for pattern in self.date_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(0)
            
            # Return first date found if any
            if dates_found:
                return dates_found[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting date: {e}")
            return None
    
    def _extract_organization(self, text: str) -> Optional[str]:
        """Extract organization name"""
        try:
            # Look for organization patterns
            for pattern in self.org_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    org_text = match.group(0) if match.groups() == () else match.group(1)
                    if org_text:
                        org_text = org_text.strip()
                        # Clean up
                        org_text = re.sub(r'\s+', ' ', org_text)
                        if len(org_text) > 3 and len(org_text) < 100:
                            return org_text
            
            # Also check first few lines for organization names
            lines = text.split('\n')[:20]
            for line in lines:
                # Look for common organization keywords
                if re.search(r'\b(?:university|college|institute|corporation|company|inc|ltd)\b', line, re.IGNORECASE):
                    # Extract the organization name
                    org_match = re.search(r'([A-Z][A-Za-z\s&]+(?:university|college|institute|corporation|company|inc|ltd))', line, re.IGNORECASE)
                    if org_match:
                        return org_match.group(1).strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting organization: {e}")
            return None
    
    def _extract_document_number(self, text: str) -> Optional[str]:
        """Extract document number"""
        try:
            # Try all document number patterns
            for pattern in self.doc_number_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    doc_num = match.group(1) if match.groups() else match.group(0)
                    if doc_num:
                        doc_num = doc_num.strip()
                        # Clean up
                        doc_num = re.sub(r'[:\-]\s*', '', doc_num)
                        if len(doc_num) > 2 and len(doc_num) < 50:
                            return doc_num
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting document number: {e}")
            return None

