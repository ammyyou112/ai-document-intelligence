"""
Research Paper Structurer

Detects and structures research paper sections:
- Abstract
- Introduction
- Methodology
- Results
- Discussion
- Conclusion
- References
"""
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Document section"""
    name: str
    text: str
    word_count: int
    start_index: int
    end_index: int


@dataclass
class StructureResult:
    """Structured document result"""
    sections: Dict[str, Section]
    section_order: List[str]
    total_sections: int
    total_words: int


class ResearchPaperStructurer:
    """Structures research papers into sections"""
    
    def __init__(self):
        """Initialize structurer with section patterns"""
        # Define section patterns (case-insensitive)
        self.section_patterns = {
            'abstract': [
                r'^\s*abstract\s*$',
                r'^\s*abstract\s*[:\-]',
                r'^\s*summary\s*$',
            ],
            'introduction': [
                r'^\s*introduction\s*$',
                r'^\s*introduction\s*[:\-]',
                r'^\s*1\.\s*introduction\s*$',
                r'^\s*background\s*$',
            ],
            'methodology': [
                r'^\s*methodology\s*$',
                r'^\s*method\s*$',
                r'^\s*methods\s*$',
                r'^\s*methodology\s*[:\-]',
                r'^\s*method\s*[:\-]',
                r'^\s*methods\s*[:\-]',
                r'^\s*experimental\s+setup\s*$',
                r'^\s*experimental\s+method\s*$',
                r'^\s*approach\s*$',
            ],
            'results': [
                r'^\s*results\s*$',
                r'^\s*result\s*$',
                r'^\s*results\s*[:\-]',
                r'^\s*experimental\s+results\s*$',
                r'^\s*findings\s*$',
            ],
            'discussion': [
                r'^\s*discussion\s*$',
                r'^\s*discussion\s*[:\-]',
                r'^\s*analysis\s*$',
                r'^\s*analysis\s+and\s+discussion\s*$',
            ],
            'conclusion': [
                r'^\s*conclusion\s*$',
                r'^\s*conclusions\s*$',
                r'^\s*conclusion\s*[:\-]',
                r'^\s*conclusions\s*[:\-]',
                r'^\s*summary\s+and\s+conclusion\s*$',
                r'^\s*concluding\s+remarks\s*$',
            ],
            'references': [
                r'^\s*references\s*$',
                r'^\s*reference\s*$',
                r'^\s*references\s*[:\-]',
                r'^\s*bibliography\s*$',
                r'^\s*citations\s*$',
                r'^\s*works\s+cited\s*$',
            ],
            'acknowledgment': [
                r'^\s*acknowledgment\s*$',
                r'^\s*acknowledgement\s*$',
                r'^\s*acknowledgments\s*$',
                r'^\s*acknowledgements\s*$',
                r'^\s*acknowledgment\s*[:\-]',
            ],
            'appendix': [
                r'^\s*appendix\s*$',
                r'^\s*appendix\s+[a-z]\s*$',
                r'^\s*appendix\s+\d+\s*$',
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for section_name, patterns in self.section_patterns.items():
            self.compiled_patterns[section_name] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                for pattern in patterns
            ]
    
    def structure(self, text: str) -> StructureResult:
        """
        Structure research paper into sections
        
        Args:
            text: Full text of the research paper
            
        Returns:
            StructureResult with sections and metadata
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for structuring")
                return StructureResult(
                    sections={},
                    section_order=[],
                    total_sections=0,
                    total_words=0
                )
            
            # Split text into lines for section detection
            lines = text.split('\n')
            
            # Find section boundaries
            section_boundaries = self._find_section_boundaries(lines)
            
            # Extract sections
            sections = {}
            section_order = []
            
            for i, (section_name, start_idx, end_idx) in enumerate(section_boundaries):
                # Extract section text
                section_text = '\n'.join(lines[start_idx:end_idx]).strip()
                
                # Calculate word count
                word_count = len(section_text.split())
                
                # Create section object
                section = Section(
                    name=section_name,
                    text=section_text,
                    word_count=word_count,
                    start_index=start_idx,
                    end_index=end_idx
                )
                
                sections[section_name] = section
                section_order.append(section_name)
            
            # Calculate totals
            total_sections = len(sections)
            total_words = sum(section.word_count for section in sections.values())
            
            logger.info(f"Structured document into {total_sections} sections: {', '.join(section_order)}")
            
            return StructureResult(
                sections=sections,
                section_order=section_order,
                total_sections=total_sections,
                total_words=total_words
            )
            
        except Exception as e:
            logger.error(f"Error structuring document: {e}", exc_info=True)
            return StructureResult(
                sections={},
                section_order=[],
                total_sections=0,
                total_words=0
            )
    
    def _find_section_boundaries(self, lines: List[str]) -> List[tuple]:
        """
        Find section boundaries in text lines
        
        Args:
            lines: List of text lines
            
        Returns:
            List of tuples: (section_name, start_index, end_index)
        """
        boundaries = []
        
        # Track found sections
        found_sections = {}
        
        # Scan lines for section headers
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check each section pattern
            for section_name, patterns in self.compiled_patterns.items():
                # Skip if we already found this section
                if section_name in found_sections:
                    continue
                
                # Check if line matches any pattern for this section
                for pattern in patterns:
                    if pattern.match(line_stripped):
                        found_sections[section_name] = i
                        break
        
        # Sort sections by position
        sorted_sections = sorted(found_sections.items(), key=lambda x: x[1])
        
        # Create boundaries
        for idx, (section_name, start_idx) in enumerate(sorted_sections):
            # Determine end index (next section or end of text)
            if idx + 1 < len(sorted_sections):
                end_idx = sorted_sections[idx + 1][1]
            else:
                end_idx = len(lines)
            
            boundaries.append((section_name, start_idx, end_idx))
        
        return boundaries
    
    def get_section(self, structure_result: StructureResult, section_name: str) -> Optional[Section]:
        """
        Get a specific section from structure result
        
        Args:
            structure_result: StructureResult from structure() method
            section_name: Name of section to retrieve
            
        Returns:
            Section object or None if not found
        """
        return structure_result.sections.get(section_name)
    
    def get_section_text(self, structure_result: StructureResult, section_name: str) -> Optional[str]:
        """
        Get text of a specific section
        
        Args:
            structure_result: StructureResult from structure() method
            section_name: Name of section to retrieve
            
        Returns:
            Section text or None if not found
        """
        section = self.get_section(structure_result, section_name)
        return section.text if section else None
    
    def to_dict(self, structure_result: StructureResult) -> Dict[str, Dict]:
        """
        Convert structure result to dictionary format
        
        Args:
            structure_result: StructureResult to convert
            
        Returns:
            Dictionary with section_name: {text, word_count}
        """
        result = {}
        for section_name, section in structure_result.sections.items():
            result[section_name] = {
                'text': section.text,
                'word_count': section.word_count
            }
        return result

