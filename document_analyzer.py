"""
Document Structure Analyzer
Extracts hierarchical structure, tables, figures, and creates structured JSON for AI training
"""
import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

class DocumentAnalyzer:
    """Analyzes OCR output to extract document structure"""
    
    def __init__(self):
        self.heading_patterns = [
            r'^#+\s+(.+)$',  # Markdown headings
            r'^(\d+\.?\s+[A-Z][^\n]+)$',  # Numbered headings
            r'^([A-Z][A-Z\s]{3,})$',  # ALL CAPS headings
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)$',  # Title Case headings with colon
        ]
        # Common regexes for type-specific extraction
        self.invoice_patterns = {
            'invoice_no': re.compile(r'(invoice\s*(?:no\.?|number)[:\-\s]*)([A-Z0-9\-]+)', re.IGNORECASE),
            'date': re.compile(r'(date)[:\-\s]*([\d]{1,2}[\/\-\.\s][\d]{1,2}[\/\-\.\s][\d]{2,4})', re.IGNORECASE),
            'total': re.compile(r'(total\s*amount|grand\s*total|total)[:\-\s]*([$€£]?\s?[\d\.,]+)', re.IGNORECASE),
            'subtotal': re.compile(r'(subtotal)[:\-\s]*([$€£]?\s?[\d\.,]+)', re.IGNORECASE),
            'tax': re.compile(r'(tax|vat)[:\-\s]*([$€£]?\s?[\d\.,]+)', re.IGNORECASE),
            'currency': re.compile(r'([$€£])\s?[\d\.,]+')
        }
        self.academic_patterns = {
            'title': re.compile(r'^(?:#\s*)?(.{5,120})$', re.MULTILINE),
            'abstract': re.compile(r'\babstract\b[:\s\-]*\n?(.{50,2000})', re.IGNORECASE | re.DOTALL),
            'authors': re.compile(r'by\s+([A-Z][A-Za-z\-\s,]+)\n', re.IGNORECASE),
            'references': re.compile(r'(?:(?:references|bibliography)\s*\n)(.+)$', re.IGNORECASE | re.DOTALL)
        }
        self.form_patterns = {
            'field_line': re.compile(r'^\s*([A-Za-z][A-Za-z0-9 _\-\/\.]{2,40})\s*[:\-]\s*(.+?)\s*$', re.MULTILINE)
        }
    
    def detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type based on content and filename"""
        filename_lower = filename.lower()
        
        # Check filename extensions
        if filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return 'image'
        
        # Analyze content patterns
        text_lower = text.lower()
        
        # Academic paper indicators
        if any(keyword in text_lower for keyword in ['abstract', 'introduction', 'methodology', 'references', 'bibliography']):
            return 'academic_paper'
        
        # Report indicators
        if any(keyword in text_lower for keyword in ['executive summary', 'table of contents', 'appendix']):
            return 'report'
        
        # Invoice/receipt indicators
        if any(keyword in text_lower for keyword in ['invoice', 'receipt', 'total', 'subtotal', 'tax']):
            return 'invoice'
        
        # Form indicators
        if any(keyword in text_lower for keyword in ['form', 'application', 'please fill', 'signature']):
            return 'form'
        
        # Letter/document indicators
        if any(keyword in text_lower for keyword in ['dear', 'sincerely', 'yours truly']):
            return 'letter'
        
        # Default
        return 'document'
    
    def extract_hierarchical_structure(self, text: str) -> Dict[str, Any]:
        """Extract hierarchical document structure (headings, sections, subsections)"""
        structure = {
            'title': None,
            'sections': [],
            'hierarchy_level': 0
        }
        
        lines = text.split('\n')
        current_section = None
        current_subsection = None
        section_stack = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Detect markdown headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                # Create section entry
                section = {
                    'level': level,
                    'heading': heading_text,
                    'content': [],
                    'subsections': []
                }
                
                # Manage hierarchy
                while section_stack and section_stack[-1]['level'] >= level:
                    section_stack.pop()
                
                if section_stack:
                    section_stack[-1]['subsections'].append(section)
                else:
                    structure['sections'].append(section)
                
                section_stack.append(section)
                current_section = section
                continue
            
            # Detect numbered headings (1., 1.1, 1.1.1, etc.)
            numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]+)$', line)
            if numbered_match:
                number = numbered_match.group(1)
                heading_text = numbered_match.group(2).strip()
                level = len(number.split('.'))
                
                section = {
                    'level': level,
                    'number': number,
                    'heading': heading_text,
                    'content': [],
                    'subsections': []
                }
                
                # Manage hierarchy
                while section_stack and section_stack[-1]['level'] >= level:
                    section_stack.pop()
                
                if section_stack:
                    section_stack[-1]['subsections'].append(section)
                else:
                    structure['sections'].append(section)
                
                section_stack.append(section)
                current_section = section
                continue
            
            # Detect ALL CAPS headings (usually level 1)
            if line.isupper() and len(line.split()) <= 10 and len(line) > 3:
                section = {
                    'level': 1,
                    'heading': line,
                    'content': [],
                    'subsections': []
                }
                structure['sections'].append(section)
                section_stack = [section]
                current_section = section
                continue
            
            # Add content to current section
            if current_section:
                current_section['content'].append(line)
            elif not structure['title']:
                # First non-empty line might be title
                if len(line) < 100 and not line.endswith('.'):
                    structure['title'] = line
                else:
                    structure['sections'].append({
                        'level': 0,
                        'heading': 'Introduction',
                        'content': [line],
                        'subsections': []
                    })
        
        return structure
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown or structured text"""
        tables = []
        
        # Markdown table pattern
        markdown_table_pattern = r'\|(.+)\|\n\|[-\s\|:]+\|\n((?:\|.+\|\n?)+)'
        markdown_matches = re.finditer(markdown_table_pattern, text, re.MULTILINE)
        
        for match in markdown_matches:
            header_row = match.group(1)
            data_rows = match.group(2)
            
            headers = [h.strip() for h in header_row.split('|') if h.strip()]
            rows = []
            
            for row_line in data_rows.strip().split('\n'):
                if '|' in row_line:
                    cells = [c.strip() for c in row_line.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        rows.append(cells)
            
            if headers and rows:
                tables.append({
                    'type': 'markdown_table',
                    'headers': headers,
                    'rows': rows,
                    'row_count': len(rows),
                    'column_count': len(headers)
                })
        
        # Detect table-like structures (aligned columns)
        lines = text.split('\n')
        table_candidates = []
        current_table = []
        
        for line in lines:
            # Check if line has multiple columns (spaces or tabs between words)
            if re.match(r'^[\w\s]+\s{2,}[\w\s]+', line):
                current_table.append(line)
            else:
                if len(current_table) >= 2:
                    # Try to parse as table
                    parsed_table = self._parse_text_table(current_table)
                    if parsed_table:
                        tables.append(parsed_table)
                current_table = []
        
        # Check last table candidate
        if len(current_table) >= 2:
            parsed_table = self._parse_text_table(current_table)
            if parsed_table:
                tables.append(parsed_table)
        
        return tables
    
    def _parse_text_table(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a text-based table structure"""
        if len(lines) < 2:
            return None
        
        # Try to detect columns by spacing
        rows = []
        for line in lines:
            # Split by multiple spaces
            cells = re.split(r'\s{2,}', line.strip())
            if len(cells) >= 2:
                rows.append(cells)
        
        if len(rows) >= 2:
            return {
                'type': 'text_table',
                'headers': rows[0] if len(rows[0]) == len(rows[1]) else None,
                'rows': rows[1:] if len(rows) > 1 else rows,
                'row_count': len(rows) - 1 if len(rows) > 1 else len(rows),
                'column_count': len(rows[0]) if rows else 0
            }
        
        return None
    
    def extract_figures(self, text: str, markdown_output: str = None) -> List[Dict[str, Any]]:
        """Extract figures and images from text"""
        figures = []
        
        # Markdown image references: ![alt](path) or ![](path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        image_matches = re.finditer(image_pattern, text)
        
        for match in image_matches:
            alt_text = match.group(1)
            image_path = match.group(2)
            figures.append({
                'type': 'image',
                'alt_text': alt_text,
                'path': image_path,
                'caption': alt_text if alt_text else None
            })
        
        # Detect figure references in text: "Figure 1", "Fig. 2", etc.
        figure_ref_pattern = r'(?:Figure|Fig\.?)\s+(\d+[a-z]?)(?:\s*[:\-]\s*(.+?))?(?=\n|$)'
        figure_refs = re.finditer(figure_ref_pattern, text, re.IGNORECASE)
        
        for match in figure_refs:
            figure_num = match.group(1)
            caption = match.group(2).strip() if match.group(2) else None
            figures.append({
                'type': 'figure_reference',
                'number': figure_num,
                'caption': caption
            })
        
        # Detect bounding box references from DeepSeek-OCR grounding output
        grounding_pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>\[([\d\.,\s]+)\]<\|/det\|>'
        grounding_matches = re.finditer(grounding_pattern, text)
        
        for match in grounding_matches:
            coords = match.group(1)
            try:
                coords_list = [float(x.strip()) for x in coords.split(',')]
                if len(coords_list) >= 4:
                    figures.append({
                        'type': 'detected_image',
                        'bounding_box': coords_list[:4],
                        'coordinates': coords_list
                    })
            except:
                pass
        
        return figures
    
    def create_structured_json(self, 
                             document_type: str,
                             hierarchical_structure: Dict,
                             tables: List[Dict],
                             figures: List[Dict],
                             full_text: str,
                             metadata: Dict) -> Dict[str, Any]:
        """Create comprehensive structured JSON for AI training"""
        
        # Calculate statistics
        total_sections = len(hierarchical_structure.get('sections', []))
        total_tables = len(tables)
        total_figures = len(figures)
        
        # Count words and paragraphs
        paragraphs = []
        for section in hierarchical_structure.get('sections', []):
            content = ' '.join(section.get('content', []))
            if content.strip():
                paragraphs.append(content)
        
        structured_json = {
            'document_metadata': {
                'document_type': document_type,
                'title': hierarchical_structure.get('title', 'Untitled Document'),
                'total_sections': total_sections,
                'total_tables': total_tables,
                'total_figures': total_figures,
                'total_paragraphs': len(paragraphs),
                'word_count': metadata.get('total_words', 0),
                'page_count': metadata.get('total_pages', 1),
                'processing_timestamp': metadata.get('processing_timestamp', ''),
                'ocr_engine': metadata.get('ocr_engine', 'unknown')
            },
            'hierarchical_structure': hierarchical_structure,
            'content_sections': [],
            'tables': tables,
            'figures': figures,
            'full_text': full_text,
            'training_data': {
                'sections': [],
                'paragraphs': paragraphs,
                'tables_data': [{'table_id': i+1, **table} for i, table in enumerate(tables)],
                'figures_data': [{'figure_id': i+1, **figure} for i, figure in enumerate(figures)]
            }
        }
        
        # Extract content sections with hierarchy
        def extract_section_content(section, parent_path=""):
            section_path = f"{parent_path}/{section['heading']}" if parent_path else section['heading']
            content = {
                'section_path': section_path,
                'level': section['level'],
                'heading': section['heading'],
                'content': '\n'.join(section.get('content', [])),
                'word_count': len(' '.join(section.get('content', [])).split()),
                'subsections': []
            }
            
            for subsection in section.get('subsections', []):
                content['subsections'].append(extract_section_content(subsection, section_path))
            
            return content
        
        for section in hierarchical_structure.get('sections', []):
            structured_json['content_sections'].append(extract_section_content(section))
        
        return structured_json

    def create_type_aware_json(self,
                               document_type: str,
                               hierarchical_structure: Dict,
                               tables: List[Dict],
                               figures: List[Dict],
                               full_text: str,
                               metadata: Dict) -> Dict[str, Any]:
        """
        Create a type-aware structured JSON that augments the generic schema
        with document-type-specific fields to improve AI training usability.
        """
        base = self.create_structured_json(
            document_type=document_type,
            hierarchical_structure=hierarchical_structure,
            tables=tables,
            figures=figures,
            full_text=full_text,
            metadata=metadata
        )

        text = full_text if full_text else ''
        type_specific: Dict[str, Any] = {}

        if document_type == 'invoice':
            type_specific = self._extract_invoice_fields(text, tables)
        elif document_type == 'academic_paper':
            type_specific = self._extract_academic_fields(text, hierarchical_structure)
        elif document_type == 'form':
            type_specific = self._extract_form_fields(text)
        elif document_type == 'report':
            type_specific = self._extract_report_fields(hierarchical_structure, tables)
        else:
            # Generic enhancement: summarize top-level sections
            type_specific = {
                'main_sections': [s.get('heading') for s in (hierarchical_structure.get('sections', []) or []) if s.get('heading')]
            }

        base['type_specific'] = type_specific
        return base

    def _extract_invoice_fields(self, text: str, tables: List[Dict]) -> Dict[str, Any]:
        """Heuristic extraction of common invoice fields and line items."""
        def find_first(pattern):
            m = pattern.search(text)
            return (m.group(2) if m and m.lastindex and m.lastindex >= 2 else None)

        currency = None
        mcur = self.invoice_patterns['currency'].search(text)
        if mcur:
            symbol = mcur.group(1)
            currency = {'$': 'USD', '€': 'EUR', '£': 'GBP'}.get(symbol, symbol)

        # Try to infer line items from the largest table
        line_items = []
        table_like = None
        if tables:
            table_like = max(tables, key=lambda t: t.get('row_count', 0))
            headers = [h.lower() for h in (table_like.get('headers') or [])]
            for row in table_like.get('rows', []):
                item = {}
                if headers and len(row) == len(headers):
                    for h, v in zip(headers, row):
                        item[h] = v
                else:
                    # Fallback mapping
                    if len(row) >= 1:
                        item['description'] = row[0]
                    if len(row) >= 2:
                        item['quantity_or_unit'] = row[1]
                    if len(row) >= 3:
                        item['price_or_amount'] = row[2]
                if item:
                    line_items.append(item)

        return {
            'invoice_number': find_first(self.invoice_patterns['invoice_no']),
            'invoice_date': find_first(self.invoice_patterns['date']),
            'subtotal': find_first(self.invoice_patterns['subtotal']),
            'tax': find_first(self.invoice_patterns['tax']),
            'total': find_first(self.invoice_patterns['total']),
            'currency': currency,
            'line_items': line_items
        }

    def _extract_academic_fields(self, text: str, hierarchical_structure: Dict) -> Dict[str, Any]:
        """Heuristic extraction for academic papers (title/authors/abstract/references)."""
        title = hierarchical_structure.get('title') or None
        if not title:
            mtitle = self.academic_patterns['title'].search(text)
            if mtitle:
                title = mtitle.group(1).strip()

        mauth = self.academic_patterns['authors'].search(text)
        authors = []
        if mauth:
            candidates = [a.strip() for a in re.split(r',| and ', mauth.group(1)) if a.strip()]
            authors = candidates[:12]

        mabst = self.academic_patterns['abstract'].search(text)
        abstract = mabst.group(1).strip() if mabst else None

        mrefs = self.academic_patterns['references'].search(text)
        references = []
        if mrefs:
            refs_text = mrefs.group(1).strip()
            references = [r.strip() for r in re.split(r'\n\d+\.|\n\[\d+\]', '\n' + refs_text) if r.strip()]

        return {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'references': references[:100]
        }

    def _extract_form_fields(self, text: str) -> Dict[str, Any]:
        """Parse key: value pairs common in forms; returns list of detected fields."""
        fields = []
        for match in self.form_patterns['field_line'].finditer(text):
            key = match.group(1).strip().lower().replace(' ', '_')
            val = match.group(2).strip()
            if key and val:
                fields.append({'field': key, 'value': val})
        return {
            'fields': fields,
            'field_count': len(fields)
        }

    def _extract_report_fields(self, hierarchical_structure: Dict, tables: List[Dict]) -> Dict[str, Any]:
        """Basic signals for reports: executive summary presence, toc, appendix, table stats."""
        headings = [s.get('heading', '').lower() for s in (hierarchical_structure.get('sections', []) or [])]
        has_exec_summary = any('executive summary' in h for h in headings)
        has_appendix = any('appendix' in h for h in headings)
        return {
            'has_executive_summary': has_exec_summary,
            'has_appendix': has_appendix,
            'table_count': sum(1 for _ in tables)
        }

