"""
Example usage of Document Intelligence features

Demonstrates:
1. Document classification
2. Metadata extraction
3. Research paper structuring
"""
import sys
import os

# Add parent directory to path (so we can import from root)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from PIL import Image
from app.processors.document_classifier import DocumentClassifier
from app.processors.metadata_extractor import MetadataExtractor
from app.processors.research_paper_structurer import ResearchPaperStructurer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Example usage"""
    
    # Sample text (in real usage, this would come from OCR)
    sample_text = """
    Deep Learning for Document Understanding
    
    John Smith, Jane Doe
    Stanford University
    
    Abstract
    This paper presents a novel approach to document understanding using deep learning.
    We propose a hybrid architecture that combines convolutional and recurrent networks.
    
    Introduction
    Document understanding is a critical task in information extraction. Traditional
    methods have limitations in handling complex layouts and diverse content types.
    
    Methodology
    Our approach uses a two-stage pipeline: first, we extract text regions using
    object detection, then we apply sequence-to-sequence models for text recognition.
    
    Results
    We evaluated our method on three benchmark datasets. Our approach achieves
    95.2% accuracy, outperforming baseline methods by 8.5%.
    
    Discussion
    The results demonstrate the effectiveness of our hybrid approach. The combination
    of spatial and sequential modeling provides robust document understanding.
    
    Conclusion
    We have presented a novel deep learning approach for document understanding.
    Future work will explore multi-modal fusion and transfer learning.
    
    References
    1. Smith, J. (2023). Document Processing. Journal of AI, 45(2), 123-145.
    2. Doe, J. (2023). Deep Learning Applications. ACM Transactions, 12(3), 67-89.
    """
    
    print("=" * 60)
    print("Document Intelligence Example")
    print("=" * 60)
    
    # 1. Document Classification
    print("\n1. Document Classification")
    print("-" * 60)
    classifier = DocumentClassifier()
    classification = classifier.classify(sample_text)
    
    print(f"Document Type: {classification.type}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Keywords Found: {', '.join(classification.keywords_found[:5])}")
    print(f"Patterns Matched: {len(classification.patterns_matched)}")
    
    # 2. Metadata Extraction
    print("\n2. Metadata Extraction")
    print("-" * 60)
    extractor = MetadataExtractor()
    metadata = extractor.extract(sample_text)
    
    print(f"Title: {metadata.title}")
    print(f"Authors: {', '.join(metadata.authors) if metadata.authors else 'None'}")
    print(f"Date: {metadata.date}")
    print(f"Organization: {metadata.organization}")
    print(f"Document Number: {metadata.document_number}")
    
    # 3. Research Paper Structuring
    print("\n3. Research Paper Structuring")
    print("-" * 60)
    structurer = ResearchPaperStructurer()
    structure = structurer.structure(sample_text)
    
    print(f"Total Sections: {structure.total_sections}")
    print(f"Total Words: {structure.total_words}")
    print(f"\nSections Found:")
    for section_name in structure.section_order:
        section = structure.sections[section_name]
        print(f"  - {section_name}: {section.word_count} words")
        print(f"    Preview: {section.text[:80]}...")
    
    # Get specific section
    print("\n4. Accessing Specific Sections")
    print("-" * 60)
    abstract = structurer.get_section_text(structure, 'abstract')
    if abstract:
        print(f"Abstract ({len(abstract.split())} words):")
        print(f"  {abstract[:200]}...")
    
    # Convert to dictionary
    print("\n5. Dictionary Format")
    print("-" * 60)
    structure_dict = structurer.to_dict(structure)
    for section_name, section_data in list(structure_dict.items())[:3]:
        print(f"{section_name}:")
        print(f"  Words: {section_data['word_count']}")
        print(f"  Text preview: {section_data['text'][:60]}...")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

