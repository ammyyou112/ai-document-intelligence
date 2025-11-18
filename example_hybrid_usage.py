"""
Example usage of the Hybrid OCR Router

This script demonstrates how to use the hybrid routing system
to process documents with automatic engine selection.
"""
from PIL import Image
from app.processors.hybrid_ocr_router import HybridOCRRouter
from deepseek_ocr_wrapper import DeepSeekOCR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Example usage"""
    
    # Initialize router
    router = HybridOCRRouter(complexity_threshold=0.7)
    
    # Initialize DeepSeek OCR (for complex documents)
    try:
        deepseek_ocr = DeepSeekOCR()
        print("DeepSeek OCR initialized")
    except Exception as e:
        print(f"Warning: Could not initialize DeepSeek OCR: {e}")
        deepseek_ocr = None
    
    # Example: Process an image
    image_path = "path/to/your/image.png"  # Replace with actual path
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"Loaded image: {image.size}")
        
        # Process with hybrid router
        result = router.process(
            image=image,
            deepseek_processor=deepseek_ocr.process if deepseek_ocr else None
        )
        
        # Print results
        print(f"\n{'='*60}")
        print("Processing Results")
        print(f"{'='*60}")
        print(f"Engine used: {result['engine_used']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Number of blocks: {len(result['blocks'])}")
        
        if result['complexity_analysis']:
            analysis = result['complexity_analysis']
            print(f"\nComplexity Analysis:")
            print(f"  Complexity: {analysis.complexity}")
            print(f"  Confidence: {analysis.confidence:.2f}")
            print(f"  Reasons: {', '.join(analysis.reasons)}")
        
        # Print first few blocks
        print(f"\nFirst 5 text blocks:")
        for i, block in enumerate(result['blocks'][:5]):
            print(f"  Block {i+1}: {block['text'][:50]}...")
            print(f"    BBox: {block['bbox']}, Confidence: {block['confidence']:.2f}")
        
        # Print statistics
        stats = router.get_stats()
        print(f"\n{'='*60}")
        print("Statistics")
        print(f"{'='*60}")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Simple OCR calls: {stats['simple_calls']} ({stats['simple_percentage']:.1f}%)")
        print(f"DeepSeek OCR calls: {stats['deepseek_calls']} ({stats['deepseek_percentage']:.1f}%)")
        print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
        
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print("Please update the image_path variable with a valid image path")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

