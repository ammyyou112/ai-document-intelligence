"""Test script to check OCR processing and see errors"""
import os
import sys
from deepseek_ocr_wrapper import DeepSeekOCR

# Test with an uploaded image
test_image = 'uploads/e8105941-3b57-4fb3-855f-47d581d5fb2f_19840015553_page_1.png'

if not os.path.exists(test_image):
    print(f"Test image not found: {test_image}")
    sys.exit(1)

print("=" * 60)
print("Testing DeepSeek OCR Processing")
print("=" * 60)
print(f"Test image: {test_image}")
print(f"Image exists: {os.path.exists(test_image)}")
print()

try:
    print("Initializing model...")
    model = DeepSeekOCR()
    print(f"Model initialized on device: {model.device}")
    print()
    
    print("Processing image...")
    result = model.process(test_image, base_size=640, image_size=640, crop_mode=False)
    
    print()
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Text extracted: {len(result.get('full_text', ''))} characters")
    print(f"Lines: {result.get('metadata', {}).get('total_lines', 0)}")
    print(f"Words: {result.get('metadata', {}).get('total_words', 0)}")
    if result.get('full_text'):
        print(f"\nFirst 200 characters:")
        print(result.get('full_text', '')[:200])
    
except Exception as e:
    print()
    print("=" * 60)
    print("ERROR OCCURRED")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)

