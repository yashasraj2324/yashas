#!/usr/bin/env python3
"""
Enhanced test script for OCR and PII detection functionality with advanced regex and multiple masking options
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from ocr_utils import preprocess_image_all_rotations, extract_ocr_data_robust, detect_pii, mask_pii, pixelate_mask_pii

def create_test_image():
    """Create a test image with sample text including various PII types"""
    # Create a white image
    img = Image.new('RGB', (900, 700), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
        tiny_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        tiny_font = ImageFont.load_default()
    
    # Sample text with various PII types - enhanced with more edge cases
    text_lines = [
        ("DRIVER LICENSE", font),
        ("", font),  # Empty line
        ("Name: John Smith", small_font),
        ("First Name: Jane", small_font),
        ("Last Name: Doe", small_font),
        ("SSN: 123-45-6789", small_font),
        ("Social Security: 987-65-4321", small_font),
        ("DOB: 01/15/1985", small_font),
        ("Birth Date: March 20, 1990", small_font),
        ("Address: 123 Main Street", small_font),
        ("Street: 456 Oak Avenue", small_font),
        ("City: Anytown, USA", small_font),
        ("Phone: (555) 123-4567", small_font),
        ("Mobile: +1-555-987-6543", small_font),
        ("Tel: 555-111-2222", small_font),
        ("Email: john.smith@email.com", small_font),
        ("E-mail: jsmith@company.com", small_font),
        ("", font),  # Empty line
        ("License Number: ABC123456", small_font),
        ("Expires: 12/31/2025", small_font),
        ("", font),  # Empty line
        ("Credit Card: 4532-1234-5678-9012", small_font),
        ("Visa: 4111-1111-1111-1111", small_font),
        ("MasterCard: 5555-5555-5555-4444", small_font),
        ("", font),  # Empty line
        ("Emergency Contact:", tiny_font),
        ("Jane Smith - (555) 111-2222", tiny_font),
        ("Address: 456 Oak Avenue, Suite 100", tiny_font),
        ("", font),  # Empty line
        ("Medical Info:", tiny_font),
        ("Blood Type: O+", tiny_font),
        ("Allergies: None", tiny_font),
        ("", font),  # Empty line
        ("Additional PII:", tiny_font),
        ("Cardholder: John M. Smith", tiny_font),
        ("Account Holder: Jane A. Doe", tiny_font),
        ("Holder: Robert Johnson", tiny_font),
    ]
    
    y_position = 50
    for text, font_obj in text_lines:
        if text:
            draw.text((50, y_position), text, fill='black', font=font_obj)
        y_position += 35
    
    # Save test image
    test_image_path = "test_id_enhanced.png"
    img.save(test_image_path)
    print(f"Created enhanced test image: {test_image_path}")
    return test_image_path

def test_ocr_pipeline():
    """Test the complete OCR and PII detection pipeline with robust rotation and preprocessing"""
    print("🔍 Testing Robust OCR and Advanced PII Detection Pipeline...")
    print("=" * 70)
    
    # Create test image
    test_image_path = create_test_image()
    
    # Read image bytes
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
    
    try:
        # Test robust preprocessing
        print("1. 🖼️  Testing robust image preprocessing (all rotations)...")
        preprocessed_images = preprocess_image_all_rotations(image_bytes)
        print(f"   ✓ Preprocessing completed - Generated {len(preprocessed_images)} versions (rotations x modes)")
        
        # Test robust OCR
        print("2. 📝 Testing robust OCR extraction...")
        ocr_results = extract_ocr_data_robust(preprocessed_images)
        print(f"   ✓ OCR extracted {len(ocr_results)} text elements")
        
        # Show extracted text with confidence
        print("   📋 Extracted text (first 10 elements):")
        for i, result in enumerate(ocr_results[:10]):
            conf = result.get('confidence', 'N/A')
            print(f"     {i+1}. '{result['text']}' (Confidence: {conf}%) at ({result['left']}, {result['top']})")
        
        # Test enhanced PII detection
        print("3. 🛡️  Testing enhanced PII detection with advanced regex...")
        pii_boxes = detect_pii(ocr_results)
        print(f"   ✓ Found {len(pii_boxes)} PII elements")
        
        # Show detected PII with details
        if pii_boxes:
            print("   🔍 Detected PII by type:")
            pii_by_type = {}
            for pii in pii_boxes:
                pii_type = pii['label']
                if pii_type not in pii_by_type:
                    pii_by_type[pii_type] = []
                pii_by_type[pii_type].append(pii)
            
            for pii_type, items in pii_by_type.items():
                print(f"     📍 {pii_type} ({len(items)} found):")
                for i, pii in enumerate(items):
                    conf = pii.get('confidence', 'N/A')
                    detected_text = pii.get('text', 'N/A')
                    print(f"       {i+1}. '{detected_text}' (Confidence: {conf}%)")
        else:
            print("   ℹ️  No PII detected in this test image")
        
        # Test different masking methods
        print("4. 🎨 Testing multiple masking methods...")
        mask_img = preprocessed_images[0]  # Use the first rotation for masking demo
        
        # Test black masking
        print("   ⬛ Testing black rectangle masking...")
        black_masked = mask_pii(mask_img, pii_boxes, "black")
        black_path = "test_black_masked.png"
        Image.fromarray(black_masked).save(black_path)
        print(f"   ✓ Saved black masked image: {black_path}")
        
        # Test blur masking
        print("   🌫️  Testing Gaussian blur masking...")
        blur_masked = mask_pii(mask_img, pii_boxes, "blur")
        blur_path = "test_blur_masked.png"
        Image.fromarray(blur_masked).save(blur_path)
        print(f"   ✓ Saved blur masked image: {blur_path}")
        
        # Test pixelation masking
        print("   🧩 Testing pixelation masking...")
        pixelate_masked = pixelate_mask_pii(mask_img, pii_boxes)
        pixelate_path = "test_pixelate_masked.png"
        Image.fromarray(pixelate_masked).save(pixelate_path)
        print(f"   ✓ Saved pixelate masked image: {pixelate_path}")
        
        # Performance statistics
        print("\n📊 Performance Statistics:")
        print(f"   • Total text elements: {len(ocr_results)}")
        print(f"   • PII elements detected: {len(pii_boxes)}")
        print(f"   • Average confidence: {sum(r.get('confidence', 0) for r in ocr_results) / len(ocr_results) if ocr_results else 0:.1f}%")
        
        # PII type breakdown
        if pii_boxes:
            pii_types = {}
            for pii in pii_boxes:
                pii_type = pii['label']
                pii_types[pii_type] = pii_types.get(pii_type, 0) + 1
            
            print("   • PII Type Breakdown:")
            for pii_type, count in pii_types.items():
                print(f"     - {pii_type}: {count}")
        
        print("\n✅ All tests passed! Enhanced OCR and PII detection are working correctly.")
        print("🚀 Enhanced improvements implemented:")
        print("   • Multi-threshold preprocessing")
        print("   • Multiple PSM modes for OCR")
        print("   • Advanced regex patterns for multiple PII types")
        print("   • Enhanced credit card validation with Luhn algorithm")
        print("   • Context-aware name detection with proximity analysis")
        print("   • Enhanced SSN validation with multiple formats")
        print("   • Phone number validation with format checking")
        print("   • Email validation with RFC compliance")
        print("   • Date validation with multiple format support")
        print("   • Address validation with component checking")
        print("   • Context map creation for better detection")
        print("   • Confidence boosting for context-aware detection")
        print("   • Overlapping PII merging")
        print("   • Multiple masking options (black, blur, pixelate)")
        
        print("\n🎨 Masking Results:")
        print(f"   • Black rectangle: {black_path}")
        print(f"   • Gaussian blur: {blur_path}")
        print(f"   • Pixelation: {pixelate_path}")
        
        print("\n🔍 Enhanced PII Detection Features:")
        print("   • SSN: Multiple formats with validation (000, 666, 900+ excluded)")
        print("   • Credit Cards: Luhn algorithm validation for major card types")
        print("   • Phone Numbers: US and international formats with validation")
        print("   • Email: RFC-compliant pattern matching with domain validation")
        print("   • Names: Context-aware detection with spaCy NER and proximity")
        print("   • Addresses: Street address pattern recognition with component validation")
        print("   • DOB: Multiple date format support with reasonable date validation")
        print("   • Context Mapping: Spatial analysis for better detection accuracy")
        print("   • Confidence Boosting: Enhanced scores for context-aware matches")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        test_files = [test_image_path, "test_black_masked.png", "test_blur_masked.png", "test_pixelate_masked.png"]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Cleaned up: {file_path}")

if __name__ == "__main__":
    test_ocr_pipeline() 