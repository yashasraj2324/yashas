import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import spacy
import re
from typing import List, Tuple, Dict

# Configure pytesseract for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load spaCy English model for NER (download with: python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')

# Enhanced Tesseract configuration for better accuracy
TESSERACT_CONFIG = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.,:()/ '

# --- 1. Robust Image Preprocessing and Auto-Rotation ---
def preprocess_image_all_rotations(image_bytes: bytes) -> List[np.ndarray]:
    """
    Returns a list of preprocessed images for all 4 rotations (0, 90, 180, 270 degrees),
    each in both grayscale and binarized (Otsu) versions.
    """
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Invalid image data')
    
    # Resize if image is too small (improves OCR accuracy)
    height, width = img.shape[:2]
    if width < 800:
        scale_factor = 800 / width
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Generate all 4 rotations
    rotations = [img]
    for i in range(1, 4):
        rotations.append(cv2.rotate(rotations[-1], cv2.ROTATE_90_CLOCKWISE))
    
    preprocessed = []
    for rot_img in rotations:
        # Grayscale
        gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        # Binarized (Otsu)
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(gray)
        preprocessed.append(binarized)
    return preprocessed

# --- 2. Robust OCR Pipeline ---
def extract_ocr_data_robust(image_list: List[np.ndarray]) -> List[Dict]:
    """
    Run OCR on all preprocessed images, merge results, and deduplicate.
    """
    all_results = []
    psm_modes = [6, 8]
    for img in image_list:
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm}'
            try:
                data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
                n_boxes = len(data['level'])
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    conf = data['conf'][i]
                    if text and conf > 20 and len(text) > 1:
                        result = {
                            'text': text,
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'confidence': conf,
                            'psm': psm
                        }
                        all_results.append(result)
            except Exception:
                continue
    # Deduplicate by text and position
    unique_results = []
    seen = set()
    for r in all_results:
        key = (r['text'], r['left'], r['top'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    return unique_results

def merge_overlapping_text(results: List[Dict]) -> List[Dict]:
    """Merge overlapping or adjacent text boxes"""
    if not results:
        return results
    
    # Sort by top position, then left position
    sorted_results = sorted(results, key=lambda x: (x['top'], x['left']))
    
    merged = []
    current_group = [sorted_results[0]]
    
    for result in sorted_results[1:]:
        # Check if this result overlaps or is adjacent to the current group
        should_merge = False
        for group_result in current_group:
            # Check horizontal overlap
            h_overlap = (result['left'] < group_result['left'] + group_result['width'] and 
                        result['left'] + result['width'] > group_result['left'])
            
            # Check vertical proximity (within 20 pixels)
            v_proximity = abs(result['top'] - group_result['top']) < 20
            
            if h_overlap and v_proximity:
                should_merge = True
                break
        
        if should_merge:
            current_group.append(result)
        else:
            # Merge current group and start new one
            if current_group:
                merged.append(merge_text_group(current_group))
            current_group = [result]
    
    # Don't forget the last group
    if current_group:
        merged.append(merge_text_group(current_group))
    
    return merged

def merge_text_group(group: List[Dict]) -> Dict:
    """Merge a group of text results into one"""
    if len(group) == 1:
        return group[0]
    
    # Sort by left position
    sorted_group = sorted(group, key=lambda x: x['left'])
    
    # Combine text
    combined_text = ' '.join([item['text'] for item in sorted_group])
    
    # Calculate bounding box
    left = min(item['left'] for item in group)
    top = min(item['top'] for item in group)
    right = max(item['left'] + item['width'] for item in group)
    bottom = max(item['top'] + item['height'] for item in group)
    
    # Use highest confidence
    max_conf = max(item['confidence'] for item in group)
    
    return {
        'text': combined_text,
        'left': left,
        'top': top,
        'width': right - left,
        'height': bottom - top,
        'confidence': max_conf,
        'psm': group[0]['psm']
    }

# --- 3. Enhanced PII Detection with Advanced Regex ---
def detect_pii(ocr_results: List[Dict]) -> List[Dict]:
    pii_boxes = []
    
    # Advanced SSN regex patterns with validation
    ssn_patterns = [
        # Standard formats with validation
        (r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b', 'SSN'),  # Standard format with validation
        (r'\b(?!000|666|9\d{2})\d{3}\s(?!00)\d{2}\s(?!0000)\d{4}\b', 'SSN'),  # Space separated
        (r'\b(?!000|666|9\d{2})\d{3}\.(?!00)\d{2}\.(?!0000)\d{4}\b', 'SSN'),  # Dot separated
        (r'\b(?!000|666|9\d{2})\d{9}\b', 'SSN'),  # No separators (with validation)
        # Additional SSN patterns for edge cases
        (r'\bSSN[:\s]*(\d{3}[-.\s]?\d{2}[-.\s]?\d{4})\b', 'SSN'),  # SSN: prefix
        (r'\bSocial\s+Security[:\s]*(\d{3}[-.\s]?\d{2}[-.\s]?\d{4})\b', 'SSN'),  # Social Security prefix
    ]
    
    # Enhanced credit card patterns
    credit_card_patterns = [
        (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b', 'CreditCard'),
        (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 'CreditCard'),  # Generic 16-digit format
        (r'\b(?:Visa|MasterCard|American Express|Discover|Amex)[:\s]*(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b', 'CreditCard'),  # Card type prefix
        (r'\bCard[:\s]*(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b', 'CreditCard'),  # Card prefix
    ]
    
    # Enhanced phone number patterns
    phone_patterns = [
        (r'\b\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b', 'Phone'),
        (r'\b\+?1[-. ]?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b', 'Phone'),  # US format with country code
        (r'\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b', 'Phone'),  # Simple format
        (r'\b(?:Phone|Tel|Mobile|Cell|Contact)[:\s]*(\+?1?[-. ]?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4})\b', 'Phone'),  # Phone prefix
        (r'\b(?:Phone|Tel|Mobile|Cell|Contact)[:\s]*(\d{3}[-. ]?\d{3}[-. ]?\d{4})\b', 'Phone'),  # Simple phone prefix
    ]
    
    # Enhanced email patterns
    email_patterns = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
        (r'\b(?:Email|E-mail|Mail|Contact)[:\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', 'Email'),  # Email prefix
    ]
    
    # Enhanced date of birth patterns
    dob_patterns = [
        (r'\b(?:DOB|Date of Birth|Birth Date|Born|Birthday)[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', 'DOB'),
        (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', 'DOB'),  # Generic date format
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', 'DOB'),
        (r'\b(?:DOB|Date of Birth|Birth Date|Born|Birthday)[:.]?\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b', 'DOB'),
    ]
    
    # Enhanced address patterns
    address_patterns = [
        (r'\b(?:Address|Addr|Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)[:.]?\s*([A-Za-z0-9\s,.-]+)\b', 'Address'),
        (r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b', 'Address'),
        (r'\b(?:Address|Addr|Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)[:.]?\s*(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct))\b', 'Address'),
    ]
    
    # Enhanced name patterns with context
    name_keywords = [
        'name:', 'first:', 'last:', 'given:', 'surname:', 'full name:', 'legal name:',
        'driver name:', 'license name:', 'cardholder:', 'account holder:', 'holder:',
        'first name:', 'last name:', 'middle name:', 'maiden name:', 'preferred name:'
    ]
    
    # Collect all text for NER and context analysis
    full_text = ' '.join([r['text'] for r in ocr_results])
    doc = nlp(full_text)
    
    # Get names from NER with improved filtering
    ner_names = set([ent.text.strip() for ent in doc.ents if ent.label_ == 'PERSON'])
    
    # Enhanced name detection patterns
    name_patterns = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
        r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # Last, First
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
        r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # First M. Last
    ]
    
    # Context analysis for better PII detection
    context_keywords = {
        'SSN': ['ssn', 'social security', 'social security number', 'ss#', 'ssn#'],
        'CreditCard': ['card', 'credit', 'visa', 'mastercard', 'amex', 'discover', 'card number', 'cc#'],
        'Phone': ['phone', 'telephone', 'mobile', 'cell', 'contact', 'tel', 'call'],
        'Email': ['email', 'e-mail', 'mail', 'contact', 'e-mail address'],
        'DOB': ['dob', 'birth', 'born', 'date of birth', 'birthday', 'birth date'],
        'Address': ['address', 'street', 'city', 'state', 'zip', 'postal', 'location', 'residence'],
    }
    
    # Create a context map for better detection
    context_map = create_context_map(ocr_results, context_keywords)
    
    for r in ocr_results:
        text = r['text'].strip()
        original_text = text
        
        # Check for SSN patterns
        for pattern, label in ssn_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the matched SSN
                ssn_text = match.group(1) if match.groups() else match.group(0)
                if label == 'SSN' and not looks_like_random_number(ssn_text):
                    pii_boxes.append({**r, 'label': label, 'text': ssn_text, 'confidence': r.get('confidence', 0)})
                    break
        
        # Check for credit card patterns
        for pattern, label in credit_card_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                card_text = match.group(1) if match.groups() else match.group(0)
                if is_valid_credit_card(card_text):
                    pii_boxes.append({**r, 'label': label, 'text': card_text, 'confidence': r.get('confidence', 0)})
                    break
        
        # Check for phone number patterns
        for pattern, label in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phone_text = match.group(1) if match.groups() else match.group(0)
                if is_valid_phone_number(phone_text):
                    pii_boxes.append({**r, 'label': label, 'text': phone_text, 'confidence': r.get('confidence', 0)})
                    break
        
        # Check for email patterns
        for pattern, label in email_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                email_text = match.group(1) if match.groups() else match.group(0)
                if is_valid_email(email_text):
                    pii_boxes.append({**r, 'label': label, 'text': email_text, 'confidence': r.get('confidence', 0)})
                    break
        
        # Check for DOB patterns
        for pattern, label in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dob_text = match.group(1) if match.groups() else match.group(0)
                if is_valid_date(dob_text):
                    pii_boxes.append({**r, 'label': label, 'text': dob_text, 'confidence': r.get('confidence', 0)})
                    break
        
        # Check for address patterns
        for pattern, label in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                addr_text = match.group(1) if match.groups() else match.group(0)
                if is_valid_address(addr_text):
                    pii_boxes.append({**r, 'label': label, 'text': addr_text, 'confidence': r.get('confidence', 0)})
                    break
        
        # Enhanced name detection with context
        name_detected = False
        
        # Check if text contains name keywords
        for keyword in name_keywords:
            if keyword.lower() in text.lower():
                # Extract the name part after the keyword
                parts = text.split(keyword, 1)
                if len(parts) > 1:
                    name_part = parts[1].strip()
                    if name_part and len(name_part) > 2 and looks_like_name(name_part):
                        pii_boxes.append({**r, 'label': 'Name', 'text': name_part, 'confidence': r.get('confidence', 0)})
                        name_detected = True
                        break
        
        # Check NER results with context validation
        if not name_detected and text in ner_names and len(text) > 2:
            if looks_like_name(text) and has_name_context(ocr_results, r, context_map):
                pii_boxes.append({**r, 'label': 'Name', 'confidence': r.get('confidence', 0)})
                name_detected = True
        
        # Check name patterns with context
        if not name_detected:
            for pattern in name_patterns:
                if re.match(pattern, text):
                    if has_name_context(ocr_results, r, context_map):
                        pii_boxes.append({**r, 'label': 'Name', 'confidence': r.get('confidence', 0)})
                        break
    
    # Remove duplicates and merge overlapping PII
    pii_boxes = merge_overlapping_pii(pii_boxes)
    
    # Apply confidence boosting for context-aware detection
    pii_boxes = boost_confidence_with_context(pii_boxes, context_map)
    
    return pii_boxes

def create_context_map(ocr_results: List[Dict], context_keywords: Dict) -> Dict:
    """Create a context map for better PII detection"""
    context_map = {}
    
    for result in ocr_results:
        text = result['text'].lower()
        position = (result['top'], result['left'])
        
        for pii_type, keywords in context_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if pii_type not in context_map:
                        context_map[pii_type] = []
                    context_map[pii_type].append({
                        'position': position,
                        'text': text,
                        'confidence': result.get('confidence', 0)
                    })
    
    return context_map

def boost_confidence_with_context(pii_boxes: List[Dict], context_map: Dict) -> List[Dict]:
    """Boost confidence scores for PII detected near relevant context"""
    for pii in pii_boxes:
        pii_type = pii['label']
        pii_position = (pii['top'], pii['left'])
        
        if pii_type in context_map:
            # Check if PII is near relevant context
            for context in context_map[pii_type]:
                context_position = context['position']
                
                # Calculate distance
                distance = ((pii_position[0] - context_position[0]) ** 2 + 
                           (pii_position[1] - context_position[1]) ** 2) ** 0.5
                
                # If within 200 pixels, boost confidence
                if distance < 200:
                    pii['confidence'] = min(100, pii.get('confidence', 0) + 10)
                    break
    
    return pii_boxes

def is_valid_credit_card(text: str) -> bool:
    """Enhanced credit card validation using Luhn algorithm"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', text)
    
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    # Check for obvious invalid patterns
    if len(set(digits)) == 1:  # All same digits
        return False
    
    if digits in ['1234567890123456', '0000000000000000']:  # Test patterns
        return False
    
    # Luhn algorithm
    total = 0
    is_even = False
    
    for digit in reversed(digits):
        d = int(digit)
        if is_even:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        is_even = not is_even
    
    return total % 10 == 0

def is_valid_phone_number(text: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', text)
    
    # Check for valid US phone number length
    if len(digits) == 10 or (len(digits) == 11 and digits.startswith('1')):
        # Check for obvious invalid patterns
        if len(set(digits)) == 1:  # All same digits
            return False
        
        if digits.startswith('000') or digits.startswith('111'):
            return False
        
        return True
    
    return False

def is_valid_email(text: str) -> bool:
    """Enhanced email validation"""
    # Basic email pattern
    email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
    
    if not re.match(email_pattern, text):
        return False
    
    # Check for common invalid patterns
    if text.startswith('.') or text.endswith('.'):
        return False
    
    if '..' in text or '--' in text:
        return False
    
    # Check for reasonable domain length
    parts = text.split('@')
    if len(parts) != 2:
        return False
    
    local_part, domain = parts
    
    if len(local_part) > 64 or len(domain) > 253:
        return False
    
    return True

def is_valid_date(text: str) -> bool:
    """Validate date format"""
    # Try to parse the date
    import datetime
    
    # Common date formats
    date_formats = [
        '%m/%d/%Y', '%m/%d/%y', '%d/%m/%Y', '%d/%m/%y',
        '%m-%d-%Y', '%m-%d-%y', '%d-%m-%Y', '%d-%m-%y',
        '%b %d, %Y', '%B %d, %Y', '%b %d %Y', '%B %d %Y'
    ]
    
    for fmt in date_formats:
        try:
            date_obj = datetime.datetime.strptime(text, fmt)
            # Check if date is reasonable (not too far in past or future)
            current_year = datetime.datetime.now().year
            if 1900 <= date_obj.year <= current_year + 10:
                return True
        except ValueError:
            continue
    
    return False

def is_valid_address(text: str) -> bool:
    """Validate address format"""
    # Check for minimum length
    if len(text) < 10:
        return False
    
    # Check for address components
    address_indicators = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd', 'drive', 'dr', 'lane', 'ln', 'court', 'ct']
    
    text_lower = text.lower()
    has_street_indicator = any(indicator in text_lower for indicator in address_indicators)
    
    # Check for numbers
    has_numbers = bool(re.search(r'\d', text))
    
    # Check for reasonable character distribution
    letter_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    
    return has_street_indicator and has_numbers and letter_ratio > 0.3

def has_name_context(ocr_results: List[Dict], current_result: Dict, context_map: Dict = None) -> bool:
    """Enhanced context validation for name detection"""
    # Look for nearby keywords that suggest this is a name field
    current_top = current_result['top']
    current_left = current_result['left']
    
    # Check within 150 pixels for name-related keywords
    for result in ocr_results:
        if abs(result['top'] - current_top) < 150 and abs(result['left'] - current_left) < 400:
            text = result['text'].lower()
            name_indicators = ['name', 'first', 'last', 'given', 'surname', 'driver', 'license', 'cardholder', 'holder']
            if any(indicator in text for indicator in name_indicators):
                return True
    
    # Check context map if available
    if context_map and 'Name' in context_map:
        for context in context_map['Name']:
            context_position = context['position']
            distance = ((current_top - context_position[0]) ** 2 + 
                       (current_left - context_position[1]) ** 2) ** 0.5
            if distance < 300:
                return True
    
    return False

def looks_like_name(text: str) -> bool:
    """Enhanced name validation with more sophisticated checks"""
    # Common non-name words to exclude
    non_names = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'within', 'without',
        'driver', 'license', 'state', 'country', 'city', 'address', 'phone',
        'email', 'date', 'birth', 'expires', 'issued', 'number', 'id', 'card',
        'social', 'security', 'ssn', 'dob', 'exp', 'valid', 'until', 'class',
        'restrictions', 'endorsements', 'sex', 'height', 'weight', 'hair',
        'eyes', 'organ', 'donor', 'signature', 'official', 'document'
    }
    
    text_lower = text.lower()
    if text_lower in non_names:
        return False
    
    # Check if it has proper name characteristics
    if len(text) < 2:
        return False
    
    # Should start with capital letter
    if not text[0].isupper():
        return False
    
    # Should contain mostly letters and spaces
    letter_ratio = sum(1 for c in text if c.isalpha() or c.isspace()) / len(text)
    if letter_ratio < 0.8:
        return False
    
    # Should not contain numbers (except for some edge cases)
    if any(c.isdigit() for c in text):
        return False
    
    # Should not be all caps (except for short names)
    if len(text) > 3 and text.isupper():
        return False
    
    # Should not contain special characters (except hyphens and apostrophes for names like Mary-Jane, O'Connor)
    if re.search(r'[^A-Za-z\s\'-]', text):
        return False
    
    return True

def looks_like_random_number(text: str) -> bool:
    """Enhanced validation for 9-digit numbers vs SSN"""
    if not text.isdigit() or len(text) != 9:
        return True
    
    # SSNs have specific patterns - check for common non-SSN patterns
    # SSNs don't start with 000, 666, or 900-999
    first_three = int(text[:3])
    if first_three == 0 or first_three == 666 or first_three >= 900:
        return True
    
    # SSNs don't have all same digits
    if len(set(text)) == 1:
        return True
    
    # SSNs don't have sequential patterns like 123456789
    if text in ['123456789', '987654321', '111111111', '000000000']:
        return True
    
    # Check for other obvious non-SSN patterns
    if text.startswith('999') or text.startswith('888'):
        return True
    
    return False

# --- 4. Enhanced Masking ---
def mask_pii(image: np.ndarray, pii_boxes: List[Dict], mask_type: str = "black") -> np.ndarray:
    """
    Mask PII with different masking techniques
    
    Args:
        image: Input image
        pii_boxes: List of PII bounding boxes
        mask_type: "black" for black rectangles, "blur" for Gaussian blur
    """
    # Convert to color if needed
    if len(image.shape) == 2:
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image.copy()
    
    if mask_type == "blur":
        return blur_mask_pii(img_color, pii_boxes)
    else:
        return black_mask_pii(img_color, pii_boxes)

def black_mask_pii(image: np.ndarray, pii_boxes: List[Dict]) -> np.ndarray:
    """Apply black rectangle masking"""
    img_masked = image.copy()
    
    for box in pii_boxes:
        x, y, w, h = box['left'], box['top'], box['width'], box['height']
        
        # Add padding around the text for better masking
        padding = 2
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_masked.shape[1], x + w + padding)
        y2 = min(img_masked.shape[0], y + h + padding)
        
        # Draw black rectangle
        cv2.rectangle(img_masked, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    return img_masked

def blur_mask_pii(image: np.ndarray, pii_boxes: List[Dict]) -> np.ndarray:
    """Apply Gaussian blur masking"""
    img_masked = image.copy()
    
    for box in pii_boxes:
        x, y, w, h = box['left'], box['top'], box['width'], box['height']
        
        # Add padding around the text for better masking
        padding = 4
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_masked.shape[1], x + w + padding)
        y2 = min(img_masked.shape[0], y + h + padding)
        
        # Extract the region to blur
        region = img_masked[y1:y2, x1:x2]
        
        if region.size > 0:  # Check if region is valid
            # Apply strong Gaussian blur to make text unreadable
            # Use odd kernel size and high sigma for strong blur
            kernel_size = min(31, max(3, min(region.shape[0], region.shape[1]) // 2 * 2 + 1))
            blurred_region = cv2.GaussianBlur(region, (kernel_size, kernel_size), 15)
            
            # Replace the original region with blurred version
            img_masked[y1:y2, x1:x2] = blurred_region
    
    return img_masked

def pixelate_mask_pii(image: np.ndarray, pii_boxes: List[Dict], pixel_size: int = 10) -> np.ndarray:
    """Apply pixelation masking"""
    img_masked = image.copy()
    
    for box in pii_boxes:
        x, y, w, h = box['left'], box['top'], box['width'], box['height']
        
        # Add padding around the text for better masking
        padding = 4
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_masked.shape[1], x + w + padding)
        y2 = min(img_masked.shape[0], y + h + padding)
        
        # Extract the region to pixelate
        region = img_masked[y1:y2, x1:x2]
        
        if region.size > 0:  # Check if region is valid
            # Resize down and up to create pixelation effect
            small = cv2.resize(region, (max(1, region.shape[1] // pixel_size), 
                                       max(1, region.shape[0] // pixel_size)), 
                              interpolation=cv2.INTER_LINEAR)
            pixelated_region = cv2.resize(small, (region.shape[1], region.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Replace the original region with pixelated version
            img_masked[y1:y2, x1:x2] = pixelated_region
    
    return img_masked

def merge_overlapping_pii(pii_boxes: List[Dict]) -> List[Dict]:
    """Merge overlapping PII boxes to avoid duplicate masking"""
    if not pii_boxes:
        return pii_boxes
    
    # Sort by top position, then left position
    sorted_pii = sorted(pii_boxes, key=lambda x: (x['top'], x['left']))
    
    merged = []
    current_group = [sorted_pii[0]]
    
    for pii in sorted_pii[1:]:
        # Check if this PII overlaps or is adjacent to the current group
        should_merge = False
        for group_pii in current_group:
            # Check horizontal overlap
            h_overlap = (pii['left'] < group_pii['left'] + group_pii['width'] and 
                        pii['left'] + pii['width'] > group_pii['left'])
            
            # Check vertical proximity (within 30 pixels)
            v_proximity = abs(pii['top'] - group_pii['top']) < 30
            
            # Check if same type of PII
            same_type = pii['label'] == group_pii['label']
            
            if h_overlap and v_proximity and same_type:
                should_merge = True
                break
        
        if should_merge:
            current_group.append(pii)
        else:
            # Merge current group and start new one
            if current_group:
                merged.append(merge_pii_group(current_group))
            current_group = [pii]
    
    # Don't forget the last group
    if current_group:
        merged.append(merge_pii_group(current_group))
    
    return merged

def merge_pii_group(group: List[Dict]) -> Dict:
    """Merge a group of PII results into one"""
    if len(group) == 1:
        return group[0]
    
    # Sort by left position
    sorted_group = sorted(group, key=lambda x: x['left'])
    
    # Combine text
    combined_text = ' '.join([item['text'] for item in sorted_group])
    
    # Calculate bounding box
    left = min(item['left'] for item in group)
    top = min(item['top'] for item in group)
    right = max(item['left'] + item['width'] for item in group)
    bottom = max(item['top'] + item['height'] for item in group)
    
    # Use highest confidence
    max_conf = max(item.get('confidence', 0) for item in group)
    
    return {
        'text': combined_text,
        'left': left,
        'top': top,
        'width': right - left,
        'height': bottom - top,
        'confidence': max_conf,
        'label': group[0]['label']
    } 