"""
Stage 1.75: OCR-based Text Extraction and HTML Mismatch Detection
Detects visible text in screenshot that doesn't exist in HTML (dynamic errors, overlays)
"""

import cv2
import numpy as np
import functools
import signal
from typing import Dict, Any, List, Tuple
from PIL import Image
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import re
import gc

# EasyOCR - The ONLY OCR engine (PaddleOCR removed due to network issues)
OCR_ENGINE = None
_ocr_reader = None
_ocr_reader_initialized = False
_ocr_import_error = None

try:
    import easyocr
    OCR_ENGINE = 'easyocr'
    print("  ℹ Using EasyOCR for text extraction")
except Exception as e:
    _ocr_import_error = str(e)
    print("⚠ Warning: EasyOCR not available")
    print(f"  Error: {e}")
    print("  Install with: pip install easyocr torch torchvision")

@functools.lru_cache(maxsize=1)
def get_ocr_reader_cached():
    """
    Get or initialize EasyOCR reader with LRU cache (singleton pattern)
    Models (~80MB) downloaded automatically on first run and cached locally
    """
    if not OCR_ENGINE or OCR_ENGINE != 'easyocr':
        return None
        
    try:
        print("  → Initializing EasyOCR (first run downloads models ~80MB)...")
        import easyocr
        
        # Initialize EasyOCR with English language support
        # gpu=False: Use CPU (works on all systems)
        # verbose=False: Suppress detailed logs
        # Models are cached in: C:\Users\<username>\.EasyOCR\
        reader = easyocr.Reader(
            ['en'], 
            gpu=False, 
            verbose=False,
            download_enabled=True  # Allow model downloads
        )
        print("  ✓ EasyOCR initialized and cached for subsequent calls")
        return reader
    except Exception as e:
        print(f"  ⚠ EasyOCR initialization failed: {e}")
        print("  → Install with: pip install easyocr torch torchvision")
        return None

def get_ocr_reader():
    """Wrapper to maintain backward compatibility"""
    return get_ocr_reader_cached()


def is_likely_ocr_garbage(text: str) -> bool:
    """Filter OCR noise and garbage text"""
    text = text.strip()
    
    # Too short
    if len(text) < 4:
        return True
    
    # Too many special characters (likely noise)
    special_count = sum(1 for c in text if not c.isalnum() and c not in ' -,.!?')
    if len(text) > 0 and special_count / len(text) > 0.4:
        return True
    
    # Random uppercase + digit sequences without context (like "S5000")
    if text.isupper() and any(char.isdigit() for char in text) and len(text) < 10:
        # Check if it's NOT a real error code pattern
        error_patterns = ['HTTP', 'ERROR', 'CODE', 'STATUS', '404', '500', '502', '503']
        if not any(pattern in text for pattern in error_patterns):
            return True
    
    # No vowels in text longer than 5 chars (likely garbage)
    if len(text) > 5:
        vowels = 'aeiouAEIOU'
        if not any(v in text for v in vowels):
            return True
    
    # All digits (unless it's an error code)
    if text.isdigit() and len(text) not in [3, 4]:  # Allow 404, 500, etc.
        return True
    
    return False


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Enhance image for better OCR accuracy
    EasyOCR works well with color images, but we enhance contrast
    """
    # EasyOCR can handle color images, so keep original if already color
    if len(img.shape) == 2:  # If grayscale, convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Increase contrast using CLAHE on each channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def extract_visible_text(screenshot_path: str) -> Dict[str, Any]:
    """
    Extract all visible text from screenshot using OCR (EasyOCR or PaddleOCR)
    
    Returns:
        Dictionary with extracted text and metadata
    """
    if OCR_ENGINE is None:
        return {
            'success': False,
            'error': 'No OCR engine available - install paddleocr or easyocr',
            'full_text': '',
            'lines': [],
            'text_length': 0,
            'line_count': 0
        }
    
    try:
        # Get OCR reader instance
        reader = get_ocr_reader()
        if reader is None:
            return {
                'success': False,
                'error': f'Failed to initialize {OCR_ENGINE} reader',
                'full_text': '',
                'lines': [],
                'text_length': 0,
                'line_count': 0
            }
        
        # Load image
        img = cv2.imread(screenshot_path)
        if img is None:
            return {
                'success': False,
                'error': 'Could not load screenshot',
                'full_text': '',
                'lines': [],
                'text_length': 0,
                'line_count': 0
            }
        
        # Extract text based on engine
        all_text_parts = []
        cleaned_lines = []
        
        if OCR_ENGINE == 'easyocr':
            # Preprocess for better OCR
            processed = preprocess_for_ocr(img)
            # EasyOCR returns list of (bbox, text, confidence)
            results = reader.readtext(processed, paragraph=False)
            
            for detection in results:
                bbox, text, confidence = detection
                text = str(text).strip()
                if text and float(confidence) > 0.3:
                    all_text_parts.append(text)
                    if len(text) >= 3 or any(kw in text.lower() for kw in ['404', '500', '502', '503']):
                        cleaned_lines.append(text)
        
        elif OCR_ENGINE == 'paddleocr':
            # PaddleOCR returns list of [bbox, (text, confidence)]
            # Use use_angle_cls parameter instead of cls (API changed)
            results = reader.ocr(screenshot_path, use_angle_cls=True)
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        text = str(text).strip()
                        
                        # Filter low confidence and garbage text
                        if text and float(confidence) > 0.4 and not is_likely_ocr_garbage(text):
                            all_text_parts.append(text)
                            if len(text) >= 3 or any(kw in text.lower() for kw in ['404', '500', '502', '503']):
                                cleaned_lines.append(text)
        
        # Combine all text
        full_text = ' '.join(all_text_parts)
        
        return {
            'success': True,
            'full_text': full_text,
            'lines': cleaned_lines,
            'text_length': len(full_text),
            'line_count': len(cleaned_lines),
            'ocr_engine': OCR_ENGINE
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'full_text': '',
            'lines': [],
            'text_length': 0,
            'line_count': 0
        }


def find_text_in_html(text_to_find: str, html_content: str, fuzzy: bool = True) -> Tuple[bool, float]:
    """
    Check if text appears in HTML content
    
    Args:
        text_to_find: Text to search for
        html_content: HTML source code
        fuzzy: Enable fuzzy matching for phrases
    
    Returns:
        (found, similarity_ratio)
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Extract visible text from HTML
    for tag in soup(['script', 'style', 'noscript', 'meta', 'link']):
        tag.decompose()
    
    html_text = soup.get_text(separator=' ', strip=True)
    html_text_lower = html_text.lower()
    text_lower = text_to_find.lower()
    
    # Remove extra whitespace
    html_text_lower = ' '.join(html_text_lower.split())
    text_lower = ' '.join(text_lower.split())
    
    # Exact match (case-insensitive)
    if text_lower in html_text_lower:
        return True, 1.0
    
    # Fuzzy match for phrases (80% similarity threshold)
    if fuzzy and len(text_lower.split()) >= 3:
        # Calculate similarity using SequenceMatcher
        ratio = SequenceMatcher(None, text_lower, html_text_lower).ratio()
        
        # Also check if most words are present
        words = set(text_lower.split())
        html_words = set(html_text_lower.split())
        word_overlap = len(words & html_words) / len(words) if words else 0
        
        # Consider match if either high similarity or high word overlap
        if ratio > 0.15 or word_overlap > 0.7:  # Lowered thresholds for partial matches
            return True, max(ratio, word_overlap)
    
    return False, 0.0


def detect_html_mismatch(screenshot_path: str, html_content: str) -> Dict[str, Any]:
    """
    Main OCR analysis function: Detect if screenshot shows text that's not in HTML
    
    This indicates:
    - Dynamic JavaScript errors
    - Security overlays (Cloudflare, CAPTCHA)
    - Iframes with error content
    - Client-side rendered errors
    
    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - status (str): 'DETECTED', 'SKIPPED', or 'ERROR'
            - has_mismatch (bool): Whether visible error text found
            - visible_error_text (list): List of error messages detected
            - ocr_extracted (str): Full OCR text extraction
            - confidence (float): Detection confidence 0.0-1.0
            - error (str): Error message if OCR failed
    """
    # Extract visible text from screenshot
    ocr_result = extract_visible_text(screenshot_path)
    
    if not ocr_result['success']:
        return {
            'status': 'SKIPPED',
            'has_mismatch': False,
            'visible_error_text': [],
            'ocr_extracted': '',
            'confidence': 0.0,
            'error': ocr_result.get('error', 'OCR failed')
        }
    
    # Define error/blocking keywords to look for (stricter matching)
    error_keywords = [
        # Error indicators
        'error', 'failed', 'failure', 'unsuccessful',
        # Connection/Network issues
        'could not', "couldn't", 'cannot', 'unable', 'not able',
        'connection', 'timed out', 'timeout', 'unreachable',
        # Security/Access issues
        'verify', 'verification', 'security', 'blocked', 'denied',
        'access denied', 'forbidden', 'unauthorized', 'restricted',
        # HTTP errors (with word boundaries to avoid false positives like "S5000")
        'not found', ' 404 ', ' 404', '404 ', 'error 404',
        ' 500 ', ' 500', '500 ', 'error 500', 'internal server',
        ' 502 ', ' 502', '502 ', 'bad gateway',
        ' 503 ', ' 503', '503 ', 'service unavailable',
        ' 504 ', ' 504', '504 ', 'gateway timeout',
        # Security challenges
        'captcha', 'challenge', 'cloudflare', 'ddos', 'protection',
        'checking your browser', 'moment', 'just a moment',
        # General issues
        'something went wrong', 'try again', 'please try', 'reload',
        'temporarily unavailable', 'maintenance', 'down for'
    ]
    
    # Check each line of OCR text for error keywords
    visible_error_lines = []
    
    for line in ocr_result['lines']:
        # Skip garbage text
        if is_likely_ocr_garbage(line):
            continue
        
        line_lower = line.lower()
        
        # Skip very short lines (likely OCR noise)
        if len(line.strip()) < 8:
            continue
        
        # Check if line contains error keywords
        matching_keywords = [kw for kw in error_keywords if kw in line_lower]
        
        if matching_keywords:
            # Additional validation: require meaningful context for HTTP codes
            if any(code in matching_keywords for code in [' 404 ', ' 500 ', ' 502 ', ' 503 ']):
                # Must have context words like "error", "not found", etc.
                if not any(word in line_lower for word in ['error', 'not', 'found', 'failed', 'unavailable']):
                    continue
            
            # Check if this line exists in HTML
            found_in_html, similarity = find_text_in_html(line, html_content)
            
            if not found_in_html:
                # Error text visible but not in HTML = MISMATCH!
                visible_error_lines.append({
                    'text': line,
                    'matching_keywords': matching_keywords,
                    'similarity_to_html': similarity
                })
    
    # Determine severity
    has_mismatch = len(visible_error_lines) > 0
    
    if has_mismatch:
        # Calculate confidence based on number of mismatched error lines
        # More mismatched lines = higher confidence
        confidence = min(0.95, 0.85 + (len(visible_error_lines) * 0.05))
        
        # Identify issue type based on keywords
        all_keywords = []
        for item in visible_error_lines:
            all_keywords.extend(item['matching_keywords'])
        
        issue_type = classify_error_type(all_keywords)
        
        return {
            'status': 'MISMATCH_DETECTED',
            'has_mismatch': True,
            'visible_error_text': [item['text'] for item in visible_error_lines],
            'detailed_mismatches': visible_error_lines,
            'issue_type': issue_type,
            'ocr_extracted': ocr_result['full_text'],
            'confidence': confidence,
            'total_ocr_lines': ocr_result['line_count']
        }
    else:
        return {
            'status': 'NO_MISMATCH',
            'has_mismatch': False,
            'visible_error_text': [],
            'ocr_extracted': ocr_result['full_text'],
            'confidence': 0.0,
            'total_ocr_lines': ocr_result['line_count']
        }


def classify_error_type(keywords: List[str]) -> str:
    """
    Classify the type of error based on keywords found
    """
    keywords_lower = [kw.lower() for kw in keywords]
    
    # Security/CAPTCHA
    security_kws = ['captcha', 'challenge', 'cloudflare', 'security', 'verify', 'ddos', 'protection', 'browser']
    if any(kw in keywords_lower for kw in security_kws):
        return 'security_challenge'
    
    # Network/Connection
    network_kws = ['connection', 'timeout', 'unreachable', 'network', 'could not', "couldn't"]
    if any(kw in keywords_lower for kw in network_kws):
        return 'network_error'
    
    # HTTP errors
    http_kws = ['404', '500', '502', '503', '504', 'not found', 'server', 'gateway']
    if any(kw in keywords_lower for kw in http_kws):
        return 'http_error'
    
    # Access/Auth
    auth_kws = ['denied', 'forbidden', 'unauthorized', 'restricted', 'access']
    if any(kw in keywords_lower for kw in auth_kws):
        return 'access_denied'
    
    # Generic error
    return 'generic_error'


def get_ocr_diagnosis(mismatch_result: Dict[str, Any]) -> str:
    """
    Generate human-readable diagnosis from OCR mismatch result
    """
    if not mismatch_result['has_mismatch']:
        return "No OCR mismatches detected"
    
    error_texts = mismatch_result['visible_error_text']
    issue_type = mismatch_result.get('issue_type', 'unknown')
    
    # Map issue type to diagnosis
    diagnoses = {
        'security_challenge': 'Security challenge/CAPTCHA blocking page access',
        'network_error': 'Network/connection error preventing page load',
        'http_error': 'HTTP error page displayed',
        'access_denied': 'Access denied or authentication required',
        'generic_error': 'Error message displayed on page'
    }
    
    base_diagnosis = diagnoses.get(issue_type, 'Error detected')
    
    # Add first error text as evidence
    if error_texts:
        first_error = error_texts[0][:100]  # Limit length
        return f"{base_diagnosis}: '{first_error}...'"
    
    return base_diagnosis
