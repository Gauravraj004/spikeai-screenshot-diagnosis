"""
Stage 1: Global Page Health Checks (Fast Heuristics - 2 seconds)
Catches obvious failures instantly: blank screens, duplicates, no visual content
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


def is_blank_or_uniform(img: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Multi-criteria blank detection:
    1. Histogram dominance (>95% single color)
    2. Low unique colors (<10)
    3. No edges (<1%)
    """
    if img is None or img.size == 0:
        return True
    
    h, w = img.shape[:2]
    total_pixels = h * w
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Criterion 1: Histogram dominance
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    max_color_coverage = np.max(hist) / total_pixels
    
    if max_color_coverage <= threshold:
        return False
    
    # Criterion 2: Unique colors
    unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
    if unique_colors > 10:
        return False
    
    # Criterion 3: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.count_nonzero(edges)
    edge_ratio = edge_pixels / total_pixels
    
    if edge_ratio > 0.01:
        return False
    
    return True


def detect_blocking_overlay(img: np.ndarray) -> Dict[str, Any]:
    """
    Detect overlays/modals that block page content:
    - Cookie consent banners
    - Login/auth walls
    - Security verification
    - CAPTCHA
    """
    if img is None or img.size == 0:
        return {"has_overlay": False}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Strategy 1: Check top-right corner for cookie consent
    top_right_y_start = 0
    top_right_y_end = h // 3
    top_right_x_start = w // 2
    top_right_x_end = w
    
    top_right_region = gray[top_right_y_start:top_right_y_end, top_right_x_start:top_right_x_end]
    
    # Check if there's content in top-right (cookie modal)
    tr_dark_pixels = np.sum(top_right_region < 200) / top_right_region.size
    tr_std = np.std(top_right_region)
    
    # Cookie modal: LOW text density (5-15%) vs normal headers (25%+)
    has_top_right_content = (0.05 < tr_dark_pixels < 0.15) and tr_std > 35
    
    # Also check center for modals
    center_y_start = h // 4
    center_y_end = 3 * h // 4
    center_x_start = w // 4
    center_x_end = 3 * w // 4
    
    center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
    
    # Find contours in center
    edges = cv2.Canny(center_region, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for large centered box
    has_central_box = False
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        center_area = center_region.shape[0] * center_region.shape[1]
        
        # Modal typically 20-70% of center region
        if 0.2 * center_area < area < 0.7 * center_area:
            has_central_box = True
            break
    
    # Strategy 2: Visual pattern - darkened background + bright center
    top_brightness = np.mean(gray[:h//5, :])
    center_brightness = np.mean(center_region)
    bottom_brightness = np.mean(gray[4*h//5:, :])
    
    # Brightness pattern: center brighter than both top AND bottom
    # This catches cookie banners / modals with darkened overlay
    has_modal_pattern = (center_brightness > top_brightness * 1.1 and 
                         center_brightness > bottom_brightness * 1.1)
    
    # Combine signals
    # Priority 1: Top-right cookie modal
    if has_top_right_content:
        return {
            "has_overlay": True,
            "overlay_type": "Cookie consent modal blocking content",
            "confidence": 0.90,
            "detection_method": "Top-right cookie banner"
        }
    
    # Priority 2: Centered modal with brightness pattern
    if has_central_box and has_modal_pattern:
        return {
            "has_overlay": True,
            "overlay_type": "Modal/auth wall blocking content",
            "confidence": 0.85,
            "detection_method": "Visual pattern + central box"
        }
    
    # Priority 3: Error page detection - dark page with minimal content
    # Check if page is mostly dark with a small content box in center
    dark_pixels = np.sum(gray < 50) / (h * w)  # Very dark pixels
    
    # If 70%+ of page is very dark, check if center has isolated content
    if dark_pixels > 0.70:
        # Count edge pixels in center region to detect text box
        center_edges = cv2.Canny(center_region, 50, 150)
        center_edge_density = np.count_nonzero(center_edges) / center_edges.size
        
        # Error pages: dark background + some text in center but overall sparse
        # Threshold: 0.5-3% edge density in center = error message box
        if 0.005 < center_edge_density < 0.03:
            return {
                "has_overlay": True,
                "overlay_type": "Error/security page blocking access",
                "confidence": 0.88,
                "detection_method": "Dark page with minimal centered content (error pattern)"
            }
    
    # No overlay detected
    return {"has_overlay": False}


def detect_vertical_duplication(img: np.ndarray, threshold: float = 0.65) -> Dict[str, Any]:
    """
    Detects if page content is repeated vertically (2x or 3x)
    Uses SSIM (Structural Similarity Index) + perceptual hashing for robust detection
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Try to import imagehash for robust detection
    try:
        import imagehash
        from PIL import Image
        IMAGEHASH_AVAILABLE = True
    except ImportError:
        IMAGEHASH_AVAILABLE = False
    
    if img is None or img.size == 0:
        return {"is_duplicate": False, "times": 0, "confidence": 0.0}
    
    h, w = img.shape[:2]
    
    # Check 2x duplication (top vs bottom half)
    mid_h = h // 2
    top_half = img[0:mid_h, :]
    bottom_half = img[mid_h:mid_h*2, :]
    
    # Use SSIM for 2x duplicate detection (returns scalar by default)
    similarity_2way = float(ssim(top_half, bottom_half, channel_axis=2))
    
    if similarity_2way > threshold:
        return {
            "is_duplicate": True,
            "times": 2,
            "confidence": float(similarity_2way),
            "direction": "vertical"
        }
    
    # Check 3x duplication (thirds)
    if h > 900:
        third = h // 3
        part1 = img[0:third, :]
        part2 = img[third:third*2, :]
        part3 = img[third*2:third*3, :]
        
        # Use SSIM for 3x duplicate detection
        sim_1_2 = float(ssim(part1, part2, channel_axis=2))
        sim_2_3 = float(ssim(part2, part3, channel_axis=2))
        sim_1_3 = float(ssim(part1, part3, channel_axis=2))
        
        # 3x duplication: all three sections must be consistently similar
        # AND variance between similarities should be low (true duplication, not just similar layout)
        avg_similarity = (sim_1_2 + sim_2_3 + sim_1_3) / 3
        variance = ((sim_1_2 - avg_similarity)**2 + (sim_2_3 - avg_similarity)**2 + (sim_1_3 - avg_similarity)**2) / 3
        
        # For very tall pages (>15000px), use lower threshold (stitching artifacts)
        # For normal pages, require higher similarity
        if h > 15000:
            threshold_3x = 0.48  # Lower for tall pages (stitching errors)
            max_variance = 0.01  # Must be consistent
        else:
            threshold_3x = 0.70  # Higher for normal pages
            max_variance = 0.05
        
        if avg_similarity > threshold_3x and variance < max_variance:
            return {
                "is_duplicate": True,
                "times": 3,
                "confidence": float(avg_similarity),
                "direction": "vertical"
            }
    
    # No duplication detected
    return {"is_duplicate": False, "times": 0, "confidence": 0.0}


def check_color_entropy(img: np.ndarray) -> Dict[str, Any]:
    """
    Check visual richness using color entropy + text detection.
    Low entropy ALONE is not enough - need to check for text edges.
    """
    if img is None or img.size == 0:
        return {"entropy": 0.0, "has_text_edges": False, "is_healthy": False}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_pixels = h * w
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Calculate Shannon entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Normalize to 0-1 range (max entropy for 8-bit = 8)
    normalized_entropy = entropy / 8.0
    
    # Check for text edges (horizontal edges typical of text)
    # Use adaptive thresholding to enhance text detection
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / total_pixels
    
    # Text detection: look for horizontal edge patterns
    # Text typically has strong horizontal edges (letter baselines)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)
    text_edge_pixels = np.count_nonzero(horizontal_edges)
    text_edge_density = text_edge_pixels / total_pixels
    
    # Has text if sufficient horizontal edges (>0.002 = ~2000px for 1M pixel image)
    has_text_edges = text_edge_density > 0.002
    
    # Low entropy + no text = truly blank page
    # Low entropy + text = error page with content (NOT blank)
    is_healthy = normalized_entropy > 0.3 or has_text_edges
    
    return {
        "entropy": float(normalized_entropy),
        "edge_density": float(edge_density),
        "text_edge_density": float(text_edge_density),
        "has_text_edges": has_text_edges,
        "is_healthy": is_healthy
    }


def stage1_global_checks(screenshot_path: str) -> Dict[str, Any]:
    """
    Stage 1 main function: Run all global health checks
    Returns immediately if any critical issue found
    
    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - status (str): 'PASS', 'BROKEN', or 'ERROR'
            - diagnosis (str): Issue description if status is BROKEN
            - confidence (float): Detection confidence 0.0-1.0
            - checks (dict): Individual check results (blank, overlay, duplication, entropy)
            - suggested_fix (str): Recommended action if issue found
    """
    result = {
        "status": "PASS",
        "diagnosis": None,
        "confidence": 0.0,
        "checks": {}
    }
    
    # Load image once
    img = cv2.imread(screenshot_path)
    if img is None:
        return {
            "status": "ERROR",
            "diagnosis": "Failed to load screenshot",
            "confidence": 1.0,
            "checks": {}
        }
    
    # Check 1: Blank detection
    is_blank = is_blank_or_uniform(img)
    result['checks']['blank'] = is_blank
    
    if is_blank:
        result['status'] = "BROKEN"
        result['diagnosis'] = "Completely blank screenshot"
        result['confidence'] = 0.95
        result['suggested_fix'] = "TOOL: Detect auth walls before capture, increase wait time to 15s"
        result['evidence'] = "CV Analysis: Blank page detected (histogram dominance >95%, <10 unique colors, edge ratio <1%)"
        return result
    
    # Check 2: Overlay/Modal detection
    overlay_result = detect_blocking_overlay(img)
    result['checks']['overlay'] = overlay_result
    
    if overlay_result['has_overlay']:
        result['status'] = "BROKEN"
        result['diagnosis'] = overlay_result['overlay_type']
        result['confidence'] = overlay_result['confidence']
        result['suggested_fix'] = "TOOL: Detect and dismiss overlays/modals before capture"
        result['evidence'] = f"CV Analysis: {overlay_result['detection_method']}"
        return result
    
    # Check 3: Duplicate detection
    dup_result = detect_vertical_duplication(img)
    result['checks']['duplicate'] = dup_result
    
    if dup_result['is_duplicate']:
        result['status'] = "BROKEN"
        result['diagnosis'] = f"Page repeated {dup_result['times']}x {dup_result['direction']}"
        result['confidence'] = dup_result['confidence']
        result['suggested_fix'] = "TOOL: Fix stitching algorithm - validate no duplicate DOM IDs before save"
        result['evidence'] = f"CV Analysis: SSIM-based duplication detected ({dup_result['times']}x repetition, similarity={dup_result['confidence']:.2f})"
        return result
    
    # Check 4: Visual health (entropy + text detection)
    entropy_result = check_color_entropy(img)
    result['checks']['visual_health'] = entropy_result
    
    if not entropy_result['is_healthy']:
        # Low entropy + no text = blank/missing CSS
        if not entropy_result['has_text_edges']:
            result['status'] = "BROKEN"
            result['diagnosis'] = "No visual content (blank page, missing CSS/resources)"
            result['confidence'] = 0.85
            result['suggested_fix'] = "TOOL: Check if page loaded correctly, wait for CSS/JS/images"
            result['evidence'] = f"CV Analysis: Blank page detected (entropy={entropy_result['entropy']:.2f}, edge_density={entropy_result['edge_density']:.4f}, no text edges)"
            return result
        # Low entropy + text = error/block page with content (check overlays again)
        # This catches error messages, security warnings, etc.
        # Don't flag as broken - let LLM analyze the text content
        pass
    
    # All checks passed
    result['status'] = "PASS"
    result['diagnosis'] = "Passed global health checks"
    result['confidence'] = 0.7
    
    return result
