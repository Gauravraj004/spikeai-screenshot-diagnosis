"""
Stage 1.5: Regional Analysis (Localized Issue Detection)
Detects small rendering failures in specific page regions that global checks miss

This stage catches:
- Broken web components in specific sections
- Missing charts/graphs in particular areas
- Localized rendering failures
- Small broken image regions
- Component-level errors
"""

import cv2
import numpy as np
import gc
from typing import Dict, Any, List, Tuple


def is_legitimate_minimalist_design(region: np.ndarray, position: Tuple[int, int]) -> bool:
    """
    Check if blank/dark region is legitimate design choice (reduces false positives)
    
    Legitimate patterns:
    - Footer regions (bottom row) can be sparse
    - Dark hero sections with white text overlay
    - Minimalist spacer regions
    """
    row, col = position
    
    # Footer regions (bottom row) often legitimately sparse
    if row == 2:  # Bottom row
        return True
    
    # Check for dark hero section with text (top center regions)
    if row == 0 and col in [1, 2]:  # Top center
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:  # Dark background
            # Check for white text on dark background
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text_pixels = np.count_nonzero(binary)
            
            if text_pixels > 100:  # Has some white text
                return True  # Legitimate dark hero section
    
    # Side margins (left/right edges) can be legitimately empty
    if col == 0 or col == 3:  # Left or right column
        return True
    
    return False


def divide_into_regions(img: np.ndarray, grid_size: Tuple[int, int] = (4, 3)) -> List[Dict[str, Any]]:
    """
    Divide screenshot into grid regions for localized analysis
    
    Args:
        img: Input image as numpy array
        grid_size: (columns, rows) - default 4x3 = 12 regions
    
    Returns:
        List of region dictionaries with image data and position info
    """
    h, w = img.shape[:2]
    cols, rows = grid_size
    regions = []
    
    region_h = h // rows
    region_w = w // cols
    
    for i in range(rows):
        for j in range(cols):
            region = img[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            regions.append({
                'image': region,
                'position': (i, j),  # (row, col)
                'coords': (j*region_w, i*region_h, (j+1)*region_w, (i+1)*region_h),  # (x1, y1, x2, y2)
                'region_id': i * cols + j
            })
    
    return regions


def check_region_quality(region: np.ndarray, region_id: int, position: Tuple[int, int]) -> Dict[str, Any]:
    """
    Check if a specific region has rendering issues
    
    Detects:
    - Blank/uniform regions (broken components)
    - Very low content regions (missing elements)
    - Error backgrounds (solid white/gray)
    - Suspiciously empty areas
    
    Args:
        region: Region image (numpy array)
        region_id: Unique identifier for this region
        position: (row, col) position in grid
    
    Returns:
        Dictionary with analysis results
    """
    # Check if this is an edge region (often legitimately sparse)
    row, col = position
    is_edge_region = (row == 2 and col in [0, 3])  # Bottom corners (R2C0, R2C3)
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = region.shape[:2]
    total_pixels = h * w
    
    # Calculate metrics
    # 1. Entropy (content complexity)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    
    # 2. Standard deviation (variation)
    std_dev = np.std(gray)
    
    # 3. Mean brightness
    mean_val = np.mean(gray)
    
    # 4. Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    edge_density = edge_count / total_pixels
    
    # 5. Color uniformity
    unique_colors = len(np.unique(gray))
    
    # Decision logic for problematic regions
    is_problematic = False
    issue_type = None
    confidence = 0.0
    details = ""
    
    # FIRST: Check if it's a legitimate design pattern (reduces false positives)
    if is_legitimate_minimalist_design(region, position):
        # This is OK - minimalist design, not a bug
        return {
            'is_problematic': False,
            'issue_type': None,
            'confidence': 0.0,
            'region_id': region_id,
            'position': position
        }
    
    # Pattern 1: Very low entropy + few edges = blank/broken component
    # More strict: require VERY low entropy to avoid false positives on minimalist designs
    # Skip edge regions unless they're EXTREMELY blank
    if entropy < 1.5 and edge_density < 0.005:  # Stricter thresholds
        if is_edge_region:
            # Edge regions need to be REALLY blank to flag (entropy < 0.8)
            if entropy < 0.8 and edge_density < 0.002:
                is_problematic = True
                issue_type = "blank_region"
                confidence = min(0.90, 0.60 + (0.8 - entropy) * 0.30)
                details = f"Very low content: entropy={entropy:.2f}, edges={edge_density:.4f}"
        else:
            is_problematic = True
            issue_type = "blank_region"
            confidence = min(0.90, 0.60 + (1.5 - entropy) * 0.20)
            details = f"Very low content: entropy={entropy:.2f}, edges={edge_density:.4f}"
    
    # Pattern 2: Very uniform color (error background)
    # Skip edge regions for this check (footers can be uniform)
    elif std_dev < 5 and (mean_val > 245 or mean_val < 15) and not is_edge_region:
        is_problematic = True
        issue_type = "uniform_error_region"
        confidence = min(0.85, 0.65 + (5 - std_dev) * 0.04)
        details = f"Uniform color: std={std_dev:.2f}, brightness={mean_val:.0f}"
    
    # Pattern 3: Very few unique colors (degraded rendering)
    elif unique_colors < 10 and entropy < 2.5:
        is_problematic = True
        issue_type = "low_color_diversity"
        confidence = 0.70
        details = f"Low diversity: {unique_colors} unique colors, entropy={entropy:.2f}"
    
    # Pattern 4: Mostly gray (common error placeholder color)
    gray_pixels = np.sum((gray > 100) & (gray < 180))
    gray_ratio = gray_pixels / total_pixels
    if gray_ratio > 0.7 and std_dev < 15:
        is_problematic = True
        issue_type = "gray_placeholder_region"
        confidence = 0.75
        details = f"Gray placeholder: {gray_ratio*100:.1f}% gray pixels"
    
    return {
        'region_id': region_id,
        'position': position,
        'is_problematic': is_problematic,
        'issue_type': issue_type,
        'confidence': confidence,
        'details': details,
        'metrics': {
            'entropy': float(entropy),
            'edge_density': float(edge_density),
            'std_dev': float(std_dev),
            'mean_brightness': float(mean_val),
            'unique_colors': int(unique_colors)
        }
    }


def stage1_5_regional_checks(screenshot_path: str, max_size: int = 1920) -> Dict[str, Any]:
    """
    Main Stage 1.5 function: Regional analysis for localized issues
    
    Args:
        screenshot_path: Path to screenshot image
        max_size: Maximum width to resize image for memory optimization (default: 1920px)
    
    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - status (str): 'PASS', 'BROKEN', or 'ERROR'
            - diagnosis (str): Issue description if problems found
            - confidence (float): Detection confidence 0.0-1.0
            - has_regional_issues (bool): Whether localized problems detected
            - problematic_regions (list): List of problem region details
            - suggested_fix (str): Recommended action if issues found
    """
    # Load image
    img = cv2.imread(screenshot_path)
    if img is None:
        return {
            'status': 'ERROR',
            'diagnosis': 'Could not load screenshot',
            'confidence': 0.0,
            'has_regional_issues': False,
            'problematic_regions': []
        }
    
    # Memory optimization: Resize large images
    h, w = img.shape[:2]
    if w > max_size:
        scale = max_size / w
        new_w = max_size
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Divide into regions (4 columns x 3 rows = 12 regions)
    regions = divide_into_regions(img, grid_size=(4, 3))
    
    # Analyze each region
    problematic_regions = []
    for region_data in regions:
        result = check_region_quality(
            region_data['image'],
            region_data['region_id'],
            region_data['position']
        )
        
        if result['is_problematic']:
            problematic_regions.append({
                'region_id': result['region_id'],
                'position': result['position'],
                'coords': region_data['coords'],
                'issue_type': result['issue_type'],
                'confidence': result['confidence'],
                'details': result['details'],
                'metrics': result['metrics']
            })
    
    # Determine overall status
    if len(problematic_regions) == 0:
        return {
            'status': 'CORRECT',
            'diagnosis': 'All regions appear normal',
            'confidence': 0.85,
            'has_regional_issues': False,
            'problematic_regions': [],
            'total_regions_checked': len(regions),
            'issue_count': 0
        }
    
    # NEW: Ignore single low-confidence issues (likely false positives)
    # Only flag if: 3+ regions OR 2+ regions with high confidence
    avg_confidence = np.mean([r['confidence'] for r in problematic_regions])
    high_confidence_issues = sum(1 for r in problematic_regions if r['confidence'] > 0.75)
    
    if len(problematic_regions) <= 2 and high_confidence_issues == 0:
        # Too few issues with low confidence - likely false positive
        return {
            'status': 'CORRECT',
            'diagnosis': 'Minor variations detected but within normal range',
            'confidence': 0.80,
            'has_regional_issues': False,
            'problematic_regions': [],  # Don't report them
            'total_regions_checked': len(regions),
            'issue_count': 0,
            'note': f'{len(problematic_regions)} low-confidence regions ignored'
        }
    
    # NEW: Check if center regions (R1C1, R1C2) are OK - indicates error page with central content
    # Error pages often have blank edges but content in center
    # Region ID mapping: 4 cols x 3 rows, so R1C1 = region 5, R1C2 = region 6
    center_region_ids = [5, 6]  # Middle row (1), center columns (1, 2)
    problematic_ids = [r['region_id'] for r in problematic_regions]
    center_ok = all(cr not in problematic_ids for cr in center_region_ids)
    
    # If center is OK but edges are blank, need to distinguish:
    # 1. Normal minimalist page with centered content = CORRECT
    # 2. Error/security page with just a message = BROKEN
    # Key difference: Error pages have VERY FEW regions with actual content (just 1-3)
    if center_ok and len(problematic_regions) <= 8:
        # Check if issues are mostly in edge/corner regions
        # Region IDs: 0=R0C0, 3=R0C3, 4=R1C0, 7=R1C3, 8=R2C0, 9=R2C1, 10=R2C2, 11=R2C3
        edge_region_ids = [0, 3, 4, 7, 8, 11]  # Corners + sides (not center)
        edge_issues = sum(1 for r in problematic_regions if r['region_id'] in edge_region_ids)
        edge_ratio = edge_issues / len(problematic_regions) if problematic_regions else 0
        
        # Count how many regions are actually OK (not problematic)
        ok_regions = len(regions) - len(problematic_regions)
        
        # If 50%+ issues are in edges and center is OK:
        # - If 6+ regions OK = real page with content (CORRECT)
        # - If only 3-5 regions OK = error page with minimal content (BROKEN)
        if edge_ratio >= 0.5:
            if ok_regions >= 6:
                # Real page with good content distribution
                return {
                    'status': 'CORRECT',
                    'diagnosis': 'Center regions contain content, edge sparsity is normal design pattern',
                    'confidence': 0.75,
                    'has_regional_issues': False,
                    'problematic_regions': [],
                    'total_regions_checked': len(regions),
                    'issue_count': 0,
                    'note': f'{ok_regions} regions with content - normal centered design'
                }
            else:
                # Too few regions with content - likely error/blocking page
                # Don't return here, fall through to normal BROKEN handling
                pass
    
    # Calculate severity based on number and confidence of issues
    issue_ratio = len(problematic_regions) / len(regions)
    
    # Determine if issues are critical
    if issue_ratio > 0.5:  # More than half regions broken
        severity = "CRITICAL"
        diagnosis = f"Multiple regions broken ({len(problematic_regions)}/{len(regions)})"
        overall_confidence = min(0.95, avg_confidence + 0.1)
    elif issue_ratio > 0.25:  # More than quarter broken
        severity = "MAJOR"
        diagnosis = f"Several regions with issues ({len(problematic_regions)}/{len(regions)})"
        overall_confidence = avg_confidence
    else:
        severity = "MINOR"
        diagnosis = f"Localized issue in {len(problematic_regions)} region(s)"
        overall_confidence = max(0.70, avg_confidence - 0.05)
    
    # Build detailed diagnosis
    issue_types = [r['issue_type'] for r in problematic_regions]
    unique_issues = list(set(issue_types))
    
    if len(unique_issues) == 1:
        diagnosis += f" - {unique_issues[0].replace('_', ' ')}"
    else:
        diagnosis += f" - mixed issues: {', '.join(unique_issues[:2])}"
    
    # Store counts before cleanup
    total_regions = len(regions)
    issue_count = len(problematic_regions)
    
    # Memory optimization: Explicit garbage collection after processing
    del img, regions
    gc.collect()
    
    return {
        'status': 'BROKEN',
        'diagnosis': diagnosis,
        'severity': severity,
        'confidence': overall_confidence,
        'has_regional_issues': True,
        'problematic_regions': problematic_regions,
        'total_regions_checked': total_regions,
        'issue_count': issue_count,
        'suggested_fix': generate_fix_suggestion(problematic_regions)
    }


def generate_fix_suggestion(problematic_regions: List[Dict[str, Any]]) -> str:
    """Generate actionable fix suggestion based on regional issues"""
    if not problematic_regions:
        return "No action needed"
    
    issue_types = [r['issue_type'] for r in problematic_regions]
    
    if 'blank_region' in issue_types:
        return "TOOL: Check component rendering - some areas failed to load completely"
    elif 'gray_placeholder_region' in issue_types:
        return "TOOL: Increase wait time for lazy-loaded content or scroll through page"
    elif 'uniform_error_region' in issue_types:
        return "TOOL: Investigate component errors - uniform color regions suggest rendering failure"
    else:
        return "TOOL: Verify page fully loaded before capture - some regions show incomplete rendering"


def format_regional_evidence(regional_result: Dict[str, Any]) -> str:
    """Format regional analysis results into readable evidence string"""
    if not regional_result.get('has_regional_issues'):
        return "Regional analysis: All 12 regions checked, no localized issues found"
    
    regions = regional_result.get('problematic_regions', [])
    if not regions:
        return "Regional analysis: No issues detected"
    
    count = len(regions)
    total = regional_result.get('total_regions_checked', 12)
    severity = regional_result.get('severity', 'MINOR')
    
    # Get sample positions
    positions = [f"R{r['position'][0]}C{r['position'][1]}" for r in regions[:3]]
    position_str = ", ".join(positions)
    if count > 3:
        position_str += f" and {count-3} more"
    
    return f"Regional analysis ({severity}): {count}/{total} regions affected at {position_str}"
