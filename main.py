# -*- coding: utf-8 -*-
"""
INTEGRATED CV + LLM SCREENSHOT DIAGNOSIS PIPELINE
Combines computer vision and semantic analysis for comprehensive web QA

APPROACH:
- Stage 1 (CV Global): Fast visual checks - catches blank, overlays, duplicates, low entropy
- Stage 1.5 (CV Regional): Localized analysis - catches broken components in specific areas  
- Stage 1.75 (OCR): Text extraction & HTML mismatch - catches dynamic errors, overlays (NEW!)
- Stage 2 (LLM): Semantic HTML analysis - explains WHY issues occur, catches context problems
- Integration: Combines all stages for higher accuracy and confidence

BENEFITS OF INTEGRATION:
✓ Stage 1 catches global visual issues
✓ Stage 1.5 catches localized component failures
✓ Stage 2 catches semantic issues that CV can't understand
✓ Combined findings reduce false positives/negatives
✓ Higher confidence when stages agree
✓ Cost-efficient: CV filters obvious cases, LLM provides deep analysis

Uses free Groq API for LLM analysis
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import stage functions
import stage1_global as stage1
import stage1_5_regional as stage1_5
import ocr_analysis  # NEW: OCR mismatch detection (Stage 1.75)
import llm_analyzer  # LLM-based universal analysis

# Import Excel export
from src.excel_export import create_excel_report

# Define directories
HTML_DIR = Path("data/html")
SCREENSHOTS_DIR = Path("data/screenshots")
RESULTS_DIR = Path("results")


def calculate_token_cost(tokens: int) -> float:
    """
    Calculate cost for Groq API token usage
    Groq Llama 3.3 70B pricing: $0.59 per 1M input tokens, $0.79 per 1M output tokens
    Using average: $0.69 per 1M tokens
    """
    cost_per_million = 0.69
    return (tokens / 1_000_000) * cost_per_million


def preprocess_screenshot(screenshot_path: str, max_size: int = 1920) -> str:
    """
    Preprocess screenshot: resize if too large for memory optimization
    Returns path to processed image (same path, modified in-place)
    """
    import cv2
    img = cv2.imread(screenshot_path)
    if img is None:
        return screenshot_path
    
    h, w = img.shape[:2]
    if w > max_size:
        scale = max_size / w
        new_w = max_size
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(screenshot_path, img)
        print(f"  📏 Resized: {w}x{h} → {new_w}x{new_h}")
    
    return screenshot_path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_checkpoint(results: List[Dict[str, Any]], checkpoint_path: str):
    """
    Save partial results as checkpoint for recovery
    """
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({
            'completed_cases': [r['case_name'] for r in results],
            'results': results,
            'timestamp': time.time()
        }, f, indent=2, cls=NumpyEncoder)


def load_checkpoint(checkpoint_path: str) -> List[Dict[str, Any]]:
    """
    Load checkpoint if exists
    """
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('results', [])
        except:
            return []
    return []


def get_case_files(html_dir: Path, screenshots_dir: Path) -> List[Dict[str, str]]:
    """
    Get all test cases by matching HTML and screenshot files
    """
    cases = []
    
    # Get all screenshot files
    if not screenshots_dir.exists():
        return cases
    
    screenshot_files = list(screenshots_dir.glob("*.png"))
    
    for screenshot_path in screenshot_files:
        case_name = screenshot_path.stem  # filename without extension
        html_path = html_dir / f"{case_name}.html"
        
        cases.append({
            "case_name": case_name,
            "screenshot_path": str(screenshot_path),
            "html_path": str(html_path)
        })
    
    return cases


def generate_user_friendly_explanation(status: str, root_cause: str, diagnosis: str) -> str:
    """Convert technical diagnosis to plain English for end users"""
    
    if status == 'CORRECT':
        return "✅ This page looks good! All elements loaded correctly and the page appears functional."
    
    # Map technical issues to user-friendly explanations
    explanations = {
        'vertical': "⚠️ The page content is repeating multiple times (shown 2-3 times vertically). This usually means the page rendering failed and duplicated content instead of loading properly.",
        'horizontal': "⚠️ The page content is repeating side-by-side. This indicates a rendering issue where content was duplicated horizontally.",
        'cookie': "⚠️ A cookie consent popup is blocking the main content. The user needs to accept or dismiss the cookie banner before accessing the page.",
        'security': "⚠️ The website is showing a security check page (like Cloudflare protection or CAPTCHA). The actual website content couldn't load because of this security challenge.",
        'blank': "❌ The page is completely blank - nothing loaded at all. This could be due to a failed connection, authentication requirement, or page timeout.",
        'region': "⚠️ Some parts of the page didn't load properly. Specific sections are blank or missing content, though other parts may look fine.",
        'entropy': "⚠️ The page appears mostly empty or stuck on a loading screen. There's minimal content visible.",
        'overlay': "⚠️ A modal or popup is covering the main content, preventing access to the underlying page.",
        'http': "❌ The page returned an HTTP error (like 404 Not Found, 500 Internal Server Error). The server couldn't provide the requested content.",
        'network': "❌ Network connection failed or timed out. The page couldn't load due to connectivity issues.",
        'challenge': "⚠️ The website requires verification (CAPTCHA or bot detection). Access is temporarily restricted.",
        'maintenance': "⚠️ The website is down for maintenance or temporarily unavailable."
    }
    
    # Find matching explanation
    root_lower = root_cause.lower()
    diagnosis_lower = diagnosis.lower()
    combined = root_lower + ' ' + diagnosis_lower
    
    for key, explanation in explanations.items():
        if key in combined:
            return explanation
    
    # Default technical explanation
    return f"⚠️ Issue detected: {root_cause}"

def generate_suggested_fix(diagnosis: str, status: str, issue_type: str = None) -> str:
    """
    Generate actionable fix suggestions based on detected issue type.
    
    Args:
        diagnosis: Issue description string
        status: Final status (correct/broken/uncertain)
        issue_type: Optional specific issue type
    
    Returns:
        str: Suggested fix action for the capture tool
    """
    if status == 'correct':
        return "No action needed - screenshot captured successfully"
    
    diagnosis_lower = diagnosis.lower()
    
    # Cookie/consent modal
    if any(word in diagnosis_lower for word in ['cookie', 'consent', 'modal', 'banner']):
        return "TOOL: Wait 2-3s after page load, detect and auto-dismiss cookie/consent modals before capture"
    
    # Blank/empty page
    if any(word in diagnosis_lower for word in ['blank', 'empty', 'black']):
        return "TOOL: Increase page load timeout to 10-15s, wait for DOM ready event, retry with longer delay"
    
    # Duplication/stitching issues
    if any(word in diagnosis_lower for word in ['duplicate', 'repeat', 'stitch', 'vertical', '2x', '3x']):
        return "TOOL: Fix stitching algorithm - validate no duplicate DOM IDs before save, adjust scroll overlap detection"
    
    # Security/CAPTCHA/anti-bot
    if any(word in diagnosis_lower for word in ['security', 'captcha', 'challenge', 'verify', 'cloudflare', 'bot']):
        return "TOOL: Implement security challenge detection and human intervention workflow, use residential proxies"
    
    # Regional/component issues
    if any(word in diagnosis_lower for word in ['region', 'component', 'localized']):
        return "TOOL: Increase wait time for lazy-loaded components, implement viewport-specific rendering checks"
    
    # Authentication/login required
    if any(word in diagnosis_lower for word in ['auth', 'login', 'sign in', 'credentials']):
        return "TOOL: Implement authentication flow, store session cookies, handle login before capture"
    
    # Low entropy/loading screen
    if any(word in diagnosis_lower for word in ['entropy', 'loading', 'spinner']):
        return "TOOL: Wait for network idle (no requests for 2s), check for absence of loading indicators"
    
    # Overlay detection
    if 'overlay' in diagnosis_lower:
        return "TOOL: Detect and dismiss popups/overlays using z-index analysis, click overlay close buttons"
    
    # Generic fallback
    return "TOOL: Review capture timing, increase wait delays, check for dynamic content blockers"


def diagnose_screenshot(case: Dict[str, str]) -> Dict[str, Any]:
    """
    ENHANCED INTEGRATED DIAGNOSIS PIPELINE
    Combines CV Global (Stage 1), CV Regional (Stage 1.5), and LLM (Stage 2) for comprehensive analysis
    
    Returns:
        Dict[str, Any]: Comprehensive diagnosis including:
            - status: 'correct' | 'broken' | 'uncertain'
            - diagnosis: Human-readable issue description
            - confidence: float (0.0-1.0)
            - suggested_fix: Actionable tool configuration change
            - root_cause: Primary failure reason
            - evidence_summary: Combined findings from all stages
            - stage_results: Individual stage outputs
            - token_usage: LLM cost metrics
    
    Flow:
    1. Preprocess: Resize large screenshots for memory optimization
    2. Stage 1 (CV Global): Fast visual checks - catches obvious issues (blank, overlay, duplicate)
    3. Stage 1.5 (CV Regional): Localized analysis - catches broken components in specific areas
    4. Stage 2 (LLM): Semantic HTML analysis - explains WHY and catches context issues
    5. Integration: Combines findings from all stages for complete diagnosis
    """
    case_name = case['case_name']
    screenshot_path = case['screenshot_path']
    html_path = case['html_path']
    
    # Preprocess: Resize large screenshots early for all stages
    screenshot_path = preprocess_screenshot(screenshot_path)
    
    result = {
        "case_name": case_name,
        "status": "UNKNOWN",
        "diagnosis": "",
        "confidence": 0.0,
        "suggested_fix": "",
        "stage_results": {},
        "evidence": {
            "cv_global_findings": [],
            "cv_regional_findings": [],
            "llm_findings": []
        }
    }
    
    # Initialize token tracking
    total_tokens = 0
    token_breakdown = {}
    
    print(f"\n{'='*60}")
    print(f"Processing: {case_name}")
    print(f"{'='*60}")
    
    # ========== STAGE 1: CV GLOBAL CHECKS ==========
    print("Stage 1: Global CV Analysis...")
    print("  Checking: blank/uniform, overlays, duplication, entropy")
    stage1_result = stage1.stage1_global_checks(screenshot_path)
    result['stage_results']['stage1_global'] = stage1_result
    
    # Collect Stage 1 evidence
    cv_global_findings = []
    if stage1_result.get('diagnosis'):
        cv_global_findings.append(f"Global: {stage1_result['diagnosis']}")
    if stage1_result.get('evidence'):
        cv_global_findings.append(stage1_result['evidence'])
    result['evidence']['cv_global_findings'] = cv_global_findings
    
    print(f"  Status: {stage1_result['status']}")
    if cv_global_findings:
        print(f"  Findings: {'; '.join(cv_global_findings)}")
    
    # ========== STAGE 1.5: CV REGIONAL CHECKS ==========
    print("\nStage 1.5: Regional CV Analysis...")
    print("  Checking: localized rendering issues in 12 regions")
    stage1_5_result = stage1_5.stage1_5_regional_checks(screenshot_path)
    result['stage_results']['stage1_5_regional'] = stage1_5_result
    
    # Collect Stage 1.5 evidence
    cv_regional_findings = []
    if stage1_5_result.get('has_regional_issues'):
        regional_evidence = stage1_5.format_regional_evidence(stage1_5_result)
        cv_regional_findings.append(f"Regional: {stage1_5_result['diagnosis']}")
        cv_regional_findings.append(regional_evidence)
    result['evidence']['cv_regional_findings'] = cv_regional_findings
    
    print(f"  Status: {stage1_5_result['status']}")
    if cv_regional_findings:
        print(f"  Findings: {'; '.join(cv_regional_findings)}")
    
    # ========== STAGE 1.75: OCR + HTML MISMATCH DETECTION (NEW!) ==========
    print("\nStage 1.75: OCR & HTML Mismatch Analysis...")
    print("  Extracting visible text from screenshot...")
    
    # Smart skip: If Stage 1 has high confidence (>0.90), skip OCR to save time
    skip_ocr = stage1_result.get('confidence', 0) > 0.90 and stage1_result['status'] == 'BROKEN'
    
    if skip_ocr:
        stage1_conf = stage1_result.get('confidence', 0)
        print(f"  ⚡ Skipping OCR (Stage 1 confidence: {stage1_conf:.2f})")
        ocr_result = {'status': 'SKIPPED', 'has_mismatch': False, 'confidence': 0.0}
    else:
        # Load HTML content for mismatch check
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        ocr_result = ocr_analysis.detect_html_mismatch(screenshot_path, html_content)
    
    result['stage_results']['ocr_mismatch'] = ocr_result
    
    # Collect OCR evidence
    ocr_findings = []
    if ocr_result.get('has_mismatch'):
        ocr_diagnosis = ocr_analysis.get_ocr_diagnosis(ocr_result)
        ocr_findings.append(f"OCR Mismatch: {ocr_diagnosis}")
        ocr_findings.append(f"Visible text not in HTML: {', '.join(ocr_result['visible_error_text'][:2])}")
    result['evidence']['ocr_mismatch_findings'] = ocr_findings
    
    if ocr_result.get('status') == 'MISMATCH_DETECTED':
        print(f"  ⚠ MISMATCH DETECTED!")
        print(f"  Visible error text not in HTML:")
        for text in ocr_result['visible_error_text'][:3]:  # Show first 3
            display_text = text[:80] + '...' if len(text) > 80 else text
            print(f"    - '{display_text}'")
        print(f"  Issue Type: {ocr_result.get('issue_type', 'unknown')}")
        print(f"  Confidence: {ocr_result.get('confidence', 0.0):.2f}")
    elif ocr_result.get('status') == 'NO_MISMATCH':
        print(f"  ✓ No mismatches - visible text matches HTML")
    else:
        print(f"  ⚠ OCR skipped: {ocr_result.get('error', 'Unknown error')}")
    
    # ========== STAGE 2: LLM SEMANTIC ANALYSIS ==========
    # Always run LLM for semantic understanding (complements CV)
    print("\nStage 2: LLM Semantic Analysis...")
    print("  (Analyzing HTML structure & context)")
    
    # Pass CV findings AND OCR findings to LLM for enhanced context
    # Combine global and regional findings for LLM context
    cv_context_parts = []
    if stage1_result['status'] == 'BROKEN':
        cv_context_parts.append(stage1_result.get('diagnosis', ''))
    if stage1_5_result.get('has_regional_issues'):
        cv_context_parts.append(stage1_5_result.get('diagnosis', ''))
    cv_context = ' | '.join(cv_context_parts) if cv_context_parts else None
    
    # Try LLM with graceful degradation
    llm_result = None
    llm_available = True
    try:
        llm_result = llm_analyzer.diagnose_with_llm(html_path, screenshot_path, case_name, cv_context, ocr_result)
        result['stage_results']['llm_analysis'] = llm_result
    except Exception as e:
        print(f"  ⚠ LLM unavailable: {e}")
        llm_available = False
        llm_result = {
            'status': 'ERROR',
            'diagnosis': 'LLM unavailable',
            'confidence': 0.0,
            'error': str(e)
        }
        result['stage_results']['llm_analysis'] = llm_result
    
    # Track LLM token usage
    if 'token_usage' in llm_result:
        token_breakdown['llm'] = llm_result['token_usage']
        total_tokens += llm_result['token_usage'].get('total_tokens', 0)
    
    # Collect LLM evidence
    llm_findings = []
    if llm_result.get('diagnosis'):
        llm_findings.append(f"Semantic: {llm_result['diagnosis']}")
    if llm_result.get('evidence'):
        llm_findings.append(llm_result['evidence'])
    result['evidence']['llm_findings'] = llm_findings
    
    print(f"  Status: {llm_result['status']}")
    if llm_findings:
        print(f"  Findings: {'; '.join(llm_findings)}")
    
    # ========== INTEGRATION: COMBINE ALL FOUR STAGES (Including OCR) ==========
    print("\n> Integrating Stage 1 + Stage 1.5 + Stage 1.75 (OCR) + Stage 2 findings...")
    
    # Aggregate all findings
    all_cv_findings = cv_global_findings + cv_regional_findings
    all_findings = all_cv_findings + ocr_findings + llm_findings
    cv_has_issues = stage1_result['status'] == 'BROKEN' or stage1_5_result.get('has_regional_issues', False)
    ocr_has_mismatch = ocr_result.get('has_mismatch', False)
    
    # PRIORITY 0: OCR Mismatch = HIGHEST PRIORITY (definitive proof of error)
    if ocr_has_mismatch:
        result['status'] = 'broken'
        ocr_diagnosis = ocr_analysis.get_ocr_diagnosis(ocr_result)
        result['diagnosis'] = ocr_diagnosis
        result['root_cause'] = ocr_diagnosis
        result['confidence'] = ocr_result.get('confidence', 0.90)
        
        # Boost confidence if CV also detected issues
        if cv_has_issues:
            result['confidence'] = min(0.98, result['confidence'] + 0.05)
        
        result['suggested_fix'] = f"TOOL: {ocr_result.get('issue_type', 'error')} detected - address blocking issue"
        result['evidence_summary'] = f"OCR MISMATCH (highest priority). {' '.join(all_findings)}"
        print(f"  ⚠⚠⚠ BROKEN (OCR mismatch: {result['confidence']:.2f})")
    
    # Scenario 1: ALL stages found issues - HIGHEST confidence BROKEN
    elif cv_has_issues and llm_result['status'] == 'BROKEN':
        result['status'] = 'broken'
        
        # Build comprehensive diagnosis
        diagnosis_parts = []
        if stage1_result['status'] == 'BROKEN':
            diagnosis_parts.append(f"Global: {stage1_result['diagnosis']}")
        if stage1_5_result.get('has_regional_issues'):
            diagnosis_parts.append(f"Regional: {stage1_5_result['diagnosis']}")
        diagnosis_parts.append(f"Semantic: {llm_result['diagnosis']}")
        
        result['diagnosis'] = ' | '.join(diagnosis_parts)
        result['root_cause'] = llm_result.get('diagnosis', stage1_result.get('diagnosis', stage1_5_result.get('diagnosis')))
        
        # Highest confidence when all agree
        avg_conf = (stage1_result.get('confidence', 0.8) + 
                   stage1_5_result.get('confidence', 0.8) + 
                   llm_result['confidence']) / 3
        result['confidence'] = min(0.97, avg_conf + 0.15)
        
        result['suggested_fix'] = generate_suggested_fix(result['diagnosis'], result['status'])
        result['evidence_summary'] = f"ALL STAGES DETECTED ISSUES. {' '.join(all_cv_findings + llm_findings)}"
        print(f"  ✓✓✓ HIGHEST CONFIDENCE BROKEN (all stages agree: {result['confidence']:.2f})")
    
    # Scenario 2: Only CV (global OR regional) found issues
    elif cv_has_issues and llm_result['status'] != 'BROKEN':
        result['status'] = 'broken'
        
        # Prioritize global over regional issues
        if stage1_result['status'] == 'BROKEN':
            result['diagnosis'] = stage1_result['diagnosis']
            result['root_cause'] = stage1_result['diagnosis']
            result['confidence'] = stage1_result['confidence']
            result['suggested_fix'] = generate_suggested_fix(result['diagnosis'], result['status'])
        else:
            result['diagnosis'] = stage1_5_result['diagnosis']
            result['root_cause'] = stage1_5_result['diagnosis']
            result['confidence'] = stage1_5_result['confidence']
            result['suggested_fix'] = generate_suggested_fix(result['diagnosis'], result['status'])
        
        result['evidence_summary'] = f"CV detected visual issue. LLM context: {llm_result.get('diagnosis', 'HTML structure appears normal')}"
        print(f"  ⚠ BROKEN (CV detection: {result['confidence']:.2f})")
    
    # Scenario 3: Only LLM found issue - semantic/context problem
    elif not cv_has_issues and llm_result['status'] == 'BROKEN':
        result['status'] = 'broken'
        result['diagnosis'] = llm_result['diagnosis']
        result['root_cause'] = llm_result['diagnosis']
        result['confidence'] = llm_result['confidence']
        result['suggested_fix'] = generate_suggested_fix(result['diagnosis'], result['status'])
        result['evidence_summary'] = f"LLM detected semantic issue. CV: All visual metrics passed. {' '.join(llm_findings)}"
        print(f"  ⚠ BROKEN (LLM detection: {result['confidence']:.2f})")
    
    # Scenario 4: All stages passed - HIGHEST confidence CORRECT
    elif not cv_has_issues and llm_result['status'] == 'CORRECT':
        result['status'] = 'correct'
        result['diagnosis'] = 'Page appears correct - all checks passed (global, regional, semantic)'
        result['root_cause'] = 'none'
        
        avg_conf = (stage1_result.get('confidence', 0.85) + 
                   stage1_5_result.get('confidence', 0.85) + 
                   llm_result['confidence']) / 3
        result['confidence'] = min(0.96, avg_conf + 0.08)
        
        result['suggested_fix'] = 'No action needed'
        result['evidence_summary'] = 'CV Global: Passed. CV Regional: All 12 regions OK. LLM: HTML structure normal.'
        print(f"  ✓✓✓ HIGHEST CONFIDENCE CORRECT (all stages agree: {result['confidence']:.2f})")
    
    # Scenario 4b: CV passed, LLM ERROR (rate limit) - treat as CORRECT with note
    elif not cv_has_issues and llm_result['status'] == 'ERROR':
        result['status'] = 'correct'
        result['diagnosis'] = 'Page appears correct - CV checks passed (LLM unavailable)'
        result['root_cause'] = 'none'
        
        avg_conf = (stage1_result.get('confidence', 0.85) + 
                   stage1_5_result.get('confidence', 0.85)) / 2
        result['confidence'] = min(0.88, avg_conf + 0.03)  # Lower confidence without LLM
        
        result['suggested_fix'] = 'No action needed'
        result['evidence_summary'] = f'CV Global: Passed. CV Regional: All 12 regions OK. LLM: {llm_result.get("diagnosis", "unavailable")}'
        print(f"  ✓✓ CORRECT (CV passed, LLM unavailable: {result['confidence']:.2f})")
    
    # Scenario 5: Conflicting or unclear results
    else:
        result['status'] = 'uncertain'
        result['diagnosis'] = f"Unclear: Global={stage1_result['status']}, Regional={stage1_5_result['status']}, LLM={llm_result['status']}"
        result['root_cause'] = 'unclear - conflicting signals'
        result['confidence'] = 0.5
        result['suggested_fix'] = 'MANUAL: Review recommended - stages gave conflicting results (CV vs LLM disagreement)'
        result['evidence_summary'] = f"Conflicting results. {' '.join(all_cv_findings + llm_findings)}"
        print(f"  ? UNCERTAIN (conflicting results: {result['confidence']:.2f})")
    
    # Add token tracking to result
    result['token_usage'] = {
        'total_tokens': total_tokens,
        'breakdown': token_breakdown,
        'estimated_cost_usd': calculate_token_cost(total_tokens)
    }
    
    # Print token summary
    if total_tokens > 0:
        print(f"\n  💰 Tokens used: {total_tokens:,} (${result['token_usage']['estimated_cost_usd']:.6f})")
    
    return result


def main():
    """Main execution"""
    print("="*80)
    print("INTEGRATED 4-STAGE SCREENSHOT DIAGNOSIS PIPELINE (WITH OCR)")
    print("="*80)
    print("\nIntegrated Approach:")
    print("  Stage 1 (CV Global): Fast checks - blank, overlay, duplication, entropy")
    print("  Stage 1.5 (CV Regional): Localized checks - 12 regions for component issues")
    print("  Stage 1.75 (OCR): Extract visible text, detect HTML mismatches (NEW!)")
    print("  Stage 2 (LLM): Semantic HTML analysis - context, structure, blockers")
    print("  Integration: Combines all findings for comprehensive diagnosis")
    print("\n  - Catches both global and localized rendering issues")
    print("  - Higher accuracy through multi-stage validation")
    print("  - Reduced false positives by cross-validation")
    print("  - Cost-efficient: CV is free, LLM provides deep analysis")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get all test cases
    cases = get_case_files(HTML_DIR, SCREENSHOTS_DIR)
    
    if not cases:
        print("\n❌ ERROR: No test cases found!")
        print(f"Check directories: {HTML_DIR} and {SCREENSHOTS_DIR}")
        return
    
    print(f"\n\nFound {len(cases)} test cases\n")
    
    # Check for checkpoint
    checkpoint_path = os.path.join(RESULTS_DIR, ".checkpoint.json")
    all_results = load_checkpoint(checkpoint_path)
    completed_cases = {r['case_name'] for r in all_results}
    remaining_cases = [c for c in cases if c['case_name'] not in completed_cases]
    
    if all_results:
        print(f"📦 Loaded checkpoint: {len(all_results)} cases already completed")
        print(f"🔄 Resuming with {len(remaining_cases)} remaining cases\n")
    
    # Process metrics
    stage1_global_caught = 0
    stage1_5_regional_caught = 0
    stage2_llm_caught = 0
    all_stages_agreed = 0
    total_run_tokens = 0
    total_run_cost = 0.0
    
    start_time = time.time()
    
    # Parallel processing with ThreadPoolExecutor
    # Use max_workers=2 to reduce memory usage and avoid rate limits  
    # Higher values cause memory issues with OCR (200MB per worker)
    max_workers = min(2, len(remaining_cases)) if len(remaining_cases) > 0 else 1
    
    if max_workers > 1 and len(remaining_cases) > 1:
        # Process in batches to prevent memory issues with large datasets
        BATCH_SIZE = 15  # Process 15 cases at a time max
        
        for batch_start in range(0, len(remaining_cases), BATCH_SIZE):
            batch = remaining_cases[batch_start:batch_start + BATCH_SIZE]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (len(remaining_cases) - 1) // BATCH_SIZE + 1
            
            if total_batches > 1:
                print(f"\n{'='*70}")
                print(f"📦 Batch {batch_num}/{total_batches} - Processing {len(batch)} cases")
                print(f"{'='*70}\n")
            else:
                print(f"⚡ Parallel processing with {max_workers} workers\n")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks for this batch
                future_to_case = {executor.submit(diagnose_screenshot, case): case for case in batch}
            
                # Process completed tasks as they finish
                for i, future in enumerate(as_completed(future_to_case), 1):
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        # Save checkpoint after each completion
                        save_checkpoint(all_results, checkpoint_path)
                        
                        # Track total tokens and cost
                        if 'token_usage' in result:
                            total_run_tokens += result['token_usage'].get('total_tokens', 0)
                            total_run_cost += result['token_usage'].get('estimated_cost_usd', 0.0)
                        
                        # Force garbage collection to prevent memory leaks
                        import gc
                        gc.collect()
                        
                        # Save individual JSON
                        output_path = os.path.join(RESULTS_DIR, f"{result['case_name']}.json")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                "case_name": result['case_name'],
                                "status": result['status'],
                                "root_cause": result.get('root_cause', ''),
                                "diagnosis": result['diagnosis'],
                                "suggested_fix": result['suggested_fix'],
                                "confidence": result['confidence'],
                                "evidence": result.get('evidence', {}),  # Keep as dict
                                "evidence_summary": result.get('evidence_summary', ''),  # Add summary
                                "token_usage": result.get('token_usage', {})
                            }, f, indent=2, cls=NumpyEncoder)
                        
                        completed_so_far = len(all_results)
                        print(f"✓ Progress: {completed_so_far}/{len(cases)} total completed")
                        
                    except Exception as e:
                        case_name = future_to_case[future]['case_name']
                        print(f"✗ Error processing {case_name}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Force garbage collection between batches
                import gc
                gc.collect()
                
                if total_batches > 1 and batch_num < total_batches:
                    print(f"\n♻️  Memory cleaned for next batch\n")
    else:
        # Sequential processing for single case or small batches
        for case in remaining_cases:
            try:
                result = diagnose_screenshot(case)
                all_results.append(result)
                
                # Save checkpoint after each completion
                save_checkpoint(all_results, checkpoint_path)
                
                # Track total tokens and cost
                if 'token_usage' in result:
                    total_run_tokens += result['token_usage'].get('total_tokens', 0)
                    total_run_cost += result['token_usage'].get('estimated_cost_usd', 0.0)
                
                # Save individual JSON
                output_path = os.path.join(RESULTS_DIR, f"{result['case_name']}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "case_name": result['case_name'],
                        "status": result['status'],
                        "root_cause": result.get('root_cause', ''),
                        "diagnosis": result['diagnosis'],
                        "suggested_fix": result['suggested_fix'],
                        "confidence": result['confidence'],
                        "evidence": result.get('evidence', {}),  # Keep as dict
                        "evidence_summary": result.get('evidence_summary', ''),  # Add summary
                        "token_usage": result.get('token_usage', {})
                    }, f, indent=2, cls=NumpyEncoder)
            except Exception as e:
                print(f"✗ Error processing {case['case_name']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Count metrics from all results
    for result in all_results:
        stage1_broken = result['stage_results'].get('stage1_global', {}).get('status') == 'BROKEN'
        stage1_5_broken = result['stage_results'].get('stage1_5_regional', {}).get('has_regional_issues', False)
        stage2_broken = result['stage_results'].get('llm_analysis', {}).get('status') == 'BROKEN'
        
        if stage1_broken:
            stage1_global_caught += 1
        if stage1_5_broken:
            stage1_5_regional_caught += 1
        if stage2_broken:
            stage2_llm_caught += 1
        
        if (stage1_broken or stage1_5_broken) and stage2_broken:
            all_stages_agreed += 1
    
    elapsed = time.time() - start_time
    
    # Clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    # Generate Excel and CSV reports
    print("\n\n" + "="*80)
    print("Generating reports...")
    
    try:
        from src.excel_export import export_to_excel, create_csv_report
        excel_path = os.path.join(os.path.dirname(RESULTS_DIR), "diagnosis_report.xlsx")
        csv_path = os.path.join(os.path.dirname(RESULTS_DIR), "diagnosis_report.csv")
        
        export_to_excel(all_results, excel_path)
        create_csv_report(all_results, csv_path)
        
        print(f"> Excel report: {excel_path}")
        print(f"> CSV report: {csv_path}")
    except Exception as e:
        print(f"⚠ Report generation error: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("INTEGRATED CV + LLM DIAGNOSIS SUMMARY")
    print("="*80)
    print(f"{'Case':<20} {'Status':<12} {'Conf':<8} {'Issue':<38}")
    print("-"*80)
    
    for r in all_results:
        if r['status'] == 'correct':
            status_label = "✓ CORRECT"
        elif r['status'] == 'broken':
            status_label = "✗ BROKEN"
        elif r['status'] == 'uncertain':
            status_label = "? UNCERTAIN"
        else:
            status_label = "⚠ ERROR"
        
        conf = f"{r['confidence']:.2f}"
        issue = r.get('root_cause', 'none')[:35] + "..." if len(r.get('root_cause', '')) > 35 else r.get('root_cause', 'none')
        print(f"{r['case_name']:<20} {status_label:<12} {conf:<8} {issue:<38}")
    
    print("-"*80)
    broken = sum(1 for r in all_results if r['status'] == 'broken')
    correct = sum(1 for r in all_results if r['status'] == 'correct')
    uncertain = sum(1 for r in all_results if r['status'] == 'uncertain')
    errors = sum(1 for r in all_results if r['status'] == 'error')
    
    print(f"\nResults: {len(all_results)} total | ✓ Correct: {correct} | ✗ Broken: {broken} | ? Uncertain: {uncertain} | ⚠ Errors: {errors}")
    
    print(f"\n3-Stage Integration Metrics:")
    print(f"  Stage 1 (Global CV) Detections: {stage1_global_caught} ({stage1_global_caught/len(cases)*100:.0f}%)")
    print(f"  Stage 1.5 (Regional CV) Detections: {stage1_5_regional_caught} ({stage1_5_regional_caught/len(cases)*100:.0f}%)")
    print(f"  Stage 2 (LLM) Detections: {stage2_llm_caught} ({stage2_llm_caught/len(cases)*100:.0f}%)")
    print(f"  Multi-stage Agreement: {all_stages_agreed}/{len(cases)} ({all_stages_agreed/len(cases)*100:.0f}%)")
    
    print(f"\n⚡ Performance: {elapsed:.1f} seconds total ({elapsed/len(cases):.2f}s per case)")
    
    # Token usage summary
    if total_run_tokens > 0:
        print(f"\n💰 Token Usage:")
        print(f"  Total tokens: {total_run_tokens:,}")
        print(f"  Average per case: {total_run_tokens/len(cases):.0f} tokens")
        print(f"  Total cost: ${total_run_cost:.6f}")
        print(f"  Cost per case: ${total_run_cost/len(cases):.6f}")
    
    print(f"\n📁 Reports saved:")
    print(f"  - JSON: {RESULTS_DIR}/")
    print(f"  - Excel: diagnosis_report.xlsx")
    print(f"  - CSV: diagnosis_report.csv")
    print("="*80)


if __name__ == "__main__":
    main()
