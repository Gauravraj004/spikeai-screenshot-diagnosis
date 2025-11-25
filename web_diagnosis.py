"""
WEB RENDERING DIAGNOSIS SYSTEM
===============================

Advanced visual diagnosis system that analyzes web page screenshots to detect
rendering issues, UI bugs, and user experience problems.

Capabilities:
- Detects content duplication (repeating sections, elements)
- Identifies blocking modals (cookie consent, security gates)
- Recognizes blank/failed pages
- Analyzes layout integrity and visual hierarchy
- Provides HTML-level insights into root causes

Uses GPT-4o vision with unlimited token analysis for comprehensive diagnosis.
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv
import pandas as pd
from bs4 import BeautifulSoup
import re
from PIL import Image
import io

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

SCREENSHOTS_DIR = Path("data/screenshots")
HTML_DIR = Path("data/html")
RESULTS_DIR = Path("diagnosis_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration: Split tall images for full detail analysis
# Set to True to analyze full image detail (higher cost but more accurate)
# Set to False to use OpenAI's automatic optimization (lower cost)
SPLIT_TALL_IMAGES = False  # Change to True for full detail analysis
MAX_HEIGHT_BEFORE_SPLIT = 4096  # Split images taller than this


def analyze_html_structure(html_path: Path) -> Dict[str, Any]:
    """
    Analyze HTML structure to identify potential rendering issues AND
    provide actionable insights for screenshot capture improvements.
    
    Returns structural insights about:
    - Duplicate element patterns
    - Modal/overlay elements
    - DOM depth and complexity
    - Actual text content extraction and comparison
    - JavaScript rendering patterns
    - Capture timing recommendations
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all major content containers (use word-boundary matching)
        semantic_container_terms = ['content', 'main', 'article', 'section', 'container']
        def _is_semantic_container(class_attr):
            if not class_attr:
                return False
            txt = str(class_attr)
            return any(re.search(r"\b" + re.escape(term) + r"\b", txt, re.I) for term in semantic_container_terms)

        main_sections = soup.find_all(['main', 'article', 'section', 'div'], class_=_is_semantic_container)
        
        # Detect modals and overlays
        modals = soup.find_all(['div', 'section'], 
                              class_=lambda x: x and any(term in str(x).lower() 
                              for term in ['modal', 'overlay', 'popup', 'cookie', 'consent']))
        
        # Extract full text content from all detected main sections (no truncation)
        section_texts = []
        for section in main_sections:
            text = section.get_text(strip=True)
            if text and len(text) > 50:  # Filter out very small containers
                section_texts.append(text)
        
        # Detect identical text sections (real duplication)
        text_duplicates = []
        seen = {}
        for idx, text in enumerate(section_texts):
            if text in seen:
                text_duplicates.append({
                    'section_index': idx,
                    'duplicate_of': seen[text],
                    'text_preview': text[:100]
                })
            else:
                seen[text] = idx
        
        # Find repeated class patterns (more precise matching)
        all_classes = []
        for tag in soup.find_all(class_=True):
            all_classes.extend(tag.get('class', []))
        
        class_counts = {}
        for cls in all_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Only flag truly suspicious patterns (high count + semantic meaning)
        suspicious_duplication = []
        semantic_terms = ['article', 'post', 'entry', 'card', 'tile', 'list', 'item']
        # Blacklist common utility substrings to avoid false positives
        blacklist_subs = ['btn', 'text-', 'align-', 'col-', 'container-fluid', 'row', 'nav', 'icon', 'fa-']
        class_duplication_threshold = 10
        for cls, count in class_counts.items():
            name = cls.lower()
            if count >= class_duplication_threshold and any(re.search(r"\b" + re.escape(term) + r"\b", name) for term in semantic_terms):
                if not any(sub in name for sub in blacklist_subs):
                    suspicious_duplication.append({
                        'class_name': cls,
                        'occurrence_count': count,
                        'note': 'High repetition - check if content is identical'
                    })
        
        # Calculate DOM depth
        def get_max_depth(element, depth=0):
            if not element or not hasattr(element, 'children'):
                return depth
            children_with_names = [child for child in element.children if hasattr(child, 'name') and child.name]
            if not children_with_names:
                return depth
            return max(get_max_depth(child, depth + 1) for child in children_with_names)
        
        max_depth = get_max_depth(soup.body) if soup.body else 0
        
        # Detect footer with CTA (normal pattern)
        footer = soup.find('footer') or soup.find(class_=lambda x: x and 'footer' in str(x).lower())
        has_footer_cta = False
        if footer:
            footer_text = footer.get_text(strip=True).lower()
            has_footer_cta = any(term in footer_text for term in ['subscribe', 'sign up', 'newsletter'])
        
        # ==== NEW: ADVANCED CAPTURE ANALYSIS ====
        
        # 1. Detect JavaScript-heavy SPAs and frameworks (more signals)
        script_tags = soup.find_all('script')
        framework_indicators = {
            'react': any('react' in str(script.get('src', '')).lower() for script in script_tags) or bool(soup.find(attrs={'data-reactroot': True})) or ('react' in html_content.lower()),
            'vue': any('vue' in str(script.get('src', '')).lower() for script in script_tags) or bool(soup.find(attrs=lambda k, v: k and 'vue' in k.lower())),
            'angular': any('angular' in str(script.get('src', '')).lower() for script in script_tags) or 'ng-app' in html_content or 'angular' in html_content.lower(),
            'next': ('_next' in html_content) or bool(soup.find(id='__next')) or bool(re.search(r"window\.__NEXT_DATA__", html_content)),
        }
        is_spa = any(framework_indicators.values())
        
        # 2. Detect loading indicators (page might not be ready)
        loading_indicators = soup.find_all(class_=lambda x: x and any(
            term in str(x).lower() for term in ['loading', 'spinner', 'skeleton', 'placeholder']
        ))
        
        # 3. Detect lazy-loaded content
        lazy_images = soup.find_all('img', attrs={'loading': 'lazy'}) or \
                     soup.find_all(attrs={'data-src': True})
        
        # 4. Detect infinite scroll patterns
        has_infinite_scroll = bool(soup.find_all(class_=lambda x: x and any(
            term in str(x).lower() for term in ['infinite', 'load-more', 'pagination']
        )))
        
        # 5. Check for dynamic content markers
        empty_containers = soup.find_all(['div', 'section'], 
                                        class_=lambda x: x and any(
                                            term in str(x).lower() for term in 
                                            ['container', 'content', 'main']
                                        ))
        empty_containers_count = sum(1 for c in empty_containers if len(c.get_text(strip=True)) < 50)
        
        # 6. Analyze script complexity
        inline_scripts = [s for s in script_tags if not s.get('src')]
        total_script_size = sum(len(s.string or '') for s in inline_scripts)
        has_heavy_js = total_script_size > 10000 or len(script_tags) > 10
        
        # 7. Check for AJAX/fetch patterns in scripts
        has_ajax_patterns = 'fetch(' in html_content or 'XMLHttpRequest' in html_content or \
                           'axios' in html_content or '$.ajax' in html_content
        
        return {
            'total_elements': len(soup.find_all()),
            'main_sections_count': len(main_sections),
            'modal_elements_count': len(modals),
            'dom_depth': max_depth,
            'text_content_duplicates': text_duplicates,
            'suspicious_class_patterns': suspicious_duplication[:3],
            'has_cookie_consent': 'cookie' in html_content.lower(),
            'has_footer_cta': has_footer_cta,
            'modal_element_ids': [m.get('id', 'no-id') for m in modals[:3]],
            # NEW FIELDS
            'capture_analysis': {
                'is_spa': is_spa,
                'frameworks_detected': [k for k, v in framework_indicators.items() if v],
                'loading_indicators_count': len(loading_indicators),
                'lazy_images_count': len(lazy_images),
                'has_infinite_scroll': has_infinite_scroll,
                'empty_containers': empty_containers_count,
                'script_complexity': 'high' if has_heavy_js else 'normal',
                'has_ajax_patterns': has_ajax_patterns,
                'modals_count': len(modals)
            }
        }
    except Exception as e:
        return {
            'error': f'HTML analysis failed: {str(e)}',
            'total_elements': 0
        }


def diagnose_screenshot(screenshot_path: Path, case_name: str) -> Dict[str, Any]:
    """
    Comprehensive visual diagnosis using GPT-4o with unlimited tokens.
    
    Analyzes screenshot to detect ANY type of rendering issue, not limited
    to predefined patterns. VLM provides deep analysis of visual problems.
    """
    print(f"\n{'='*80}")
    print(f"[*] Analyzing: {case_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Check image dimensions and warn about tall images
    try:
        import struct
        import imghdr
        
        with open(screenshot_path, 'rb') as fhandle:
            head = fhandle.read(24)
            if len(head) >= 24:
                # PNG signature check
                if imghdr.what(screenshot_path) == 'png':
                    check = struct.unpack('>i', head[4:8])[0]
                    if check == 0x0d0a1a0a:  # PNG signature
                        img_width, img_height = struct.unpack('>ii', head[16:24])
                        
                        if img_height > MAX_HEIGHT_BEFORE_SPLIT and not SPLIT_TALL_IMAGES:
                            print(f"[!] WARNING: Very tall image ({img_width}×{img_height} px)")
                            print(f"    OpenAI resizes images >~4096px - VLM may not see bottom details")
                            print(f"    For full detail analysis, set SPLIT_TALL_IMAGES=True (higher cost)")
                        elif img_height > MAX_HEIGHT_BEFORE_SPLIT and SPLIT_TALL_IMAGES:
                            print(f"[*] Tall image detected ({img_width}×{img_height} px)")
                            print(f"    SPLIT_TALL_IMAGES=True: Will analyze in full detail chunks")
    except:
        pass  # Skip dimension check if it fails
    
    # Load and process screenshot (with optional splitting for tall images)
    image_content_parts = []
    
    if SPLIT_TALL_IMAGES:
        # Split tall images into chunks for full detail analysis
        try:
            img = Image.open(screenshot_path)
            img_width, img_height = img.size
            
            if img_height > MAX_HEIGHT_BEFORE_SPLIT:
                print(f"[*] Splitting tall image into chunks for full detail analysis...")
                
                # Calculate number of chunks needed
                chunk_height = MAX_HEIGHT_BEFORE_SPLIT
                num_chunks = (img_height + chunk_height - 1) // chunk_height  # Ceiling division
                
                for i in range(num_chunks):
                    top = i * chunk_height
                    bottom = min((i + 1) * chunk_height, img_height)
                    
                    # Crop chunk
                    chunk = img.crop((0, top, img_width, bottom))
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    chunk.save(buffer, format='PNG')
                    chunk_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    image_content_parts.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{chunk_base64}',
                            'detail': 'high'
                        }
                    })
                    
                print(f"    Split into {num_chunks} chunks ({chunk_height}px each)")
            else:
                # Image not tall, use as-is
                with open(screenshot_path, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                    image_content_parts.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}',
                            'detail': 'high'
                        }
                    })
        except Exception as e:
            print(f"[!] Image splitting failed: {e}, using full image...")
            with open(screenshot_path, 'rb') as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
                image_content_parts.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}',
                        'detail': 'high'
                    }
                })
    else:
        # No splitting - use full image (OpenAI will auto-resize)
        with open(screenshot_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
            image_content_parts.append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{base64_image}',
                    'detail': 'high'
                }
            })
    
    # Analyze HTML structure if available (handle _correct suffix)
    # Try exact match first, then without _correct suffix
    html_path = HTML_DIR / f"{case_name}.html"
    if not html_path.exists() and case_name.endswith('_correct'):
        # Try without _correct suffix
        base_name = case_name.rsplit('_correct', 1)[0]
        html_path = HTML_DIR / f"{base_name}.html"
    
    html_analysis = analyze_html_structure(html_path) if html_path.exists() else {}
    html_file_found = html_path.exists()
    
    # Extract key structural elements from HTML for validation
    html_validation_note = ""
    if html_analysis and html_analysis.get('total_elements', 0) < 100:
        html_validation_note = f"\n⚠️ **LOW HTML ELEMENT COUNT ({html_analysis['total_elements']} elements)**: This suggests a JavaScript-rendered SPA (Single Page App). The HTML is just a skeleton - actual content loads via JS. If screenshot shows partial/incomplete content, this is likely a RENDER FAILURE or JS ERROR."
    
    # Universal diagnostic prompt - handles ANY rendering issue
    # Special handling for marketing pages with footer CTAs
    marketing_context = ""
    if html_analysis.get('has_footer_cta'):
        marketing_context = "\n**IMPORTANT**: HTML indicates this is a marketing page with a footer CTA. Having the same subscription form at the top AND bottom of the page is a STANDARD MARKETING PATTERN, not a bug."
    
    # Add note about image chunks if splitting is enabled
    image_note = ""
    if SPLIT_TALL_IMAGES and len(image_content_parts) > 1:
        image_note = f"\n**NOTE**: This tall page has been split into {len(image_content_parts)} sequential image chunks for full detail analysis. Analyze all chunks together as one continuous page."
    
    prompt = f"""You are an expert web rendering diagnostician analyzing a screenshot of a web page.

Your task: Determine if this page renders correctly or has rendering issues.{image_note}

**HTML STRUCTURE VALIDATION:**{html_validation_note}

**CRITICAL RULES FOR DETECTING ISSUES:**

1. **Partial Page Loads / Incomplete Rendering (BROKEN)**:
   - Page shows header/nav but main content area is blank/minimal
   - Visible loading spinners, "Loading..." text, or skeleton screens
   - Large white/empty spaces where content should be
   - Only top 20-30% of expected page is visible
   - Sign: HTML has <100 elements but screenshot shows minimal content

2. **Overlays Blocking Content (BROKEN)**:
   - Cookie consent modal covering 50%+ of page WITH no dismiss button visible
   - Security check overlay (Cloudflare) blocking ALL content
   - Modal/popup preventing ANY interaction with main content
   - Sign: User cannot access page without action, but action is impossible

3. **CRITICAL RULES FOR MARKETING PAGES:**

1. **Blog/Marketing Pages with Multiple CTAs = CORRECT**
   - Hero section at top with subscription form = NORMAL
   - Footer at bottom with same subscription form = NORMAL MARKETING PATTERN
   - This is NOT duplication - it's intentional conversion optimization
   - Status: CORRECT unless other issues present

2. **Blog Article Lists = CORRECT**
   - Grid/list of blog cards with titles/excerpts = NORMAL
   - 5-15 similar article cards = STANDARD BLOG LAYOUT
   - Status: CORRECT unless cards have identical content

3. **Real Duplication Issues (BROKEN)**:
   - ENTIRE article/section body repeats 2-3x with IDENTICAL text
   - Same paragraph appearing multiple times in succession
   - Visual evidence: scroll down and see the exact same large content block (300px+) repeating

**WHAT TO ACTUALLY FLAG AS BROKEN:**
- **Partial Page Load**: Header visible but main content missing/minimal (especially if HTML has <100 elements)
- **Incomplete JS Rendering**: Page stuck loading, shows skeleton/placeholder, large empty areas
- **Overlays Blocking Access**: Modal/overlay covering 50%+ with NO visible close/dismiss button
- **Massive Content Duplication**: The MAIN BODY content (articles, paragraphs) repeats identically 2-3x
- **Blocking Modals**: Modal covering 80%+ of page preventing ALL interaction
- **Blank Pages**: Completely empty, no content at all
- **Security Blocks**: Full-page Cloudflare "checking your browser" blocking everything
- **Complete Render Failures**: Broken CSS, unstyled content, major layout collapse

**WHAT IS NORMAL (CORRECT):**
- Header at top + Footer at bottom with links/CTAs = STANDARD
- Hero/CTA at page top + same CTA in footer = MARKETING BEST PRACTICE
- List of blog articles with thumbnails and excerpts = BLOG DESIGN
- Navigation menu repeated in header and footer = NORMAL
- "Subscribe" appearing twice (hero + footer) = INTENTIONAL{marketing_context}

**HTML ANALYSIS (if available):**
{json.dumps(html_analysis, indent=2) if html_analysis else 'No HTML data available'}

**TEXT CONTENT DUPLICATION CHECK:**
{'✅ HTML analysis found text content duplicates - check if MAIN BODY repeats' if html_analysis.get('text_content_duplicates') else '✅ No text content duplication detected in HTML'}

**HTML ANALYSIS DATA:**
{json.dumps(html_analysis, indent=2) if html_analysis else 'No HTML data available'}

**YOUR TASK - INTELLIGENT ANALYSIS:**
You have access to:
1. The VISUAL screenshot - what the page actually looks like
2. The HTML STRUCTURE - element counts, frameworks, class patterns, text content
3. The HTML FILE metadata - total elements, DOM depth, script complexity

Use ALL of this information together to make intelligent decisions:
- Look at what the screenshot SHOWS visually
- Cross-reference with HTML structure (e.g., "HTML has 48 elements but screenshot is blank")
- Identify the ROOT CAUSE (e.g., "Low element count + blank visual = SPA captured too early")
- Determine the BEST capture strategy for THIS specific page

Do NOT just count things and apply rules. UNDERSTAND what's happening:
- WHY does this page look this way?
- WHAT does the HTML tell us about loading behavior?
- HOW should we capture THIS specific page?

Think like an engineer debugging a capture issue, not a pattern matcher.

**OUTPUT FORMAT (JSON):**
{{
    "status": "CORRECT" or "BROKEN",
    "issue_type": "partial_load" | "js_render_failure" | "blocking_overlay" | "duplicate_content" | "blocking_modal" | "security_block" | "blank_page" | "layout_break" | "missing_assets" | "normal_page" | "other",
    "severity": "critical" | "major" | "minor" | "none",
    "confidence": 0.0-1.0,
    "confidence_reasoning": "Explain why you have this confidence level",
    "visual_description": "Detailed description of what you see in the screenshot",
    "diagnosis": "Clear explanation of the issue (if any)",
    "evidence": "Specific visual evidence - measurements, locations, text content seen",
    "html_correlation": "If HTML data available, explain how structure relates to visual issue",
    "user_impact": "How this affects user experience",
    "root_cause": "What likely caused this rendering issue",
    "capture_improvement": "Analyze the visual evidence + HTML data. What capture strategy would prevent this issue? Be specific and thoughtful.",
    "capture_recommendations": {{
        "primary_issue": "What is the main capture problem you identified?",
        "wait_strategy": "What wait approach makes sense? (time-based/selector-based/network-idle/hybrid) - explain why",
        "wait_duration": "How long to wait? Base this on the specific page characteristics",
        "selectors_to_wait_for": ["Which specific CSS selectors should we wait for? Suggest realistic ones based on HTML"],
        "scroll_needed": true/false - "Does this page need scrolling? Why or why not?",
        "modal_handling": "Are there modals to handle? What's the best approach?",
        "technical_implementation": "Provide complete working code snippet that solves THIS specific case"
    }}
}}

**CRITICAL**: Return ONLY valid JSON. Ensure all string values are properly escaped:
- Use \\n for newlines within strings
- Escape quotes as \\"
- Do not include actual line breaks in string values
- All strings must be on a single line or properly escaped

Provide thorough, detailed analysis."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[API] Analyzing with GPT-4o vision model... {'(retry ' + str(attempt + 1) + ')' if attempt > 0 else ''}")
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o',
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt}
                        ] + image_content_parts  # Add all image chunks
                    }],
                    # NO max_tokens limit - allow unlimited analysis
                    'temperature': 0.1
                },
                timeout=180  # Longer timeout for detailed analysis
            )
            
            if response.status_code != 200:
                error_msg = f'API error: {response.status_code} - {response.text}'
                if attempt < max_retries - 1:
                    print(f"   [!] {error_msg}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {
                        'case': case_name,
                        'status': 'ERROR',
                        'error': error_msg,
                        'processing_time': time.time() - start_time
                    }
            
            break
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"   [!] Timeout, retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    'case': case_name,
                    'status': 'ERROR',
                    'error': 'Timeout after retries',
                    'processing_time': time.time() - start_time
                }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   [!] Error: {str(e)}, retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    'case': case_name,
                    'status': 'ERROR',
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
    
    # Parse response
    try:
        data = response.json()
        usage = data.get('usage', {})
        tokens_in = usage.get('prompt_tokens', 0)
        tokens_out = usage.get('completion_tokens', 0)
        total_tokens = tokens_in + tokens_out
        
        # Calculate image tokens separately
        # Image tokens are included in prompt_tokens but we can estimate:
        # For 'detail: high', approximate formula:
        # Base: 85 tokens + (170 * number_of_tiles)
        # We'll extract from usage if available, otherwise estimate
        prompt_tokens_details = usage.get('prompt_tokens_details', {})
        image_tokens = prompt_tokens_details.get('cached_tokens', 0) if 'cached_tokens' in prompt_tokens_details else 0
        
        # If not available in details, estimate based on typical high-detail image
        # Note: This is an approximation - exact count comes from OpenAI's tokenizer
        if image_tokens == 0:
            # Rough estimate: 85% of input tokens are typically from image for large screenshots
            image_tokens = int(tokens_in * 0.85)
        
        text_prompt_tokens = tokens_in - image_tokens
        
        # Cost calculation (GPT-4o pricing)
        cost = (tokens_in * 2.50 + tokens_out * 10.00) / 1_000_000
        
        # Parse VLM response
        content = data['choices'][0]['message']['content']
        
        # Extract JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        # Try to parse JSON, handle malformed responses
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            # VLM sometimes includes unescaped newlines in strings
            # Try to fix common issues
            print(f"   [!] JSON parse issue, attempting repair...")
            
            # Replace unescaped newlines within strings
            import re
            # Find all string values and escape newlines within them
            content_fixed = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: m.group(0).replace('\n', '\\n'), content)
            
            try:
                result = json.loads(content_fixed)
                print(f"   [OK] JSON repaired successfully")
            except json.JSONDecodeError:
                # Last resort: ask VLM to regenerate with proper escaping
                raise json.JSONDecodeError(f"Could not parse VLM response even after repair. Original error: {e}", content, e.pos)
        
        processing_time = time.time() - start_time
        
        print(f"\n[OK] Analysis Complete:")
        print(f"   Status: {result['status']}")
        print(f"   Issue Type: {result['issue_type']}")
        print(f"   Severity: {result['severity']}")
        print(f"   Confidence: {result['confidence']:.0%}")
        print(f"   Confidence Reasoning: {result['confidence_reasoning']}")
        print(f"\n[DIAGNOSIS] {result['diagnosis']}")
        
        # Print VLM-generated capture recommendations
        if result.get('capture_recommendations'):
            print(f"\n[CAPTURE RECOMMENDATIONS] VLM Analysis:")
            capture_recs = result['capture_recommendations']
            print(f"\n   Primary Issue: {capture_recs.get('primary_issue', 'N/A')}")
            print(f"   Wait Strategy: {capture_recs.get('wait_strategy', 'N/A')}")
            print(f"   Wait Duration: {capture_recs.get('wait_duration', 'N/A')}")
            if capture_recs.get('selectors_to_wait_for'):
                print(f"   Selectors: {', '.join(capture_recs['selectors_to_wait_for'][:3])}")
            print(f"   Scroll Needed: {capture_recs.get('scroll_needed', False)}")
            if capture_recs.get('modal_handling'):
                print(f"   Modal Handling: {capture_recs['modal_handling']}")
        
        if result.get('capture_improvement'):
            print(f"\n[VLM] Detailed Recommendation: {result['capture_improvement'][:250]}...")
        
        print(f"\n[METRICS] Token Usage:")
        print(f"   Input Tokens: {tokens_in:,}")
        print(f"     ├─ Image Tokens: ~{image_tokens:,}")
        print(f"     └─ Text Prompt Tokens: ~{text_prompt_tokens:,}")
        print(f"   Output Tokens: {tokens_out:,}")
        print(f"   Total Tokens: {total_tokens:,}")
        print(f"   Cost: ${cost:.4f}")
        print(f"   Time: {processing_time:.2f}s")
        
        # Save detailed JSON for this case
        case_result = {
            'case_name': case_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': {
                'status': result['status'],
                'issue_type': result['issue_type'],
                'severity': result['severity'],
                'confidence': result['confidence'],
                'confidence_reasoning': result['confidence_reasoning'],
                'visual_description': result['visual_description'],
                'diagnosis': result['diagnosis'],
                'evidence': result['evidence'],
                'html_correlation': result.get('html_correlation', 'N/A'),
                'user_impact': result['user_impact'],
                'root_cause': result['root_cause'],
                'capture_improvement': result.get('capture_improvement', 'N/A')
            },
            'html_structure_analysis': html_analysis,
            'capture_recommendations': result.get('capture_recommendations', {}),
            'metrics': {
                'tokens_input_total': tokens_in,
                'tokens_input_image': image_tokens,
                'tokens_input_text': text_prompt_tokens,
                'tokens_output': tokens_out,
                'tokens_total': total_tokens,
                'cost_usd': cost,
                'processing_time_seconds': processing_time
            }
        }
        
        # Save individual JSON file
        json_path = RESULTS_DIR / f"{case_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)
        print(f"   [SAVED] {json_path}")
        
        # Format capture recommendations for CSV from VLM response
        capture_rec_text = ""
        if result.get('capture_recommendations'):
            recs = result['capture_recommendations']
            capture_rec_text = f"""Primary Issue: {recs.get('primary_issue', 'N/A')}
Wait Strategy: {recs.get('wait_strategy', 'N/A')}
Wait Duration: {recs.get('wait_duration', 'N/A')}
Selectors: {', '.join(recs.get('selectors_to_wait_for', [])) if recs.get('selectors_to_wait_for') else 'None'}
Scroll Needed: {recs.get('scroll_needed', False)}
Modal Handling: {recs.get('modal_handling', 'N/A')}
Technical Implementation: {recs.get('technical_implementation', 'N/A')}"""
        
        return {
            'case': case_name,
            'status': result['status'],
            'issue_type': result['issue_type'],
            'severity': result['severity'],
            'confidence': result['confidence'],
            'diagnosis': result['diagnosis'],  # FULL text, no truncation
            'capture_improvement': result.get('capture_improvement', 'N/A'),  # FULL text
            'top_recommendation': result.get('capture_recommendations', {}).get('primary_issue', 'N/A'),
            'capture_recommendations_details': capture_rec_text,  # VLM-generated recommendations formatted
            'html_element_count': html_analysis.get('total_elements', 0),
            'is_spa': html_analysis.get('capture_analysis', {}).get('is_spa', False),
            'frameworks_detected': ', '.join(html_analysis.get('capture_analysis', {}).get('frameworks_detected', [])) or 'None',
            'tokens_in_total': tokens_in,
            'tokens_in_image': image_tokens,
            'tokens_in_text': text_prompt_tokens,
            'tokens_out': tokens_out,
            'total_tokens': total_tokens,
            'cost': cost,
            'processing_time': processing_time
        }
        
    except Exception as e:
        print(f"[ERROR] Parse Error: {str(e)}")
        print(f"   Raw response: {content[:200]}...")
        return {
            'case': case_name,
            'status': 'ERROR',
            'error': f'Parse error: {str(e)}',
            'processing_time': time.time() - start_time
        }


def main():
    print("\n" + "="*80)
    print("WEB RENDERING DIAGNOSIS SYSTEM")
    print("="*80)
    print("\nAdvanced visual analysis using GPT-4o with unlimited token depth")
    print("Detects ANY type of rendering issue through comprehensive inspection")
    print("+ Capture improvement recommendations\n")
    
    # Get all screenshots
    screenshots = sorted(list(SCREENSHOTS_DIR.glob("*.png")))
    
    if not screenshots:
        print("X No screenshots found in data/screenshots/")
        return
    
    print(f"Found {len(screenshots)} screenshots to analyze\n")
    
    # Process each screenshot
    results = []
    for i, screenshot_path in enumerate(screenshots, 1):
        case_name = screenshot_path.stem
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(screenshots)}]")
        result = diagnose_screenshot(screenshot_path, case_name)
        results.append(result)
    
    # Save summary CSV (with retry if file is locked)
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "diagnosis_summary.csv"
    
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        # File is open in another program, try alternative name
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = RESULTS_DIR / f"diagnosis_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[!] Note: Original CSV was locked, saved as {csv_path.name}")
    
    # Calculate statistics
    total_cases = len(results)
    error_cases = [r for r in results if r.get('status') == 'ERROR']
    successful_cases = [r for r in results if r.get('status') != 'ERROR']
    broken_cases = [r for r in successful_cases if r.get('status') == 'BROKEN']
    correct_cases = [r for r in successful_cases if r.get('status') == 'CORRECT']
    
    # Summary
    print("\n" + "="*80)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*80)
    
    print(f"\n📈 Results:")
    print(f"   Total Analyzed: {total_cases}")
    print(f"   Correct Pages: {len(correct_cases)}")
    print(f"   Broken Pages: {len(broken_cases)}")
    if error_cases:
        print(f"   Errors: {len(error_cases)}")
    
    # Issue type breakdown
    if successful_cases:
        print(f"\n🔍 Issue Types Detected:")
        issue_counts = {}
        for r in successful_cases:
            issue_type = r.get('issue_type', 'unknown')
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {issue_type}: {count}")
    
    # Cost analysis
    total_tokens_in = sum(r.get('tokens_in_total', 0) for r in results)
    total_tokens_in_image = sum(r.get('tokens_in_image', 0) for r in results)
    total_tokens_in_text = sum(r.get('tokens_in_text', 0) for r in results)
    total_tokens_out = sum(r.get('tokens_out', 0) for r in results)
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    total_cost = sum(r.get('cost', 0) for r in results)
    avg_tokens_in = total_tokens_in / len(results) if results else 0
    avg_tokens_in_image = total_tokens_in_image / len(results) if results else 0
    avg_tokens_in_text = total_tokens_in_text / len(results) if results else 0
    avg_tokens_out = total_tokens_out / len(results) if results else 0
    avg_cost = total_cost / len(results) if results else 0
    
    print(f"\n💰 Token Usage & Cost Analysis:")
    print(f"   Total Input Tokens: {total_tokens_in:,}")
    print(f"     ├─ Image Tokens: ~{total_tokens_in_image:,} ({100*total_tokens_in_image/total_tokens_in:.1f}%)")
    print(f"     └─ Text Tokens: ~{total_tokens_in_text:,} ({100*total_tokens_in_text/total_tokens_in:.1f}%)")
    print(f"   Total Output Tokens: {total_tokens_out:,}")
    print(f"   Total Tokens: {total_tokens:,}")
    print(f"   Total Cost: ${total_cost:.4f}")
    print(f"   Average Input/Case: {avg_tokens_in:,.0f} ({avg_tokens_in_image:,.0f} image + {avg_tokens_in_text:,.0f} text)")
    print(f"   Average Output/Case: {avg_tokens_out:,.0f}")
    print(f"   Average Cost/Case: ${avg_cost:.4f}")
    
    # Performance
    total_time = sum(r.get('processing_time', 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"\n⏱️  Performance:")
    print(f"   Total Time: {total_time:.1f}s")
    print(f"   Average Time/Case: {avg_time:.1f}s")
    
    print(f"\n📁 Results saved to:")
    print(f"   Summary CSV: {csv_path}")
    print(f"   Individual JSONs: {RESULTS_DIR}/*.json")
    print(f"\n💡 Note: OpenAI automatically optimizes very tall images (>4k px)")
    print(f"   This reduces cost but VLM may miss details at bottom of page")
    print(f"   Set SPLIT_TALL_IMAGES=True in config for full analysis (9-10x cost)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
