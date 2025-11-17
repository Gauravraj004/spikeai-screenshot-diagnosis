"""
LLM-based HTML and Screenshot Analysis
Uses free Groq API for intelligent semantic understanding
"""

import os
import json
import base64
import time
from typing import Dict, Any, Callable
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

# Load environment
load_dotenv()


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0):
    """
    Decorator for retry logic with exponential backoff
    Use for LLM API calls that may fail due to rate limits or timeouts
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        print(f"  ⚠ API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ✗ API call failed after {max_retries} attempts")
            # If all retries failed, return error dict instead of raising
            return {"error": f"Failed after {max_retries} retries: {str(last_exception)}", "issue_detected": False}
        return wrapper
    return decorator


def extract_html_summary(html_content: str, max_length: int = 3000) -> str:
    """
    Extract a compact summary of HTML for LLM analysis
    Focus on: structure, components, text content (not full HTML)
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove script/style tags
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    
    # Get visible text
    visible_text = soup.get_text(separator=' ', strip=True)
    visible_text = ' '.join(visible_text.split())[:1000]  # First 1000 chars
    
    # Count key elements
    structure = {
        "sections": len(soup.find_all(['section', 'article', 'div'], class_=True)),
        "headings": len(soup.find_all(['h1', 'h2', 'h3'])),
        "images": len(soup.find_all('img')),
        "links": len(soup.find_all('a')),
        "buttons": len(soup.find_all(['button', 'input'])),
        "forms": len(soup.find_all('form'))
    }
    
    # Get class names (indicators of components)
    all_classes = []
    for tag in soup.find_all(class_=True):
        all_classes.extend(tag.get('class', []))
    
    # Get most common classes (web components)
    from collections import Counter
    common_classes = Counter(all_classes).most_common(10)
    
    # Build summary
    summary = f"""HTML Structure Summary:
- Visible text preview: {visible_text[:500]}...
- Element counts: {structure}
- Common component classes: {[cls for cls, count in common_classes]}
- Total unique classes: {len(set(all_classes))}
"""
    
    return summary


@retry_on_failure(max_retries=3, backoff_factor=2.0)
def call_groq_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    """
    Call Groq LLM API (free tier) - using llama-3.3-70b-versatile
    With retry logic and exponential backoff for rate limits
    
    Returns:
        Dict with 'content' (str) and 'token_usage' (dict)
    """
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise Exception("Error: No GROQ_API_KEY configured in .env file")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()
        
        # Extract token usage
        usage = result.get('usage', {})
        token_data = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }
        
        return {
            'content': result['choices'][0]['message']['content'],
            'token_usage': token_data
        }
    except requests.exceptions.HTTPError as e:
        # Get error details from response
        try:
            error_details = e.response.json()
            error_msg = error_details.get('error', {}).get('message', str(e))
            # Re-raise for retry logic to handle
            raise Exception(f"API Error: {error_msg}")
        except:
            raise Exception(f"API Error: {str(e)}")
    except Exception as e:
        raise  # Re-raise for retry decorator


def analyze_html_with_llm(html_content: str, page_url: str, cv_finding: str = None, ocr_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    UNIVERSAL LLM-based HTML analysis - works for ANY website
    Analyzes HTML semantically to find issues like:
    - Cookie/consent modals blocking content
    - Security challenges (Cloudflare, bot detection)
    - Login/auth walls
    - Missing CSS/JS resources
    - Anti-bot scripts
    - Overlay/popup blockers
    
    Args:
        html_content: Full HTML source
        page_url: URL of the page
        cv_finding: Optional CV detection result (e.g., "low entropy", "duplicate content")
        ocr_result: Optional OCR mismatch detection result
    """
    
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Extract key indicators for LLM
    # 1. Visible text (first 1000 chars)
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    visible_text = soup.get_text(separator=' ', strip=True)
    visible_text_preview = ' '.join(visible_text.split())[:1000]
    
    # 2. External scripts (blocking/security indicators)
    scripts = []
    for script in soup.find_all('script', src=True):
        src = script.get('src', '')
        if src and not src.startswith('data:'):
            scripts.append(src)
    
    # 3. Look for security/blocking keywords
    security_keywords = []
    html_lower = html_content.lower()
    keywords = ['cookie', 'consent', 'gdpr', 'cloudflare', 'captcha', 'challenge', 
                'security', 'bot', 'verify', 'human', 'login', 'sign in', 'authentication']
    for kw in keywords:
        if kw in html_lower:
            security_keywords.append(kw)
    
    # 4. CSS resources
    css_links = []
    for link in soup.find_all('link', rel='stylesheet'):
        href = link.get('href', '')
        if href:
            css_links.append(href)
    
    # Build OCR context if available
    ocr_context = ""
    if ocr_result and ocr_result.get('has_mismatch'):
        ocr_context = f"""
⚠️ OCR MISMATCH DETECTED (HIGH PRIORITY):
The following text is VISIBLE in the screenshot but NOT FOUND in the HTML:
{json.dumps(ocr_result.get('visible_error_text', []), indent=2)}

This indicates dynamically rendered error/overlay content (JavaScript, iframe, or security challenge).
Issue Type: {ocr_result.get('issue_type', 'unknown')}
Confidence: {ocr_result.get('confidence', 0.0):.2f}
"""
    elif ocr_result and not ocr_result.get('has_mismatch'):
        ocr_context = "\n✓ OCR Check: All visible text matches HTML content (no dynamic errors detected)\n"
    
    # Build context for LLM
    context = f"""Webpage Diagnosis - Universal Analysis

URL: {page_url}

CV FINDING: {cv_finding if cv_finding else "No CV issues detected - page loaded successfully"}
{ocr_context}
HTML Summary:
- Visible text preview: "{visible_text_preview[:300]}..."
- External scripts: {len(scripts)} total
- CSS stylesheets: {len(css_links)} total
- Security/blocking keywords found: {security_keywords}
- Page title: {soup.title.string if soup.title else "None"}

Key scripts (check for actual BLOCKERS, not just cookie scripts):
{json.dumps(scripts[:10], indent=2)}

IMPORTANT CONTEXT:
- The screenshot tool captures pages AFTER they are fully loaded (not during load)
- Cookie consent scripts (cookiehub, onetrust) are NORMAL and don't block unless they show a MODAL
- Only detect issues if there's EVIDENCE of actual blocking (security challenge, auth wall, missing resources)

Your task: Determine if the CV finding indicates a BLOCKING ISSUE in the HTML.

CV Finding Analysis:
- If "duplicate/repeated": Check HTML for actual duplication (repeated IDs, infinite scroll bugs) vs screenshot stitching error
- If "blank/low entropy": Check for missing CSS, security challenges (Cloudflare/captcha text), login walls, anti-bot scripts
- If "cookie modal detected": Check if cookie scripts are actually BLOCKING or just present
- If "no CV issues": Check HTML for potential hidden issues (missing resources, JS errors)

DETECTION RULES (priority order):

**HIGHEST PRIORITY - OCR MISMATCH:**
If OCR detected error text visible but not in HTML, this is DEFINITIVE PROOF of a blocking issue.
The error was dynamically rendered (JavaScript, iframe, security overlay).

1. SECURITY/ERROR PAGE (HIGH PRIORITY): 
   - OCR found: "verify", "security", "connection", "couldn't verify", "access denied", "captcha", "challenge"
   - These are ERROR PAGES blocking the actual website
   - Issue type: "security_block" or "security_challenge"
   
2. DUPLICATE CONTENT: Only if CV found "repeated/duplicate" AND HTML has legitimate duplication (not stitching bug)

3. AUTH WALL: OCR or HTML contains "login", "sign in", "authentication required", "restricted" as MAIN content

4. MISSING RESOURCES: CV found "blank" AND CSS links are broken/missing AND visible text is minimal

5. COOKIE MODAL BLOCKING: CV explicitly detected "cookie modal blocking" AND cookiehub/onetrust scripts present

CRITICAL: If OCR mismatch exists, prioritize it over other findings!

DO NOT report cookie scripts as blocking UNLESS CV detected a modal blocking content!

RESPOND WITH JSON ONLY (no markdown):
{{"issue_detected":true,"issue_type":"security_block","confidence":0.95,"diagnosis":"Security verification page blocking access to actual content","evidence":"visible text: 'We couldn't verify the security of your connection'","tool_fix":"TOOL: Add security bypass, use different IP/browser fingerprint"}}

If no blocking issue found:
{{"issue_detected":false,"diagnosis":"Page structure appears normal","confidence":0.95,"tool_fix":"No action needed"}}"""

    llm_response = call_groq_llm(context)
    
    # Check if retry decorator returned error dict
    if isinstance(llm_response, dict) and 'error' in llm_response:
        return llm_response  # Already has error and issue_detected: False
    
    # Extract content and token usage
    response = llm_response.get('content', '') if isinstance(llm_response, dict) else llm_response
    token_usage = llm_response.get('token_usage', {}) if isinstance(llm_response, dict) else {}
    
    # Check for API errors in string response
    if isinstance(response, str) and (response.startswith("Error:") or response.startswith("API Error:")):
        return {"error": response, "issue_detected": False}
    
    try:
        # Remove markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # Extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Universal response format with token tracking
            return {
                "issue_detected": parsed.get("issue_detected", False),
                "issue_type": parsed.get("issue_type", "unknown"),
                "confidence": parsed.get("confidence", 0.5),
                "diagnosis": parsed.get("diagnosis", "No issues detected"),
                "evidence": parsed.get("evidence", ""),
                "tool_fix": parsed.get("tool_fix", "No action needed"),
                "token_usage": token_usage
            }
        else:
            # If CV didn't find issues and LLM can't parse, assume page is correct
            if cv_finding and "Passed global health checks" in cv_finding:
                return {
                    "issue_detected": False,
                    "diagnosis": "Page structure appears normal",
                    "confidence": 0.85,
                    "evidence": "",
                    "tool_fix": "No action needed",
                    "token_usage": token_usage
                }
            return {"error": f"No JSON found in LLM response. Raw response: {response[:200]}", "issue_detected": False}
    except json.JSONDecodeError as e:
        # If JSON parsing fails but CV passed, treat as correct
        if cv_finding and "Passed global health checks" in cv_finding:
            return {
                "issue_detected": False,
                "diagnosis": "Page structure appears normal",
                "confidence": 0.85,
                "evidence": "",
                "tool_fix": "No action needed",
                "token_usage": token_usage
            }
        return {"error": f"Failed to parse JSON: {str(e)}. Response: {response[:200]}", "issue_detected": False}
    except Exception as e:
        return {"error": f"Failed to parse LLM response: {str(e)}", "issue_detected": False}


def analyze_screenshot_with_llm(screenshot_path: str, expected_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use VLM to verify if screenshot matches expected components
    Uses Google Gemini vision (Groq vision models deprecated)
    """
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return {"status": "ERROR", "diagnosis": "No GOOGLE_API_KEY configured"}
    
    # Load and encode image
    from PIL import Image
    import io
    
    img_pil = Image.open(screenshot_path)
    w, h = img_pil.size
    
    # Resize if too large
    if w > 1536:
        ratio = 1536 / w
        new_w = 1536
        new_h = int(h * ratio)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Encode to base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=85)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Build prompt with expected components
    expected_components = expected_analysis.get('expected_components', [])
    repeated_components = expected_analysis.get('repeated_components', {})
    potential_blockers = expected_analysis.get('potential_blockers', [])
    
    prompt = f"""Analyze this webpage screenshot and verify if it's properly captured.

EXPECTED COMPONENTS (should be visible):
{', '.join(expected_components)}

REPEATED COMPONENTS (check if count seems reasonable):
{json.dumps(repeated_components, indent=2)}

POTENTIAL ISSUES TO CHECK:
{', '.join(potential_blockers)}

Questions:
1. Are all expected components visible and properly rendered?
2. Are there any BLOCKING OVERLAYS (cookie banners, modals, auth walls)?
3. Does the page appear DUPLICATED/STITCHED incorrectly?
4. Are there MISSING COMPONENTS or broken sections?
5. Is the page BLANK or partially loaded?

Respond ONLY with valid JSON (no markdown):
{{"status":"CORRECT","missing_components":[],"blocking_elements":[],"duplication_detected":false,"confidence":0.9,"diagnosis":"one sentence"}}"""

    # Use Google Gemini Vision API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Remove markdown if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Parse JSON response
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Ensure required fields
            return {
                "status": parsed.get("status", "UNKNOWN"),
                "missing_components": parsed.get("missing_components", []),
                "blocking_elements": parsed.get("blocking_elements", []),
                "duplication_detected": parsed.get("duplication_detected", False),
                "confidence": parsed.get("confidence", 0.5),
                "diagnosis": parsed.get("diagnosis", "Unknown issue")
            }
        else:
            return {"status": "ERROR", "diagnosis": "Failed to parse VLM response"}
        
    except requests.exceptions.HTTPError as e:
        # Get error details
        try:
            error_details = e.response.json()
            error_msg = error_details.get('error', {}).get('message', str(e))
            return {"status": "ERROR", "diagnosis": f"VLM API error: {error_msg}"}
        except:
            return {"status": "ERROR", "diagnosis": f"VLM API error: {str(e)}"}
    except Exception as e:
        return {"status": "ERROR", "diagnosis": f"VLM error: {str(e)}"}


def diagnose_with_llm(html_path: str, screenshot_path: str, page_url: str, cv_finding: str = None, ocr_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    UNIVERSAL LLM-based diagnosis - works for ANY website
    
    If CV Stage 1 found an issue (blank, duplicate), pass it here for semantic analysis.
    Otherwise, analyze HTML to detect blocking issues (modals, security challenges, etc.)
    
    Args:
        html_path: Path to HTML file
        screenshot_path: Path to screenshot
        page_url: URL of the page
        cv_finding: Optional CV detection result (e.g., "low entropy", "duplicate content")
        ocr_result: Optional OCR analysis result with visible error text
    
    Returns:
        Dict[str, Any]: Universal diagnosis result with keys:
            - status (str): 'CORRECT', 'BROKEN', or 'ERROR'
            - diagnosis (str): Detailed issue description
            - confidence (float): Detection confidence 0.0-1.0
            - suggested_fix (str): Recommended action to resolve issue
            - evidence (str): Supporting evidence for diagnosis
    """
    
    # Read HTML
    if not os.path.exists(html_path):
        return {
            "status": "ERROR",
            "diagnosis": "HTML file not found",
            "confidence": 0.0,
            "suggested_fix": "Check file path"
        }
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Use LLM to analyze HTML semantically (with ENHANCED OCR context)
    print("  → LLM: Semantic HTML analysis for blocking issues...")
    
    # Enhance CV finding with OCR results for better LLM context
    enhanced_cv_finding = cv_finding or ""
    if ocr_result and ocr_result.get('has_mismatch'):
        ocr_context = f" | OCR CRITICAL: Detected visible error text not in HTML: {', '.join(ocr_result.get('visible_error_text', [])[:3])}"
        enhanced_cv_finding = (enhanced_cv_finding + ocr_context).strip()
    
    html_analysis = analyze_html_with_llm(html_content, page_url, enhanced_cv_finding, ocr_result)
    
    if "error" in html_analysis:
        # If no API key configured and CV passed, assume page is correct
        if "No GROQ_API_KEY" in html_analysis['error'] and cv_finding and "Passed global health checks" in cv_finding:
            return {
                "status": "CORRECT",
                "diagnosis": "Page appears correct (CV checks passed, LLM unavailable)",
                "confidence": 0.80,
                "suggested_fix": "No action needed",
                "evidence": "CV Analysis: All visual metrics passed"
            }
        return {
            "status": "ERROR",
            "diagnosis": f"LLM analysis failed: {html_analysis['error']}",
            "confidence": 0.0,
            "suggested_fix": "Retry analysis"
        }
    
    # Check if LLM detected an issue
    if html_analysis.get("issue_detected"):
        issue_type = html_analysis.get("issue_type", "unknown")
        diagnosis = html_analysis.get("diagnosis", "Issue detected")
        tool_fix = html_analysis.get("tool_fix", "Check page rendering")
        
        print(f"  → Issue detected: {issue_type}")
        print(f"  → Diagnosis: {diagnosis}")
        
        return {
            "status": "BROKEN",
            "diagnosis": diagnosis,
            "confidence": html_analysis.get("confidence", 0.8),
            "suggested_fix": tool_fix,
            "issue_type": issue_type,
            "evidence": html_analysis.get("evidence", ""),
            "token_usage": html_analysis.get("token_usage", {})
        }
    
    # No issue detected - page appears correct
    print("  → No blocking issues detected in HTML")
    return {
        "status": "CORRECT",
        "diagnosis": html_analysis.get("diagnosis", "Page structure appears normal"),
        "confidence": html_analysis.get("confidence", 0.90),
        "suggested_fix": "No action needed",
        "token_usage": html_analysis.get("token_usage", {})
    }


def generate_fix_suggestion(vlm_result: Dict, html_analysis: Dict) -> str:
    """Generate actionable fix based on diagnosis"""
    
    if vlm_result.get("blocking_elements"):
        return "TOOL: Detect and dismiss overlays/modals before capture"
    
    if vlm_result.get("duplication_detected"):
        return "TOOL: Fix stitching algorithm - validate no duplicate sections"
    
    if vlm_result.get("missing_components"):
        missing = vlm_result["missing_components"]
        if any("lazy" in str(html_analysis.get("potential_blockers", [])).lower()):
            return "TOOL: Wait for lazy-loaded content, scroll to trigger loads"
        return f"TOOL: Increase wait time, ensure {missing[0]} loads"
    
    return "TOOL: Check page rendering, validate complete load"
