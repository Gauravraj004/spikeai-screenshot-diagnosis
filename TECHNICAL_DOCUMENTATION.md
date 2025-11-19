# SpikeAI Screenshot Diagnosis Pipeline – Technical Documentation

## 1. High-Level Purpose
SpikeAI performs automated quality assurance of webpage captures (HTML + screenshot pairs). It integrates:
- Computer Vision (CV) heuristics for fast visual anomaly detection.
- Regional CV analysis to localize rendering/component failures.
- OCR-based mismatch detection for dynamic overlays / error pages (Stage 1.75).
- LLM semantic HTML understanding to interpret context and classify blocking issues.
- Multi‑stage integration logic producing a final status: correct | broken | uncertain plus confidence, diagnosis, fix.

The system is optimized for reliability on CPU-only machines (no GPU dependencies required) and resilience when LLM or OCR subsystems are unavailable (graceful degradation).

---
## 2. Architecture Overview

```
                +------------------------------+
                |        Input Artifacts       |
                |  data/screenshots/*.png      |
                |  data/html/*.html            |
                +---------------+--------------+
                                |
                                v
 +-------------------------------+----------------------------------+
 |                  Orchestration (main.py)                        |
 |  - Loads cases (stem match .png/.html)                          |
 |  - Sequential or batched parallel execution (ThreadPool)       |
 |  - Calls diagnose_screenshot(case)                             |
 +-------------------------------+----------------------------------+
                                |
                                v
                 +-------------------------------+
                 | diagnose_screenshot()         |
                 | 1. Preprocess (resize)        |
                 | 2. Stage 1 Global CV          |
                 | 3. Stage 1.5 Regional CV      |
                 | 4. Stage 1.75 OCR Mismatch    |
                 | 5. Stage 2 LLM Semantic       |
                 | 6. Integration Logic          |
                 +-------------------------------+
                                |
                                v
        +------------------- Aggregated Result --------------------+
        | {                                                     }  |
        |  case_name                                             | |
        |  status (correct/broken/uncertain)                     | |
        |  diagnosis (human-readable)                            | |
        |  confidence (0..1)                                     | |
        |  suggested_fix (actionable tool change)                | |
        |  stage_results { stage1_global, stage1_5_regional,     | |
        |                  ocr_mismatch, llm_analysis }          | |
        |  evidence { cv_global_findings[], cv_regional_findings[],| |
        |            ocr_mismatch_findings[], llm_findings[] }    | |
        |  evidence_summary                                      | |
        |  token_usage { total_tokens, breakdown, cost }         | |
        +---------------------------------------------------------+
                                |
                                v
     +--------------------+     +-----------------------+
     | JSON per case      |     | Excel / CSV Reports   |
     | results/<case>.json|     | diagnosis_report.*    |
     +--------------------+     +-----------------------+
```

---
## 3. Module-Level Responsibilities

| Module | Responsibility | Key Functions |
|--------|----------------|---------------|
| `main.py` | Orchestration, concurrency, integration, reporting | `diagnose_screenshot`, `get_case_files`, `calculate_token_cost`, `main` |
| `stage1_global.py` | Fast global heuristics for blank, overlays, duplication, entropy | `stage1_global_checks`, `is_blank_or_uniform`, `detect_blocking_overlay`, `detect_vertical_duplication`, `check_color_entropy` |
| `stage1_5_regional.py` | Grid-based localized anomaly detection | `stage1_5_regional_checks`, `divide_into_regions`, `check_region_quality` |
| `ocr_analysis.py` | OCR extraction + HTML mismatch detection | `extract_visible_text`, `detect_html_mismatch`, `get_ocr_diagnosis` |
| `llm_analyzer.py` | Semantic reasoning via Groq LLM; fallback logic | `diagnose_with_llm`, `analyze_html_with_llm`, `call_groq_llm` |
| `src/excel_export.py` | Report generation with formatting | `export_to_excel`, `create_excel_report`, `create_csv_report` |

---
## 4. Detailed Stage Logic

### 4.1 Stage 1 – Global CV (`stage1_global_checks`)
Focus: Fast elimination of obvious failures.
Heuristics:
- Blank page: Histogram dominance >95%, <10 unique colors, edge ratio <1%.
- Overlay/modal: Pattern detection in top-right (cookie) or center (auth/security) via brightness & contour analysis.
- Vertical duplication (2x or 3x): Structural similarity (SSIM) between halves or thirds. Adaptive thresholds for tall stitched images.
- Visual health: Color entropy + horizontal text edge density. Low entropy + no text edges ⇒ missing content.
Output Fields:
```
{
  status: PASS | BROKEN | ERROR,
  diagnosis: str,
  confidence: float,
  checks: { blank, overlay:{...}, duplicate:{...}, visual_health:{...} },
  evidence: str,
  suggested_fix: str
}
```
Confidence calibration: Discrete values (e.g., blank=0.95, overlay from detection confidence, duplication=SSIM value).

### 4.2 Stage 1.5 – Regional CV (`stage1_5_regional_checks`)
Grid: 4 columns × 3 rows = 12 regions (R{row}C{col}).
Per-region metrics:
- Entropy
- Std deviation
- Edge density (Canny)
- Unique colors
- Mean brightness
Checks classify region issues: `blank_region`, `uniform_error_region`, `low_color_diversity`, `gray_placeholder_region`.
False Positive Mitigation:
- Legitimate minimalist design detection (footers, side margins, hero sections).
- Ignore ≤2 low-confidence regions.
- Center content heuristic: If edges blank but center valid, treat as intentional layout unless overall content sparse.
Severity tiers: MINOR / MAJOR / CRITICAL via proportion affected.

### 4.3 Stage 1.75 – OCR Mismatch (`detect_html_mismatch`)
Goal: Detect dynamic error or blocking overlays not present in static HTML.
Process:
1. EasyOCR reader (singleton via LRU cache) → extracts candidate lines (confidence >0.3).
2. Filter garbage via `is_likely_ocr_garbage` (length, character mix, no vowels, etc.).
3. Error keyword scanning (HTTP codes, security challenge phrases, connection failures).
4. For each candidate line: fuzzy presence check in HTML via `find_text_in_html` (exact or ratio/word overlap).
5. If visible error text absent from HTML ⇒ mismatch.
Confidence: Base 0.85 + 0.05 per mismatched line (capped 0.95); boosted in integration if CV also found issues.
Issue Types: `security_challenge`, `network_error`, `http_error`, `access_denied`, `generic_error`.
Skip Conditions: If Stage 1 already BROKEN with confidence > 0.90 (performance trade-off) – potential improvement area.

### 4.4 Stage 2 – LLM Semantic (`diagnose_with_llm`)
Prompt Engineering:
- Inject CV findings and OCR context (including mismatched lines for high-priority errors).
- Provide detection rules with priority ordering and JSON-only output schema.
- Extract and parse JSON (resilient to markdown fences) with fallback heuristics when parsing fails.
LLM Output Canonicalized to:
```
{
  status: CORRECT | BROKEN | ERROR,
  diagnosis: str,
  confidence: float,
  suggested_fix: str,
  issue_type: str,
  evidence: str,
  token_usage: {prompt_tokens, completion_tokens, total_tokens}
}
```
Retry Logic: Exponential backoff (2^attempt) up to 3 attempts. Graceful downgrade: If LLM unreachable and CV passes → treat case as CORRECT with reduced confidence.

### 4.5 Integration Logic (in `diagnose_screenshot`)
Priority Order:
1. OCR mismatch (definitive proof) ⇒ BROKEN (confidence may be boosted if CV also broken).
2. CV (global/regional) + LLM both broken ⇒ aggregated high-confidence BROKEN.
3. CV broken alone ⇒ BROKEN (semantic confirmation absent).
4. LLM broken alone ⇒ BROKEN (semantic issue, visual metrics passed).
5. All pass ⇒ high-confidence CORRECT.
6. Mixed/ambiguous ⇒ UNCERTAIN (manual review recommended).
Result Enrichment:
- User-friendly explanation via `generate_user_friendly_explanation`.
- Suggested automated tool remediation via `generate_suggested_fix`.
- Token accounting + theoretical cost (`calculate_token_cost`).

Confidence Composition Examples:
- All three broken: avg(stage confidences) + 0.15 (cap 0.97).
- All three pass: avg + 0.08 (cap 0.96).
- Single-source evidence: direct stage confidence or adjusted heuristic.

---
## 5. Data Structures & Key Objects

### 5.1 Case Definition
```
{ case_name: str, screenshot_path: str, html_path: str }
```
Generated by matching filename stems; no formal manifest required.

### 5.2 Aggregated Result Object (Final)
```
{
  case_name: str,
  status: 'correct' | 'broken' | 'uncertain',
  diagnosis: str,
  confidence: float,
  suggested_fix: str,
  root_cause: str,
  evidence_summary: str,
  stage_results: {
    stage1_global: {...},
    stage1_5_regional: {...},
    ocr_mismatch: {...},
    llm_analysis: {...}
  },
  evidence: {
    cv_global_findings: [str],
    cv_regional_findings: [str],
    ocr_mismatch_findings: [str],
    llm_findings: [str]
  },
  token_usage: {
    total_tokens: int,
    breakdown: { llm: { prompt_tokens, completion_tokens, total_tokens } },
    estimated_cost_usd: float
  }
}
```

### 5.3 OCR Result
```
{
  status: 'MISMATCH_DETECTED' | 'NO_MISMATCH' | 'SKIPPED',
  has_mismatch: bool,
  visible_error_text: [str],
  detailed_mismatches: [ { text, matching_keywords, similarity_to_html } ],
  issue_type: str,
  ocr_extracted: str,
  confidence: float,
  error?: str
}
```

### 5.4 Token Pricing Logic
```
calculate_token_cost(tokens): cost_per_million = 0.69
return (tokens / 1_000_000) * 0.69
```
Used even on free-tier to surface hypothetical cost awareness.

---
## 6. Concurrency & Performance Model

```
for batches of cases (size <= 15):
  ThreadPoolExecutor(max_workers=2)
    submit diagnose_screenshot(case)
      -> heavy OCR (first run ~80MB model download)
      -> LLM call (network latency)
Save checkpoint after each case to allow recovery.
GC invoked after each future completion to mitigate memory spikes.
```
Rationale:
- Limiting workers to 2 prevents excessive memory usage (OCR model + arrays).
- Batch size avoids queueing too many memory-heavy tasks.
- Checkpointing allows long-running sessions to resume if interrupted.

---
## 7. Assumptions
- Screenshot/HTML filenames share stem (e.g., `example.png` + `example.html`).
- Screenshots are upright, not rotated; mostly desktop viewport captures.
- English text predominantly (OCR initialized with `['en']`).
- HTML contains representative static content minus dynamic overlays (overlays often injected post-load and absent from saved HTML). This assumption enables mismatch detection.
- LLM key available (Groq) OR system degrades gracefully.
- CPU-only environment (PyTorch CPU wheels enforced in requirements).
- Maximum screenshot width resizing at 1920 maintains diagnostic integrity while reducing memory footprint.

---
## 8. Technical Specifications & Trade-offs

| Aspect | Choice | Rationale | Trade-off |
|--------|--------|-----------|-----------|
| OCR Engine | EasyOCR (CPU) | Reliable downloads, simple API | Slower than GPU PaddleOCR on large batches |
| Duplication Metric | SSIM | Robust to minor pixel shifts | Can yield false positives on similar layout repeated intentionally |
| Region Grid | 4x3 (12 regions) | Balance granularity vs overhead | Very small components may be missed |
| Confidence Strategy | Heuristic aggregation | Transparent, tunable | Not probabilistically calibrated |
| LLM Model | `llama-3.3-70b-versatile` | Strong reasoning, relatively low cost | Token latency vs smaller models |
| Error Mismatch Detection | OCR vs HTML diff | Captures dynamic overlays | Fails if overlay text also present in HTML comments |
| Parallelism | 2 threads | Memory safety during OCR | Underutilizes multi-core systems |
| Token Cost Constant | 0.69 / 1M tokens | Simplified blended rate | Not exact per input/output class |

Latency vs Accuracy:
- Early CV stages reject invalid captures quickly (no LLM cost).
- OCR is skipped when Stage 1 is high-confidence broken (performance optimization risking missed layered issues).

---
## 9. Edge Cases & Limitations

| Edge Case | Current Handling | Limitation |
|-----------|------------------|------------|
| Rotated screenshots | Not corrected | OCR & region checks degrade |
| Non-English pages | OCR trained only on English | Missed overlay text |
| Extremely tall pages (> 20k px) | Lower 3x SSIM threshold | Might misclassify long scroll captures |
| Animated or transient overlays | Single-frame capture | Temporal issues ignored |
| Light-on-light text (low contrast) | CLAHE preprocessing | May still fail on subtle gradients |
| SVG text / Canvas rendering | OCR cannot read vector shapes reliably | Missed error banners drawn in canvas |
| Multi-language mixed content | Keyword list English-only | False negatives for localized error pages |
| Script-injected dynamic HTML saved to file | Mismatch logic fails (text appears in HTML) | Overlay undetected |
| High noise backgrounds | Garbage text filtering heuristic | Risk of filtering meaningful small print |

Potential Improvements:
- Multilingual OCR (`['en','es','fr']` etc.).
- Rotation/deskew preprocessing.
- GPU acceleration optional path.
- Model-based overlay classification vs heuristics.
- Adaptive confidence calibration via validation dataset.

---
## 10. Known Gaps / Clarifying Questions
1. Should OCR ever be forced even when Stage 1 reports high-confidence BROKEN (e.g., to differentiate error type)?
2. Are filenames guaranteed unique and stable, or should manifest-based mapping be introduced?
3. Do we require multi-language support for production environments?
4. Should duplication detection consider horizontal repetition (currently only vertical)?
5. Is there a need to persist raw intermediate metrics for external analytics?
6. Token pricing – should input/output costs be tracked separately instead of blended?
7. Should security challenge classification trigger an automatic retry with alternate network conditions?

---
## 11. Suggested Tool / Capture Remediations (Generated Fix Categories)
| Issue Type | Suggested Fix (Examples) |
|------------|--------------------------|
| Cookie modal | Auto-dismiss via DOM query + delay before screenshot |
| Blank page | Increase timeout, wait for network idle (no requests 2s) |
| Security challenge | Introduce human verification step / proxy rotation |
| Duplication | Adjust scroll stitching overlap detection & DOM uniqueness checks |
| Access denied | Pre-authenticate session / inject cookies before navigation |
| Low entropy loading screen | Wait for disappearance of spinners / skeleton elements |
| Regional blank components | Trigger lazy load (scroll) / extend hydration wait |
| OCR mismatch generic error | Capture console logs / network waterfall for evidence |

---
## 12. Step-by-Step Code Walkthrough (Critical Paths)

### 12.1 `main.py::main()`
1. Prints pipeline summary.
2. Ensures `results/` exists.
3. Collects cases via `get_case_files` (HTML + PNG).
4. Loads checkpoint (if present) to resume partial runs.
5. Determines parallelism (max 2 workers) and batches (size 15).
6. For each case → calls `diagnose_screenshot(case)`.
7. Writes per-case JSON, updates checkpoint, collects tokens/cost.
8. After completion: builds Excel + CSV reports.
9. Prints summary table and performance/cost metrics.

### 12.2 `diagnose_screenshot(case)` (Central Integration)
1. Resize screenshot if width > 1920.
2. Run Stage 1 Global → early return pattern (if BROKEN) captured in result object.
3. Run Stage 1.5 Regional → collects localized findings; filters low-confidence.
4. Decide skip vs run OCR (Stage 1.75) based on Stage 1 BROKEN confidence.
5. Run OCR mismatch detection → classify error type if mismatch.
6. Run LLM semantic analysis (injecting CV + OCR context).
7. Integrate four signals using priority cascade.
8. Compute token cost → append to result.
9. Print token summary (optional).

### 12.3 Global CV Checks (`stage1_global_checks`)
- Sequential evaluation: blank → overlay → duplication → entropy.
- Return immediately on first definitive issue to minimize overhead.

### 12.4 Regional CV (`stage1_5_regional_checks`)
- Divide image into 12 regions.
- Evaluate each with metrics.
- Apply legitimacy heuristics (hero sections, footers, margins).
- Aggregate severity & confidence; skip noise (≤2 low-confidence).

### 12.5 OCR (`detect_html_mismatch`)
- Extract line candidates.
- Keyword-driven error detection.
- Fuzzy/word overlap HTML presence check.
- Build mismatch list & compute confidence.

### 12.6 LLM (`diagnose_with_llm` + `analyze_html_with_llm`)
- Compose structured prompt with explicit JSON schema.
- Pass enhanced CV + OCR context.
- Retry logic around API call.
- Parse JSON (strip code fences).
- Fallback heuristics if parsing fails.

### 12.7 Reporting (`excel_export.py`)
- Format conditional styling for CORRECT/BROKEN.
- Text wrapping, width constraints.
- Evidence synthesis (global + regional + LLM).
- CSV fallback.

---
## 13. Error Handling & Resilience
| Layer | Strategy |
|-------|----------|
| Image Load | Return ERROR status early if cv2.imread fails |
| OCR Init Failure | Log, degrade gracefully; pipeline continues without OCR |
| LLM API Failure | Retry 3× with exponential backoff; fallback to CORRECT if CV passes |
| JSON Parse Failure | Attempt fallback; default safe assumption when CV passed |
| Memory Pressure | Batch + GC between futures |
| Checkpoint Recovery | Partial results saved after each case; deleted on success |

---
## 14. Performance Characteristics
Approximate per-case (CPU-only, first OCR run downloads ~80MB models):
- Stage 1: <150 ms
- Stage 1.5: 200–400 ms (depends on image size)
- OCR: 0.8–1.5 s (first case longer due to model download)
- LLM: 0.5–1.2 s (network latency + generation)
- Integration: negligible (<50 ms)
Parallelism (2 workers) yields overlap but not full linear scaling due to IO & model contention.

---
## 15. Future Enhancements Roadmap
1. Replace heuristic overlay detection with CNN-based classifier (modal vs content).
2. Introduce adaptive OCR retry if initial extraction yields < N lines.
3. Add multilingual OCR support.
4. Persist intermediate metrics for ML calibration (confidence learning).
5. Horizontal duplication detection (side-by-side clones).
6. Visual diffing against baseline screenshot library.
7. Structured logging (JSON Lines) for pipeline observability.
8. Automatic model warm-up phase to reduce first-run latency.

---
## 16. Final Summary
SpikeAI integrates layered perception (global heuristics, localized diagnostics, OCR semantic mismatch) with higher-level reasoning (LLM). Priority ordering yields deterministic resolution while remaining extensible. Resilience features (checkpointing, graceful degradation) support production robustness. The documentation above clarifies logic, trade-offs, and evolution paths.

---
## 17. Open Questions For Meeting
- Confirm required internationalization scope for OCR/LLM.
- Define SLA for processing time per batch.
- Decide threshold tuning methodology (static vs feedback loop).
- Determine how to expose granular metrics (API vs reports).
- Clarify risk tolerance for false negatives vs false positives.

---
END OF DOCUMENT
