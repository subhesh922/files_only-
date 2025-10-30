# server/agents/context_agent.py
import asyncio
import json
import logging
import re
from typing import List, Dict, Optional
import difflib
from difflib import SequenceMatcher
from collections import Counter

from server.utils.azure_openai_client import get_chat_client, get_chat_deployment , get_embedding_deployment, get_embedding_client
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from server.agents.vectorstore_agent import VectorStoreAgent

logger = logging.getLogger(__name__)


def normalize_column(col: str, schema: List[str]) -> str:
    """Map user/LLM column names to closest schema column."""
    col = col.strip()
    if col in schema:
        return col
    match = difflib.get_close_matches(col, schema, n=1, cutoff=0.6)
    return match[0] if match else col


def apply_json_filters(rows: List[Dict], rules: List[Dict], schema: List[str]) -> List[Dict]:
    filtered = []
    for row in rows:
        keep = True
        for rule in rules:
            col = normalize_column(rule["column"], schema)
            op = rule["op"]
            val = rule["value"]

            cell_val = row.get(col, "")

            # numeric ops
            if isinstance(cell_val, (int, float)) and isinstance(val, (int, float)):
                if op == "=" and not (cell_val == val):
                    keep = False
                if op == "!=" and not (cell_val != val):
                    keep = False
                if op == ">" and not (cell_val > val):
                    keep = False
                if op == "<" and not (cell_val < val):
                    keep = False
                if op == ">=" and not (cell_val >= val):
                    keep = False
                if op == "<=" and not (cell_val <= val):
                    keep = False

            # text ops (case-insensitive)
            else:
                cell_val_str = str(cell_val).lower()
                val_str = str(val).lower()
                if op == "=" and val_str != cell_val_str:
                    keep = False
                if op == "contains" and val_str not in cell_val_str:
                    keep = False

        if keep:
            filtered.append(row)
    return filtered


def parse_focus_prompt_with_llm(prompt: str, llm, schema: List[str], deployment: str) -> List[Dict]:
    """
    Translate a natural language filter prompt into JSON rules using the LLM.
    """
    if not prompt:
        return []

    system_msg = f"""
You are a filter translator for DFMEA data.
Valid columns: {schema}

Convert the user request into JSON rules.
Each rule = {{ "column": <col>, "op": <op>, "value": <val> }}
- column: must be from {schema}
- op: one of '=', '!=', '>', '<', '>=', '<=', 'contains'
- value: number (for Severity, Occurrence, Detection, RPN) or string for text columns
Return ONLY a valid JSON list of rules, no explanation.
"""

    user_msg = f"User request: {prompt}"

    try:
        response = llm.chat.completions.create(
            model=deployment,  # use passed deployment string
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"❌ [FocusPrompt] Failed to parse prompt: {e}")
        return []


class ContextAgent:
    def __init__(self, llm_client=None, vector_agent=None, batch_size: int = 15):
        self.vector_agent = vector_agent or VectorStoreAgent()
        self.llm = llm_client or get_chat_client()
        self.deployment = get_chat_deployment()
        self.batch_size = batch_size

        self.schema = [
            "Subsystem", "Component", "Function", "Failure Mode", "Effect", "Cause",
            "Severity", "Occurrence", "Detection", "RPN",
            "Controls Prevention", "Controls Detection", "Recommended Actions",
            "FM Origin",
        ]

    async def run(
        self,
        campaign: str,                                   # ✅ NEW
        subsystems: List[str],
        admin_prompt: Optional[str] = None,
        top_k: int = 100,
        chunk_cap: int = 300,
        max_concurrent: int = 3,
    ) -> List[Dict]:
        logger.info(f"🚀 [ContextAgent] Running for subsystems={subsystems}, admin_prompt={admin_prompt}")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_subsystem(subsystem: str) -> List[Dict]:
            async with semaphore:
                # 1) get vector results (payload includes 'source' and 'text')
                # --- CHANGED: dual-query using subsystem words (adds PRD-friendly phrasing) ---
                q1 = f"{subsystem} dfmea failure mode effect cause control detection prevention"
                q2 = f"{subsystem} product requirements prd risk verification validation acceptance criteria"

                r1 = self.vector_agent.search(query=q1, subsystems=[subsystem], top_k=top_k, campaign=campaign)  # ✅ NEW
                r2 = self.vector_agent.search(query=q2, subsystems=[subsystem], top_k=top_k, campaign=campaign)  # ✅ NEW

                seen_keys, results = set(), []
                for hit in (r1 or []) + (r2 or []):
                    payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
                    # Dedup key: text + source + filename (stable across sources)
                    key = (
                        (payload.get("text") or ""),
                        (payload.get("source") or "").lower(),
                        (payload.get("filename") or ""),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    results.append(hit)

                if not results:
                    logger.warning(f"⚠️ [ContextAgent] No results for {subsystem}")
                    return []

                # (Optional tiny breadcrumb) show distribution before cap
                try:
                    from collections import Counter as _Ctr
                    _cnt = _Ctr([(hit.get("payload", {}) or {}).get("source", "unknown").lower() for hit in results])
                    logger.info(f"📊 [ContextAgent][STATS] subsystem={subsystem} source_counts(pre-cap)={dict(_cnt)}")
                except Exception:
                    pass
                # --- END CHANGE ---

                # 2) separate chunks by source and keep (source, text, payload) tuples
                kb_chunks, sprll_chunks, prd_chunks, rd_chunks = [], [], [], []
                hits_for_fallback = []  # keep original hits for fallback decisions
                for hit in results[:chunk_cap]:
                    payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
                    src = (payload.get("source") or "").lower()
                    text = payload.get("text", "") or ""
                    hits_for_fallback.append({"payload": payload, "text": text, "source": src})

                    # classify by source (feedback is guidance only: skip it as evidence)
                    if "kb" in src or "knowledge" in src:
                        kb_chunks.append((src, text, payload))
                    elif "sprll" in src:
                        sprll_chunks.append((src, text, payload))
                    elif "prd" in src or "requirement" in src:
                        prd_chunks.append((src, text, payload))
                    elif "feedback" in src:
                        # do not add to evidence buckets
                        pass
                    else:
                        rd_chunks.append((src, text, payload))  # repair_data/others

                # 3) build batches
                ordered_chunks = []
                for src, text, payload in kb_chunks + sprll_chunks + prd_chunks + rd_chunks:
                    if "kb" in src or "knowledge" in src:
                        label = "kb"
                    elif "sprll" in src:
                        label = "sprll"
                    elif "prd" in src or "requirement" in src:
                        label = "prds"
                    else:
                        label = "repair_data"

                    ordered_chunks.append(
                        {
                            "source_label": label,
                            "text": text,
                            "payload": payload,
                        }
                    )

                batches = [
                    ordered_chunks[i : i + self.batch_size]
                    for i in range(0, len(ordered_chunks), self.batch_size)
                ]

                # -----------------------
                # LOGGING: stats counters
                # -----------------------
                context_source_counts = Counter()   # chunks per source fed to LLM (all batches)
                batch_source_counts = []           # per-batch chunk mix
                row_support_rows = Counter()       # rows with ≥1 supporting chunk per source
                row_support_chunk_hits = Counter() # total matched-chunk hits by source
                rows_with_no_support = 0           # rows where FM matched no chunk text in batch

                def _support_sources_for_fm(fm_text: str, batch_chunks: List[Dict]) -> Counter:
                    """
                    Return per-source match counts for this Failure Mode using semantic matching.
                    Mirrors the logic used in _determine_fm_origin so logs reflect reality.
                    """
                    fm = (fm_text or "").casefold().strip()
                    hits = Counter()
                    if not fm:
                        return hits

                    # Adjust to your actual repair-data payload keys as needed
                    PAYLOAD_KEYS = [
                        "Fault_Code", "FaultType", "Fault_Type", "Issue", "Issue_Desc",
                        "Symptom", "Part_Category", "Component", "FailureMode", "Failure_Mode",
                        "filename",
                    ]

                    def compose_match_text(ch: Dict) -> str:
                        txt = (ch.get("text") or "")
                        p = (ch.get("payload") or {})
                        extras = " ".join(str(p.get(k, "")) for k in PAYLOAD_KEYS if k in p)
                        return (txt + " " + extras).casefold()

                    def semantic_hit(match_text: str, thresh: float = 0.13) -> bool:
                        if not match_text:
                            return False
                        if fm in match_text:  # fast path
                            return True
                        return SequenceMatcher(None, fm, match_text).ratio() >= thresh

                    for ch in batch_chunks:
                        if semantic_hit(compose_match_text(ch)):
                            hits[ch.get("source_label", "unknown")] += 1

                    return hits

                rows_total = []
                for b_idx, batch in enumerate(batches, start=1):
                    # LOG: count what we are feeding this batch
                    c = Counter([c["source_label"] for c in batch])
                    batch_source_counts.append(c)
                    context_source_counts.update(c)

                    # create combined context with explicit SOURCE tags
                    # combined_context_parts = []
                    # for chunk in batch:
                    #     header = f"SOURCE: {chunk['source_label']}"
                    #     combined_context_parts.append(f"{header}\n{chunk['text']}")
                    # combined_context = "\n\n".join(combined_context_parts)
                    combined_context_parts = []
                    for chunk in batch:
                        payload = chunk.get("payload", {})
                        src = chunk.get("source_label", "unknown")
                        file = payload.get("filename", "unknown_file")
                        # campaign = payload.get("campaign", "unknown_campaign")
                        # combined_context_parts.append(
                        #     f"[SOURCE: {src}] [FILE: {file}] [CAMPAIGN: {campaign}]\n{chunk['text']}"
                        # )
                        campaign_name = payload.get("campaign", "unknown_campaign")
                        combined_context_parts.append(
                            f"[SOURCE: {src}] [FILE: {file}] [CAMPAIGN: {campaign_name}]\n{chunk['text']}"
                        )
                    combined_context = "\n\n".join(combined_context_parts)


                    # 4) system prompt
                    system_msg_template = """
                    You are an expert reliability engineer creating a DFMEA table.

                    ### CRITICAL RULES (READ CAREFULLY)
                    - Output must be STRICT JSON array of objects matching keys exactly: {schema}
                    - No markdown, no extra commentary, only the JSON array.
                    - Always scope all rows to the SUBSYSTEM passed in the user message. Do not generate unrelated components.
                    - Severity, Occurrence, Detection must be integers between 1 and 10 inclusive. No percentages, no decimals.
                    - RPN must equal Severity × Occurrence × Detection (integer).
                    - "FM Origin" must always be populated.
                    - Rule for FM Origin:
                        • If the Failure Mode is supported by SOURCE: kb chunks → "kb"
                        • Else if supported by SOURCE: sprll chunks → "sprll"
                        • Else if supported by SOURCE: prds chunks → "prds"
                        • Else if supported by SOURCE: repair_data chunks → "repair_data"
                        • If no supporting chunk is found → "AI"
                    - FM Origin must be a single string exactly as above (kb, sprll, prds, repair_data, or AI).
                    - Functions must be full descriptive statements written in natural language (e.g., "Allows user interaction via touch input", not "Touch Input").
                    - Effects must explicitly describe the consequence to the user, the system, or product performance.
                    They must not be vague or overly short.
                    - Causes must explain the mechanism of failure (how/why), not just list a keyword.
                    - Controls Prevention must describe design/process measures in action-oriented form.
                    - Controls Detection must list at least 2 concrete, action-oriented inspection/test methods.
                    - Recommended Actions must include 2–3 precise engineering actions, phrased as imperatives.
                    - Do not repeat the same Component + Failure Mode + Cause combination across rows.
                    - Do not use the same Component more than 3 times.
                    - Generate diverse Effects (usability, reliability, cosmetic, safety, etc.)
                    - Generate at least 15–20 rows for this batch. Rows must be unique, non-generic, and meaningful.
                    Respond ONLY with the JSON array.

                    ### OUTPUT SHAPE ANCHOR (FORMAT ONLY — DO NOT COPY CONTENT)
                    - Return a JSON array.
                    - Each element MUST be an object with exactly these keys (no more, no less):
                    ["Subsystem","Component","Function","Failure Mode","Effect","Cause",
                    "Severity","Occurrence","Detection","RPN",
                    "Controls Prevention","Controls Detection","Recommended Actions","FM Origin"]
                    - Never return an empty array. If uncertain or evidence is weak, return a smaller set (≥5 and ≤20) supported by the context.
                    - If a row has no supporting chunk, you may still include it, but set "FM Origin" to "AI" and keep the description conservative and grounded in the subsystem.
                    """
                    system_msg = system_msg_template.format(schema=json.dumps(self.schema))

                    # 5) user message with explicit subsystem and context
                    user_msg = f"""
                    Generate DFMEA rows for subsystem: {subsystem}

                    Context (chunks are prefixed with SOURCE labels; use these to decide FM Origin):
                    {combined_context}
                    """
                    if admin_prompt:
                        user_msg += f"\nADMIN FILTER: {admin_prompt}\n"

                    try:
                        response = await asyncio.to_thread(
                            self.llm.chat.completions.create,
                            model=self.deployment,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=0.0,
                        )
                        raw_content = response.choices[0].message.content
                        logger.info(f"🧠 [ContextAgent] Raw LLM output (batch {b_idx}/{len(batches)}): len={len(raw_content)}")

                        # strip ```json ... ``` or ~~~ fences if present
                        cleaned = re.sub(
                            r'^\s*(?:```|~~~)[\w-]*\s*\n',
                            '',
                            raw_content.strip().replace("\r\n", "\n"),
                            flags=re.IGNORECASE,
                        )
                        cleaned = re.sub(r'\n\s*(?:```|~~~)\s*$', '', cleaned)
                        rows = json.loads(cleaned)

                    except Exception as e:
                        logger.error(f"❌ [ContextAgent] JSON parse failed for {subsystem}, batch {b_idx}: {e}")
                        rows = self._fallback(hits_for_fallback[:self.batch_size], subsystem)

                    # 6) Post-validate rows
                    validated = []
                    for row in rows:
                        # ensure subsystem label
                        row["Subsystem"] = subsystem

                        # numeric conversion
                        try:
                            row["Severity"] = int(float(str(row.get("Severity", "0")).strip().replace("%", "")))
                            row["Occurrence"] = int(float(str(row.get("Occurrence", "0")).strip().replace("%", "")))
                            row["Detection"] = int(float(str(row.get("Detection", "0")).strip().replace("%", "")))
                        except Exception:
                            continue

                        # per-row support accounting (informational)
                        support_hits = _support_sources_for_fm(row.get("Failure Mode"), batch)
                        if support_hits:
                            for src_label in ("kb", "sprll", "prds", "repair_data"):
                                if support_hits.get(src_label, 0) > 0:
                                    row_support_rows[src_label] += 1
                            row_support_chunk_hits.update(support_hits)
                        else:
                            rows_with_no_support += 1

                        # optional: fill S/O/D from best-matching repair_data payload if present
                        self._fill_sod_from_repair(row, rd_chunks)

                        # clamp to 1..10
                        if not (1 <= row["Severity"] <= 10 and 1 <= row["Occurrence"] <= 10 and 1 <= row["Detection"] <= 10):
                            continue

                        # recalc RPN
                        row["RPN"] = row["Severity"] * row["Occurrence"] * row["Detection"]

                        # FM Origin
                        row["FM Origin"] = self._determine_fm_origin(
                            row, kb_chunks, sprll_chunks, prd_chunks, rd_chunks
                        )

                        # normalize Controls Detection
                        cd = row.get("Controls Detection", [])
                        if isinstance(cd, str):
                            row["Controls Detection"] = [x.strip() for x in re.split(r"[;,]\s*", cd) if x.strip()]
                        elif not isinstance(cd, list):
                            row["Controls Detection"] = [str(cd)]

                        # ensure presence
                        if not row.get("Controls Prevention"):
                            row["Controls Prevention"] = ["TBD"]
                        if not row.get("Recommended Actions"):
                            row["Recommended Actions"] = ["TBD"]

                        validated.append(row)

                    rows_total.extend(validated)

                # 7) Deduplicate and enforce per-component limit
                unique_rows, seen = [], set()
                component_counts = {}
                for r in rows_total:
                    sig = (
                        r.get("Subsystem"), r.get("Component"), r.get("Function"),
                        r.get("Failure Mode"), r.get("Effect"), r.get("Cause"),
                    )
                    comp = (r.get("Component") or "").lower().strip()
                    fm = (r.get("Failure Mode") or "").lower().strip()

                    # exclude software/firmware related rows (focus = HW design only)
                    if any(term in fm for term in ["software", "firmware", "sw/fw"]) or \
                       any(term in comp for term in ["software", "firmware", "sw/fw"]):
                        continue

                    # enforce per-component limit
                    component_counts[comp] = component_counts.get(comp, 0) + 1
                    if component_counts[comp] > 6:
                        continue

                    # check for near-duplicate Failure Modes
                    duplicate = False
                    for kept in unique_rows:
                        fm_kept = (kept.get("Failure Mode") or "").lower().strip()
                        if fm and fm_kept:
                            sim = SequenceMatcher(None, fm, fm_kept).ratio()
                            if sim > 0.85:  # very similar
                                duplicate = True
                                break

                    if not duplicate and sig not in seen:
                        unique_rows.append(r)
                        seen.add(sig)

                logger.info(f"✅ [ContextAgent] Subsystem={subsystem} → {len(unique_rows)} rows after dedup/validate")

                # Apply Admin/Focus Prompt filtering (post-validation)
                if admin_prompt and unique_rows:
                    logger.info("🎛️ [ContextAgent] Applying Admin/Focus prompt filtering...")
                    rules = parse_focus_prompt_with_llm(admin_prompt, self.llm, self.schema, self.deployment)
                    if rules:
                        before = len(unique_rows)
                        unique_rows = apply_json_filters(unique_rows, rules, self.schema)
                        logger.info(f"🔻 [ContextAgent] Focus/Admin filtering reduced rows from {before} → {len(unique_rows)}")

                logger.info(f"🏁 [ContextAgent] Completed Subsystem={subsystem} → {len(unique_rows)} rows total")
                return unique_rows

        # run subsystem tasks concurrently
        tasks = [process_subsystem(sub) for sub in subsystems]
        results_nested = await asyncio.gather(*tasks)
        final_results = [item for sublist in results_nested for item in sublist]
        logger.info(f"🏁 [ContextAgent] Completed All Subsystems → {len(final_results)} rows total")

        # global row cap with smart ordering
        TARGET_TOTAL_ROWS = 200  # set your final desired total here

        if TARGET_TOTAL_ROWS and len(final_results) > TARGET_TOTAL_ROWS:
            # Prefer non-AI first, then higher RPN
            def _row_key(r):
                fm_origin = str(r.get("FM Origin", "")).strip().lower()
                rpn_val = r.get("RPN", 0)
                try:
                    rpn_num = int(rpn_val)
                except Exception:
                    try:
                        rpn_num = int(float(str(rpn_val)))
                    except Exception:
                        rpn_num = 0
                # sort: (AI last, RPN desc)
                return (fm_origin == "ai", -rpn_num)

            final_results.sort(key=_row_key)
            final_results = final_results[:TARGET_TOTAL_ROWS]
            logger.info(f"✂️ [ContextAgent] Trimmed to TARGET_TOTAL_ROWS={TARGET_TOTAL_ROWS} → {len(final_results)} rows")
        return final_results



    # --- helper to fill S/O/D from repair_data chunks ---
    def _fill_sod_from_repair(self, row: Dict, rd_chunks: List[Dict]) -> None:
        """
        If a repair_data chunk best supports this Failure Mode and carries S/O/D in its payload,
        overwrite the row's Severity/Occurrence/Detection from that payload.
        """
        fm = (row.get("Failure Mode") or "").lower().strip()
        if not fm or not rd_chunks:
            return

        best_payload = None
        best_score = 0.0

        for item in rd_chunks:
            # rd_chunks entries are tuples: (src, text, payload)
            text = (item[1] if isinstance(item, tuple) else item.get("text", "")) or ""
            payload = (item[2] if isinstance(item, tuple) else item.get("payload", {})) or {}
            t = text.lower()
            if not t:
                continue

            score = 1.0 if fm in t else SequenceMatcher(None, fm, t).ratio()
            if score > best_score:
                best_score = score
                best_payload = payload

        # accept only if at least a modest similarity
        if best_payload and best_score >= 0.13:
            try:
                s = best_payload.get("Severity", None)
                o = best_payload.get("Occurrence", None)
                d = best_payload.get("Detection", None)
                if s is not None and o is not None and d is not None:
                    s = int(s); o = int(o); d = int(d)
                    row["Severity"], row["Occurrence"], row["Detection"] = s, o, d
            except Exception:
                # If anything goes wrong, leave original values intact
                pass

    def _determine_fm_origin(
    self,
    row: Dict,
    kb_chunks: List[Dict],
    sprll_chunks: List[Dict],
    prd_chunks: List[Dict],
    rd_chunks: List[Dict],
) -> str:
        """
        Balanced final version (Repair restored):
        - Keeps KB/SPRLL/PRD/AI steady.
        - Brings back Repair Data (~10–15%) with gentle looseness.
        """
        import re
        from collections import Counter

        fm_raw = (row.get("Failure Mode") or "").strip()
        if not fm_raw:
            return "AI"

        # ---------- helpers ----------
        def norm(s: str) -> str:
            s = (s or "").casefold()
            s = s.replace("–", "-").replace("—", "-").replace("/", " ").replace("_", " ")
            s = re.sub(r'["“”]', "", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def get_payload(c):
            if isinstance(c, tuple):
                return c[2] if len(c) >= 3 else {}
            return c.get("payload", {}) if isinstance(c, dict) else {}

        def get_text(c) -> str:
            p = get_payload(c) or {}
            mt = p.get("match_text_fm")
            if mt:
                return norm(str(mt))
            if isinstance(c, tuple):
                return norm(c[1] or "")
            return norm((c.get("text") or ""))

        def tokenize(s: str):
            return [t for t in re.split(r"[^a-z0-9]+", s) if t]

        fm_norm = norm(fm_raw)
        fm_tokens = [t for t in tokenize(fm_norm) if len(t) >= 4]
        fm_token_set = set(fm_tokens)

        all_chunks = (kb_chunks or []) + (sprll_chunks or []) + (prd_chunks or []) + (rd_chunks or [])
        if not all_chunks:
            return "AI"

        MIN_TOKEN_LEN = 4
        GENERIC_DF_FRAC = 0.25

        df = Counter()
        total_docs = 0
        for ch in all_chunks:
            text = get_text(ch)
            if not text:
                continue
            total_docs += 1
            toks = {t for t in tokenize(text) if len(t) >= MIN_TOKEN_LEN}
            df.update(toks)

        generic = {t for t, c in df.items() if (c / max(1, total_docs)) >= GENERIC_DF_FRAC}
        fm_tokens_rare = [t for t in fm_token_set if t not in generic]
        fm_tokens_rare_set = set(fm_tokens_rare)

        def has_substring(chunks):
            for c in chunks:
                if fm_norm and fm_norm in get_text(c):
                    return True
            return False

        def rare_token_fallback(chunks, min_hits, min_coverage):
            if not chunks or not fm_tokens_rare_set:
                return False
            total_rare = len(fm_tokens_rare_set)
            for c in chunks:
                t = get_text(c)
                hits = {tok for tok in fm_tokens_rare_set if tok in t}
                if len(hits) >= min_hits and (len(hits) / max(1, total_rare)) >= min_coverage:
                    return True
            return False

        def direct_semantic_match(chunks):
            """Chunk-size independent mild check"""
            if not chunks:
                return False
            match_count = 0
            for c in chunks:
                text = get_text(c)
                if not text:
                    continue
                overlap = len(fm_token_set & set(tokenize(text)))
                if overlap >= 1:
                    match_count += 1
            return match_count > 0 and (match_count / max(1, len(chunks))) >= 0.10


        # === Phase 1: Repair Data (slightly controlled) ===
        if (has_substring(rd_chunks) or rare_token_fallback(rd_chunks, 5, 0.08) or direct_semantic_match(rd_chunks)) \
                and not rare_token_fallback(kb_chunks, 3, 0.25):
            return "repair_data"

        # === Phase 2: SPRLL ===
        if has_substring(sprll_chunks) or rare_token_fallback(sprll_chunks, 1, 0.10):
            return "sprll"

        # === Phase 3: PRDs ===
        if has_substring(prd_chunks) or rare_token_fallback(prd_chunks, 2, 0.25):
            return "prds"


        # === Phase 4: KB (slightly boosted) ===
        if has_substring(kb_chunks) or rare_token_fallback(kb_chunks, 3, 0.28):
            return "kb"

        # === Phase 5: AI fallback ===
        all_text = " ".join(get_text(c) for c in all_chunks)
        # if any(tok in all_text for tok in fm_tokens):
        #     return "AI"
        if not (has_substring(kb_chunks) or has_substring(sprll_chunks) or has_substring(prd_chunks) or has_substring(rd_chunks)):
            return "AI"


        return "AI"

    def _fallback(self, results: List[Dict], subsystem: str) -> List[Dict]:
        rows = []
        for r in results:
            payload = r.get("payload", {})
            rows.append(
                {
                    "Subsystem": subsystem,
                    "Component": payload.get("Component", ""),
                    "Function": payload.get("Function", ""),
                    "Failure Mode": payload.get("Failure Mode", ""),
                    "Effect": payload.get("Effect", ""),
                    "Cause": payload.get("Cause", ""),
                    "Severity": payload.get("Severity", ""),
                    "Occurrence": payload.get("Occurrence", ""),
                    "Detection": payload.get("Detection", ""),
                    "RPN": payload.get("RPN", ""),
                    "Controls Prevention": payload.get("Controls Prevention", ""),
                    "Controls Detection": payload.get("Controls Detection", ""),
                    "Recommended Actions": payload.get("Recommended Actions", "TBD"),
                    "FM Origin": "AI",  # Always AI in fallback
                }
            )
        return rows
