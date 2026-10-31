"""Structure-aware context selection and evidence-derived coverage checks."""

from dataclasses import dataclass
import re
from typing import Iterable

from .query_understanding import QueryAnalysis, QueryIntent, lexical_terms


class ContextCoverageError(ValueError):
    """The retrieved structural scope is incomplete."""


class AnswerCoverageError(ValueError):
    """The generated answer omits evidence units present in context."""


class CitationValidationError(ValueError):
    """A claim is not supported by its cited evidence scope."""


@dataclass(frozen=True)
class CoverageUnit:
    """A verifiable list item, condition, step, or scoped fact."""

    key: str
    text: str
    documents: tuple[int, ...] = ()


_CLAUSE_PATTERN = re.compile(r"(?m)(?:^|\n\s*\n)(\d+)\.\s+")
_LETTER_ITEM_PATTERN = re.compile(
    r"(?<!\w)([a-zđ])(?:\.|\))\s+", re.IGNORECASE
)
_STEP_PATTERN = re.compile(r"(?i)(?<!\w)bước\s+(\d+)\s*:\s*")
_BULLET_PATTERN = re.compile(r"(?:^|\s)-\s+(?=\S)")
_NEGATION_TERMS = {"không", "chưa", "khỏi", "cấm", "trừ"}
_VALIDATION_STOP_WORDS = {
    "các", "có", "của", "cho", "được", "là", "một", "những", "sinh",
    "thì", "trong", "và", "viên", "với", "theo", "do", "về",
}


def _article_key(chunk: dict) -> tuple[object, object, object] | None:
    article = chunk.get("article")
    if article is None:
        return None
    return (chunk.get("part"), chunk.get("chapter"), article)


def _chunk_identity(chunk: dict) -> tuple[object, ...]:
    return (
        _article_key(chunk),
        tuple(chunk.get("clauses") or []),
        chunk.get("page_start"),
        chunk.get("page_end"),
        chunk.get("text"),
    )


def _copy_fragment(chunk: dict, text: str, clauses: list[int]) -> dict:
    fragment = {**chunk, "text": text.strip(), "clauses": clauses}
    fragment["_origin_identity"] = _chunk_identity(chunk)
    return fragment


def split_structural_fragments(chunk: dict) -> list[dict]:
    """Split mixed chunks at top-level clauses while preserving metadata."""
    text = str(chunk.get("text") or "").strip()
    if not text:
        return []
    matches = list(_CLAUSE_PATTERN.finditer(text))
    if not matches:
        return [_copy_fragment(chunk, text, list(chunk.get("clauses") or []))]

    fragments: list[dict] = []
    first_start = matches[0].start()
    if first_start > 0:
        prefix = text[:first_start].strip()
        if prefix:
            # Content before the first numbered clause is unnumbered. This is
            # important when a PDF page starts a neighboring numbered list.
            fragments.append(_copy_fragment(chunk, prefix, []))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fragments.append(
            _copy_fragment(chunk, text[start:end], [int(match.group(1))])
        )
    return fragments


def _scope_score(analysis: QueryAnalysis, chunk: dict) -> float:
    text = str(chunk.get("text") or "")
    if not chunk.get("clauses") and chunk.get("article_title"):
        text = f"{chunk['article_title']} {text}"
    terms = set(lexical_terms(text))
    focus = set(analysis.focus_terms)
    coverage = len(terms & focus) / len(focus) if focus else 0.0
    density = len(terms & focus) / max(len(terms), 1)
    score = coverage + min(density * 4.0, 0.35)
    folded = text.casefold()
    positions = [folded.find(term) for term in focus if term in folded]
    if positions:
        latest_first_match = max(positions)
        score += 0.35 * (1.0 - min(latest_first_match / 240.0, 1.0))
    question_terms = lexical_terms(analysis.normalized_question)
    phrase_bonus = 0.0
    for size, bonus in ((3, 0.9), (2, 0.3)):
        if any(
            " ".join(question_terms[index:index + size]) in folded
            for index in range(max(0, len(question_terms) - size + 1))
        ):
            phrase_bonus = bonus
            break
    score += phrase_bonus
    if analysis.has_intent(QueryIntent.PROCEDURE) and (
        "bước 1" in folded or "trình tự" in folded
    ):
        score += 0.6
    if analysis.has_intent(QueryIntent.CONDITIONS) and any(
        cue in folded
        for cue in ("điều kiện sau", "các trường hợp sau", "nếu ", "khi ")
    ):
        score += 0.3
    if analysis.has_intent(QueryIntent.CONDITIONS) and positions:
        list_intro = folded.find("các trường hợp sau")
        target_position = folded.find(analysis.focus_terms[0])
        if 0 <= list_intro < target_position:
            score -= 0.6
        if "nếu có nhu cầu" in folded:
            score -= 0.9
        if "trừ trường hợp" in folded and folded.find("trừ trường hợp") < target_position:
            score -= 0.8
    if analysis.has_intent(QueryIntent.NUMERIC_THRESHOLD) and re.search(
        r"\d", text
    ):
        score += 0.1
    return score


def _scope_key(chunk: dict) -> tuple[object, ...]:
    article_key = _article_key(chunk)
    clauses = tuple(chunk.get("clauses") or [])
    if article_key is not None:
        return (*article_key, clauses)
    return (
        chunk.get("part"),
        chunk.get("page_start"),
        chunk.get("page_end"),
        clauses,
        chunk.get("_origin_identity"),
    )


def _deduplicate(chunks: Iterable[dict]) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple[object, ...]] = set()
    for chunk in chunks:
        identity = (
            _chunk_identity(chunk),
            chunk.get("text"),
            tuple(chunk.get("clauses") or []),
        )
        if identity not in seen:
            seen.add(identity)
            selected.append(chunk)
    return selected


def _merge_same_scope_fragments(chunks: list[dict]) -> list[dict]:
    """Reassemble split pieces of one clause into one citation document."""
    merged: list[dict] = []
    positions: dict[tuple[object, ...], int] = {}
    for chunk in chunks:
        key = _scope_key(chunk)
        if key not in positions:
            positions[key] = len(merged)
            merged.append(dict(chunk))
            continue

        target = merged[positions[key]]
        existing = str(target.get("text") or "").rstrip()
        addition = str(chunk.get("text") or "").lstrip()
        overlap = 0
        limit = min(len(existing), len(addition))
        for size in range(limit, 19, -1):
            if existing[-size:] == addition[:size]:
                overlap = size
                break
        separator = "" if overlap else "\n\n"
        target["text"] = existing + separator + addition[overlap:]
        starts = [target.get("page_start"), chunk.get("page_start")]
        ends = [target.get("page_end"), chunk.get("page_end")]
        valid_starts = [value for value in starts if value is not None]
        valid_ends = [value for value in ends if value is not None]
        target["page_start"] = min(valid_starts) if valid_starts else None
        target["page_end"] = max(valid_ends) if valid_ends else None
    return merged


def prepare_structured_context(
    analysis: QueryAnalysis,
    results: list[dict],
    all_chunks: list[dict],
    *,
    max_chars: int,
    max_chunks: int,
) -> list[dict]:
    """Select relevant clauses and expand only their structural containers."""
    if not results:
        return []

    initial_fragments = [
        fragment
        for result in results
        for fragment in split_structural_fragments(result)
    ]
    if not analysis.requires_complete_coverage:
        selected = _deduplicate(initial_fragments)
        if analysis.has_intent(QueryIntent.NUMERIC_THRESHOLD) and selected:
            scored_numeric = [
                (chunk, _scope_score(analysis, chunk)) for chunk in selected
            ]
            best_numeric = max(score for _, score in scored_numeric)
            selected = [
                chunk
                for chunk, score in scored_numeric
                if score >= best_numeric * 0.98
            ]
        selected = selected[:max_chunks]
        for chunk in selected:
            chunk["_context_complete"] = True
        return selected

    scored_initial = [
        (fragment, _scope_score(analysis, fragment))
        for fragment in initial_fragments
    ]
    best_initial = max((score for _, score in scored_initial), default=0.0)
    focus = set(analysis.focus_terms)
    title_coverages: dict[tuple[object, object, object], float] = {}
    for fragment, _ in scored_initial:
        article_key = _article_key(fragment)
        if article_key is None or not focus:
            continue
        title_terms = set(lexical_terms(str(fragment.get("article_title") or "")))
        title_coverages[article_key] = max(
            title_coverages.get(article_key, 0.0),
            len(title_terms & focus) / len(focus),
        )
    best_title_coverage = max(title_coverages.values(), default=0.0)
    has_step_candidate = any(
        _STEP_PATTERN.search(fragment.get("text", ""))
        for fragment in initial_fragments
    )
    if analysis.has_intent(QueryIntent.PROCEDURE) and has_step_candidate:
        anchor_articles: set[tuple[object, object, object]] = set()
    elif best_title_coverage >= 0.75:
        anchor_articles = {
            key
            for key, coverage in title_coverages.items()
            if coverage >= best_title_coverage * 0.9
        }
    else:
        anchor_threshold = max(0.35, best_initial * 0.9)
        anchor_articles = {
            _article_key(fragment)
            for fragment, score in scored_initial
            if _article_key(fragment) is not None and score >= anchor_threshold
        }

    title_anchored_enumeration = (
        bool(anchor_articles)
        and best_title_coverage >= 0.75
        and analysis.has_intent(QueryIntent.ENUMERATION)
    )

    expanded_chunks = list(results)
    if title_anchored_enumeration:
        expanded_chunks = [
            chunk for chunk in results if _article_key(chunk) in anchor_articles
        ]
        expanded_chunks.extend(
            chunk for chunk in all_chunks if _article_key(chunk) in anchor_articles
        )
    elif anchor_articles:
        # Anchor chỉ dùng để mở rộng ngữ cảnh, không loại kết quả từ Điều khác.
        expanded_chunks.extend(
            chunk for chunk in all_chunks
            if _article_key(chunk) in anchor_articles
        )
    candidates = _deduplicate(
        fragment
        for chunk in expanded_chunks
        for fragment in split_structural_fragments(chunk)
    )
    scored = [(chunk, _scope_score(analysis, chunk)) for chunk in candidates]
    best_score = max((score for _, score in scored), default=0.0)

    condition_cues = (
        "điều kiện sau",
        "trường hợp",
        "nếu ",
        "sẽ bị",
        "đối với",
        " mà ",
    )
    is_condition_query = analysis.has_intent(QueryIntent.CONDITIONS)

    # Nới ngưỡng riêng cho câu hỏi điều kiện.
    if is_condition_query:
        threshold_ratio = 0.80
    elif "khi nào" in analysis.normalized_question.casefold():
        threshold_ratio = 0.82
    else:
        threshold_ratio = 0.90

    scope_threshold = max(0.35, best_score * threshold_ratio)

    title_anchored_enumeration = (
        bool(anchor_articles)
        and best_title_coverage >= 0.75
        and analysis.has_intent(QueryIntent.ENUMERATION)
    )

    # Nếu đã có kết quả Điều/Khoản đủ mạnh,
    # ưu tiên các quy định có cấu trúc Điều/Khoản (loại trừ tài liệu nhiễu không có Điều/Khoản).
    structured_scopes = [
        (chunk, score)
        for chunk, score in scored
        if chunk.get("article") is not None
    ]
    has_strong_structured_match = any(
        score >= scope_threshold
        for _, score in structured_scopes
    )

    candidate_pool = (
        structured_scopes
        if has_strong_structured_match
        else scored
    )

    if title_anchored_enumeration:
        selected_scope_keys = {
            _scope_key(chunk)
            for chunk, score in candidate_pool
            if _article_key(chunk) in anchor_articles
        }
    else:
        selected_scope_keys = {
            _scope_key(chunk)
            for chunk, score in candidate_pool
            if score >= scope_threshold
            and (
                not is_condition_query
                or any(
                    cue in str(chunk.get("text") or "").casefold()
                    for cue in condition_cues
                )
            )
        }

    if not selected_scope_keys and candidate_pool:
        # Do not erase a direct factual passage merely because it lacks one of
        # the optional linguistic condition cues used for reranking.
        fallback_score = max(score for _, score in candidate_pool)
        selected_scope_keys = {
            _scope_key(chunk)
            for chunk, score in candidate_pool
            if score == fallback_score
        }

    if "buộc thôi học" in analysis.normalized_question.casefold():
        print("\n========== STRUCTURE SCORING ==========")
        print("Focus terms:", analysis.focus_terms)
        print(f"Best score: {best_score:.4f}")
        print(f"Scope threshold: {scope_threshold:.4f}")
        print("Anchor articles:", anchor_articles)
        print("Title anchored enumeration:", title_anchored_enumeration)
        condition_cues = (
            "điều kiện sau",
            "trường hợp",
            "nếu ",
            "sẽ bị",
            "đối với",
            " mà ",
        )
        for chunk, score in sorted(
            scored,
            key=lambda item: item[1],
            reverse=True,
        ):
            key = _scope_key(chunk)
            text = str(chunk.get("text") or "")
            has_condition_cue = any(
                cue in text.casefold() for cue in condition_cues
            )
            is_selected = key in selected_scope_keys
            print(
                f"\nScore: {score:.4f} | Selected: {is_selected} | HasCue: {has_condition_cue} "
                f"| Phần {chunk.get('part')} | Điều {chunk.get('article')} | Khoản {chunk.get('clauses')} | {chunk.get('article_title')}"
            )
            first_line = " ".join(text.split())[:120]
            print(f"  Snippet: {first_line}")
        print("========== END STRUCTURE SCORING ==========\n")
    scoped = [chunk for chunk in candidates if _scope_key(chunk) in selected_scope_keys]
    scoped.sort(
        key=lambda chunk: (
            str(chunk.get("part") or ""),
            str(chunk.get("chapter") or ""),
            chunk.get("article") if chunk.get("article") is not None else 10**6,
            chunk.get("page_start") if chunk.get("page_start") is not None else 10**6,
            tuple(chunk.get("clauses") or []),
        )
    )

    selected: list[dict] = []
    used_chars = 0
    complete = True
    for chunk in scoped:
        text = str(chunk.get("text") or "")
        if len(selected) >= max_chunks or (
            selected and used_chars + len(text) > max_chars
        ):
            complete = False
            continue
        selected.append(chunk)
        used_chars += len(text)

    selected = _merge_same_scope_fragments(selected)
    for chunk in selected:
        chunk["_context_complete"] = complete
    return selected


def context_is_relevant(
    analysis: QueryAnalysis,
    results: list[dict],
    *,
    known_terms: set[str] | None = None,
) -> bool:
    if not results:
        return False
    focus = set(analysis.focus_terms)
    if not focus:
        return True

    title_terms = set()
    for chunk in results:
        title = str(chunk.get("article_title") or "")
        title_terms.update(lexical_terms(title))

    if known_terms is not None:
        normalized_vocabulary = set()
        for term in known_terms:
            normalized_vocabulary.update(lexical_terms(term))
        searchable_terms = normalized_vocabulary | title_terms
        if focus - searchable_terms:
            return False

    best_coverage = 0.0
    for chunk in results:
        title = str(chunk.get("article_title") or "")
        content = str(chunk.get("text") or "")
        relevance_text = f"{title} {content}"
        terms = set(lexical_terms(relevance_text))
        coverage = len(terms & focus) / len(focus)
        best_coverage = max(best_coverage, coverage)
    return best_coverage >= 0.75


def _marked_units(text: str) -> list[tuple[str, str]]:
    for prefix, pattern in (("step", _STEP_PATTERN), ("item", _LETTER_ITEM_PATTERN)):
        matches = list(pattern.finditer(text))
        labels = {match.group(1).casefold() for match in matches}
        if len(labels) < 2:
            continue
        units: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            units.append((f"{prefix}-{match.group(1).casefold()}", text[match.start():end].strip()))
        return units

    bullets = list(_BULLET_PATTERN.finditer(text))
    if len(bullets) >= 2:
        return [
            (
                f"bullet-{index + 1}",
                text[match.end():bullets[index + 1].start() if index + 1 < len(bullets) else len(text)].strip(),
            )
            for index, match in enumerate(bullets)
        ]
    return []


def extract_coverage_units(
    analysis: QueryAnalysis,
    results: list[dict],
) -> list[CoverageUnit]:
    """Derive expected answer units from the selected evidence, not a topic."""
    if not analysis.requires_complete_coverage:
        return []
    grouped: dict[tuple[object, ...], list[tuple[int, str]]] = {}
    for document, chunk in enumerate(results, start=1):
        grouped.setdefault(_scope_key(chunk), []).append(
            (document, str(chunk.get("text") or ""))
        )

    units: list[CoverageUnit] = []
    for scope_index, entries in enumerate(grouped.values(), start=1):
        marked: dict[str, tuple[str, int]] = {}
        for document, text in entries:
            for label, unit_text in _marked_units(text):
                current = marked.get(label)
                if current is None or len(unit_text) > len(current[0]):
                    marked[label] = (unit_text, document)
        if marked:
            for label, (unit_text, document) in marked.items():
                units.append(
                    CoverageUnit(
                        f"S{scope_index}-{label}", unit_text, (document,)
                    )
                )
        else:
            combined = " ".join(
                text.strip() for _, text in entries if text.strip()
            )
            if combined:
                units.append(
                    CoverageUnit(
                        f"S{scope_index}",
                        combined,
                        tuple(document for document, _ in entries),
                    )
                )
    return units


def coverage_requirements(units: list[CoverageUnit]) -> list[str]:
    """Render bounded evidence-derived hints for the answer model."""
    if not units:
        return []

    labels = ", ".join(unit.key for unit in units)
    requirements = [
        f"Checklist có {len(units)} đơn vị độc lập ({labels}). "
        f"Phải tạo ít nhất {len(units)} claim theo đúng thứ tự, mỗi đơn vị có "
        "claim và citation riêng; không dùng một đơn vị để thay cho đơn vị khác."
    ]
    for unit in units:
        excerpt = " ".join(unit.text.split())
        if len(excerpt) > 320:
            excerpt = excerpt[:317].rstrip() + "..."
        documents = ", ".join(str(document) for document in unit.documents)
        location = f" (Tài liệu {documents})" if documents else ""
        requirements.append(f"Coverage {unit.key}{location}: {excerpt}")
    return requirements


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in lexical_terms(text)
        if term not in _VALIDATION_STOP_WORDS and len(term) > 1
    }


def _unit_supported(unit: CoverageUnit, citations: list[dict]) -> bool:
    normalized_unit = " ".join(unit.text.casefold().split())
    unit_terms = _content_terms(unit.text)
    for citation in citations:
        if unit.documents and citation.get("document") not in unit.documents:
            continue
        evidence = str(citation.get("evidence") or "")
        normalized_evidence = " ".join(evidence.casefold().split())
        if normalized_evidence and (
            normalized_evidence in normalized_unit or normalized_unit in normalized_evidence
        ):
            return True
        cited_terms = _content_terms(evidence)
        if unit_terms and len(unit_terms & cited_terms) / len(unit_terms) >= 0.75:
            return True
    return False


def validate_answer_coverage(
    units: list[CoverageUnit],
    citations: list[dict],
) -> None:
    """Require every extracted information unit to have direct cited support."""
    missing = [unit.key for unit in units if not _unit_supported(unit, citations)]
    if missing:
        raise AnswerCoverageError(
            "Câu trả lời thiếu các đơn vị evidence: " + ", ".join(missing)
        )


def validate_answer_style(
    analysis: QueryAnalysis,
    citations: list[dict],
) -> None:
    """Keep broad list answers concise without weakening evidence checks."""
    if not analysis.has_intent(QueryIntent.ENUMERATION):
        return
    question = analysis.normalized_question.casefold()
    if any(
        cue in question
        for cue in ("chi tiết", "cụ thể", "tất cả", "đầy đủ")
    ):
        return

    for citation in citations:
        evidence = str(citation.get("evidence") or "")
        claim = str(citation.get("claim") or "")
        if "cụ thể:" in evidence.casefold() and len(claim.split()) > 40:
            raise AnswerCoverageError(
                "Claim đang chép danh sách ví dụ sau 'Cụ thể:' quá dài; "
                "hãy chỉ nêu ngắn gọn tên nhóm pháp lý trong claim và giữ "
                "đoạn nguyên văn ở evidence"
            )


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", text))


def validate_claim_faithfulness(
    analysis: QueryAnalysis,
    results: list[dict],
    citations: list[dict],
) -> None:
    """Check lexical entailment, scope, negation, and numeric preservation."""
    scope_texts: dict[tuple[object, ...], str] = {}
    for chunk in results:
        key = _scope_key(chunk)
        title = (
            str(chunk.get("article_title") or "")
            if not chunk.get("clauses")
            else ""
        )
        scope_texts[key] = (
            f"{scope_texts.get(key, '')} {title} {chunk.get('text', '')}".strip()
        )

    for citation in citations:
        document = citation.get("document")
        if not isinstance(document, int) or not 1 <= document <= len(results):
            raise CitationValidationError("Citation chứa document không hợp lệ")
        scope_text = scope_texts[_scope_key(results[document - 1])]
        claim = str(citation.get("claim") or "")
        claim_terms = _content_terms(claim)
        scope_terms = _content_terms(scope_text)
        if claim_terms and len(claim_terms & scope_terms) / len(claim_terms) < 0.45:
            raise CitationValidationError(
                "Claim không được evidence trong đúng phạm vi viện dẫn hỗ trợ"
            )

        focus = set(analysis.focus_terms)
        descriptive_terms = {
            term
            for term in claim_terms - focus
            if any(character.isalpha() for character in term)
        }
        if not descriptive_terms:
            raise CitationValidationError(
                "Claim chỉ nêu nhãn hoặc tỷ lệ mà chưa nêu đối tượng hay điều kiện áp dụng"
            )
        claim_focus = focus & set(lexical_terms(claim))
        required_focus = claim_focus or focus
        if (
            required_focus
            and len(required_focus & scope_terms) / len(required_focus) < 0.75
        ):
            raise CitationValidationError(
                "Citation không thuộc đúng phạm vi khái niệm của câu hỏi"
            )

        claim_negation = _NEGATION_TERMS & set(lexical_terms(claim))
        scope_negation = _NEGATION_TERMS & set(lexical_terms(scope_text))
        if claim_negation and not scope_negation:
            raise CitationValidationError(
                "Claim phủ định nhưng evidence không chứa căn cứ phủ định"
            )

        claim_numbers = _numeric_tokens(claim)
        if not claim_numbers.issubset(_numeric_tokens(scope_text)):
            raise CitationValidationError(
                "Claim chứa số hoặc ngưỡng không có trong evidence"
            )
