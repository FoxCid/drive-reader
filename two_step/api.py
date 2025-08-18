from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Callable
import importlib, inspect

logger = logging.getLogger(__name__)

# ======================================================================================
# Résolution robuste & PARESSEUSE (lazy) de la fonction "phase 2"
# ======================================================================================

_CANDIDATE_MODULES = [
    "two_step.semantic_search_two_step",   # s'il existe un shim
    "semantic_search_two_step",            # fichier à la racine (cas le plus probable)
    "semantic_search_20250713",            # legacy
]
_CANDIDATE_NAMES = [
    "two_step_search",
    "semantic_search_two_step",
    "semantic_search",
    "search_two_step",
    "search",
]

def _looks_like_phase2(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return False
    # on veut au moins 'query' OU 'question'
    return ("query" in params) or ("question" in params)

def _resolve_phase2() -> Callable[..., Any]:
    for mod in _CANDIDATE_MODULES:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        # noms probables
        for name in _CANDIDATE_NAMES:
            fn = getattr(m, name, None)
            if callable(fn) and _looks_like_phase2(fn):
                return fn
        # balayage heuristique
        for name, obj in vars(m).items():
            if callable(obj) and any(t in name for t in ("two_step", "phase2", "semantic_search", "search")) and _looks_like_phase2(obj):
                return obj
    raise ImportError("two_step.api: aucune implémentation 'phase 2' trouvée")

# en haut du fichier tu as déjà: from typing import Any, Optional, Callable

_PHASE2_FN: Optional[Callable[..., Any]] = None  # garde cette ligne une seule fois

def semantic_search_phase2_only(
    *,
    query: Optional[str] = None,
    role: str = "",
    top_k: int = 8,
    **kwargs: Any,
):
    """
    Proxy paresseux vers l'implémentation Phase 2.
    Signature explicite pour satisfaire les tests : query, role, top_k.
    - Accepte aussi 'question=' comme alias de 'query'.
    - Forwarde tous les autres kwargs tels quels.
    """
    global _PHASE2_FN
    # compat : autoriser 'question='
    if query is None and kwargs.get("question") is not None:
        query = kwargs.pop("question")
    if query is None:
        raise TypeError("semantic_search_phase2_only() missing required 'query'")

    if _PHASE2_FN is None:
        _PHASE2_FN = _resolve_phase2()

    return _PHASE2_FN(query=query, role=role, top_k=top_k, **kwargs)


__all__ = ["semantic_search_phase2_only"]

# ======================================================================================
# Hooks simples (patchables dans les tests)
# ======================================================================================

def context_is_sufficient(summary: str) -> bool:
    """
    Heuristique par défaut: considère suffisant sauf si le résumé est vide/annoté '(vide)'.
    Les tests monkeypatchent cette fonction -> on respecte leur verdict.
    """
    if not summary:
        return False
    s = str(summary).strip().lower()
    return "(vide)" not in s and "aucune" not in s

def enrich_with_web(query: str, role: str):
    """No-op par défaut, patché dans les tests."""
    return []

# ======================================================================================
# Utilitaires de normalisation/compat (patchables)
# ======================================================================================

def _normalize_synthesize_output(
    resp: Any,
    *,
    ranked_chunks: Optional[List[Dict[str, Any]]] = None,
    summary_context: Optional[str] = None,
):
    """
    Normalise la sortie de synthesize_answer en (answer, sources_dict, used_debug, summary).
    Accepte :
      - (answer, sources_dict, used_debug, summary)
      - (answer, sources_dict, used_debug)
      - (answer, info_dict) où info_dict peut contenir 'sources', 'chunks_used', 'summary'
      - { "answer":..., "sources":..., "chunks_used":..., "summary":... }
      - "answer" (fallback extrême)
    """
    default_used = {"chunks_used": [c.get("text", "") for c in (ranked_chunks or [])]}
    answer: str = ""
    sources_dict: Any = {"sources": []}
    used_debug: Dict[str, Any] = dict(default_used)
    summary: Optional[str] = summary_context

    def _merge_info(d: Dict[str, Any]):
        nonlocal sources_dict, used_debug, summary
        if not isinstance(d, dict):
            return
        if "sources" in d:
            sources_dict = {"sources": d.get("sources") or []}
        if "chunks_used" in d:
            used_debug = {"chunks_used": d.get("chunks_used") or default_used["chunks_used"]}
        if summary is None and "summary" in d:
            summary = d.get("summary")

    if isinstance(resp, tuple):
        items = list(resp)
        if items:
            answer = str(items[0])
        for x in items[1:]:
            if isinstance(x, dict):
                _merge_info(x)
            elif isinstance(x, list):
                if isinstance(sources_dict, dict) and not sources_dict.get("sources"):
                    sources_dict = {"sources": x}
            elif isinstance(x, str) and summary is None:
                summary = x
    elif isinstance(resp, dict):
        answer = str(resp.get("answer", ""))
        _merge_info(resp)
    else:
        answer = str(resp)

    return answer, sources_dict, used_debug, summary

def to_standard_chunk(c: Any) -> Dict[str, Any]:
    """Transforme un chunk (dict/Document-like) en dict standardisé."""
    if isinstance(c, dict):
        meta = c.get("metadata") or {}
        if "source" not in meta and c.get("source"):
            meta["source"] = c["source"]
        text = c.get("text") if c.get("text") is not None else c.get("page_content", "")
        out = dict(c)
        out["text"] = text if isinstance(text, str) else str(text)
        out["metadata"] = meta
        out["source"] = out.get("source") or meta.get("source", "")
        return out
    text = getattr(c, "page_content", "")
    meta = getattr(c, "metadata", {}) or {}
    src = meta.get("source", "")
    return {"text": text, "metadata": meta, "source": src}

def enhance_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    """Assure la présence des champs de score; idempotent."""
    meta = c.get("metadata") or {}
    meta.setdefault("faiss_score", 0.0)
    meta.setdefault("bm25_score", 0.0)
    meta.setdefault("entity_score", 0.0)
    meta.setdefault("bonus_metadata", 0.0)
    meta.setdefault("feedback_penalty", 0.0)
    c["metadata"] = meta
    c["source"] = c.get("source") or meta.get("source", "")
    return c

# ======================================================================================
# Points d’extension (délèguent aux modules internes quand présents)
# ======================================================================================

def extract_entities_keywords(q: str) -> Tuple[List[str], List[str]]:
    try:
        from . import retrieval as _r
        return _r.extract_entities_keywords(q)
    except Exception:
        return [], [w for w in q.split() if w]

def retrieve_context_chunks(query: str, role: str, top_k: int = 8) -> List[Dict[str, Any]]:
    try:
        from . import retrieval as _r
        return _r.retrieve_context_chunks(query, role=role, top_k=top_k)
    except Exception:
        return []

def summarize_context_for_role(chunks: List[Dict[str, Any]], role: str) -> str:
    try:
        from . import retrieval as _r
        return _r.summarize_context_for_role(chunks, role=role)
    except Exception:
        srcs = [to_standard_chunk(c).get("source", "") for c in chunks]
        srcs = [s for s in srcs if s]
        if not srcs:
            return "- Contexte:\n- Sources:\n  - (aucune)"
        return "- Contexte pour {role}\n- Sources:\n" + "\n".join(f"- {s}" for s in srcs)

def retrieve_general_chunks(query: str, role: str, top_k: int = 64) -> List[Any]:
    try:
        from . import retrieval as _r
        return _r.retrieve_general_chunks(query, role=role, top_k=top_k)
    except Exception:
        return []

def score_sort_and_rerank(
    chunks: List[Dict[str, Any]],
    question: str,
    entities: Optional[List[str]] = None,
    feedback_data: Any = None,
    top_k: int = 64,
) -> List[Dict[str, Any]]:
    """
    Impl par défaut : calcule un total_score simple à partir de metadata.
    Patchable dans les tests.
    """
    out: List[Dict[str, Any]] = []
    for c in chunks:
        m = c.get("metadata", {}) or {}
        total = (
            float(m.get("faiss_score", 0.0))
            + float(m.get("bm25_score", 0.0))
            + float(m.get("entity_score", 0.0))
            + float(m.get("bonus_metadata", 0.0))
            - float(m.get("feedback_penalty", 0.0))
        )
        m["total_score"] = total
        c["metadata"] = m
        out.append(c)
    out.sort(key=lambda x: x.get("metadata", {}).get("total_score", 0.0), reverse=True)
    return out[: max(1, top_k)]

def select_final_chunks(
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
    min_score: float = 0.0,
    diversify: bool = True,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for c in chunks:
        score = float(c.get("metadata", {}).get("total_score", 0.0))
        if score >= float(min_score):
            kept.append(c)
        if len(kept) >= int(top_k):
            break
    return kept

def generate_final_answer(question: str, chunks: List[Dict[str, Any]], role: str = "") -> str:
    if not chunks:
        return "Je n’ai pas trouvé de document pertinent."
    sources = []
    for c in chunks:
        src = c.get("source") or c.get("metadata", {}).get("source", "")
        if src and src not in sources:
            sources.append(src)
    body = f"Réponse synthétique pour: {question}\n\n"
    body += "\n".join(f"- {c.get('text','')}" for c in chunks[:3])
    body += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    return body

def synthesize_answer(
    question: str,
    ranked_chunks: List[Dict[str, Any]],
    role: str = "",
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[str]]:
    """Point d'extension (patchable). Retourne (answer, sources_dict, used_dict, summary)."""
    answer = generate_final_answer(question, ranked_chunks, role=role)
    sources = {"sources": list({c.get("source") or c.get("metadata", {}).get("source", "") for c in ranked_chunks if c})}
    used = {"chunks_used": [c.get("text", "") for c in ranked_chunks]}
    return answer, sources, used, None

# ======================================================================================
# Orchestrateur central
# ======================================================================================

def _is_context_only_role(role: str) -> bool:
    try:
        from . import config as _cfg
    except Exception:
        return False
    value = getattr(_cfg, "CONTEXT_ONLY_ROLES", {})
    if isinstance(value, dict):
        return bool(value.get(role, False))
    if isinstance(value, (set, list, tuple)):
        return role in value
    return False

def _min_score_from_env(default: float = 0.0) -> float:
    try:
        return float(os.getenv("CHUNK_MIN_SCORE", str(default)))
    except Exception:
        return default

def run_two_step(
    question: str,
    role: str = "",
    top_k: int = 8,
    retrieval_only: bool = False,
    **kwargs: Any,
) -> Tuple[str, List[str], Dict[str, Any], Optional[str]]:
    """
    Pipeline principal. Retourne:
      (answer, sources_list, used_debug_dict, summary_context)
    """
    logger.info("run_two_step(question=%s, role=%s, top_k=%s, retrieval_only=%s)", question, role, top_k, retrieval_only)

    entities, _keywords = extract_entities_keywords(question)

    # Phase 1 : contexte
    ctx_chunks_raw = retrieve_context_chunks(question, role=role, top_k=min(8, max(1, top_k)))
    ctx_summary = summarize_context_for_role(ctx_chunks_raw, role=role)

    # Fallback Web si contexte insuffisant (hook patchable)
    try:
        need_web = not context_is_sufficient(ctx_summary)
    except Exception:
        need_web = False
    if need_web:
        try:
            _ = enrich_with_web(question, role)
        except Exception:
            pass

    # Choix des chunks à utiliser
    use_context_only = _is_context_only_role(role)
    if use_context_only:
        base_raw = ctx_chunks_raw
    else:
        base_raw = retrieve_general_chunks(question, role=role, top_k=max(64, top_k * 4))

    # Normalisation -> scoring -> sélection
    base_std = [enhance_chunk(to_standard_chunk(c)) for c in (base_raw or [])]
    ranked = score_sort_and_rerank(
        base_std,
        question=question,
        entities=entities,
        feedback_data=None,
        top_k=max(64, top_k * 4),
    )
    selected = select_final_chunks(
        ranked,
        top_k=int(top_k),
        min_score=_min_score_from_env(0.0),
        diversify=True,
    )

    # Synthèse
    resp = synthesize_answer(
        question=question,
        role=role,
        ranked_chunks=(base_std if use_context_only else ranked),
        summary_context=ctx_summary,
    )
    answer, sources_dict, used_debug, _ = _normalize_synthesize_output(
        resp,
        ranked_chunks=(base_std if use_context_only else ranked),
        summary_context=ctx_summary,
    )

    sources_list = sources_dict.get("sources", []) if isinstance(sources_dict, dict) else []
    return answer, sources_list, used_debug, ctx_summary

# ======================================================================================
# Entrée API "complet" attendue par certains appels/tests
# ======================================================================================

def semantic_search_two_step(
    *,
    query: Optional[str] = None,
    question: Optional[str] = None,
    role: str = "",
    top_k: int = 8,
    model: Optional[str] = None,
    retrieval_only: bool = False,
    **kwargs: Any,
) -> Any:
    """
    - retrieval_only=True  -> retourne **une LISTE de chunks normalisés**
    - retrieval_only=False -> retourne (answer, sources, chunks, summary)
    """
    # compat "question"
    if query is None and question is not None:
        query = question
    if query is None:
        raise TypeError("semantic_search_two_step() missing required 'query'")

    logger.info(
        "two_step.semantic_search_two_step(query=%s, role=%s, top_k=%s, retrieval_only=%s)",
        query, role, top_k, retrieval_only
    )

    if retrieval_only:
        entities, _ = extract_entities_keywords(query)
        ctx_chunks_raw = retrieve_context_chunks(query, role=role, top_k=min(8, max(1, top_k)))
        use_context_only = _is_context_only_role(role)
        if use_context_only:
            base_raw = ctx_chunks_raw
        else:
            base_raw = retrieve_general_chunks(query, role=role, top_k=max(64, top_k * 4))

        base_std = [enhance_chunk(to_standard_chunk(c)) for c in (base_raw or [])]
        ranked = score_sort_and_rerank(
            base_std, question=query, entities=entities, feedback_data=None, top_k=max(64, top_k * 4)
        )
        selected = select_final_chunks(
            ranked, top_k=int(top_k), min_score=_min_score_from_env(0.0), diversify=True,
        )
        return selected

    # Mode complet : renvoie le tuple classique
    answer, _sources_list, used_debug, summary = run_two_step(
        question=query, role=role, top_k=top_k, retrieval_only=False
    )

    # Re-sélection pour fournir la liste des chunks utilisés
    entities, _ = extract_entities_keywords(query)
    base_raw = retrieve_general_chunks(query, role=role, top_k=max(64, top_k * 4))
    base_std = [enhance_chunk(to_standard_chunk(c)) for c in (base_raw or [])]
    ranked = score_sort_and_rerank(base_std, question=query, entities=entities, feedback_data=None, top_k=max(64, top_k * 4))
    selected = select_final_chunks(ranked, top_k=int(top_k), min_score=_min_score_from_env(0.0), diversify=True)

    sources = []
    for c in selected:
        src = c.get("source") or c.get("metadata", {}).get("source", "")
        if src and src not in sources:
            sources.append(src)

    return answer, sources, selected, summary

# compléter __all__ si ces symboles existent
if callable(semantic_search_two_step):
    __all__.append("semantic_search_two_step")
if callable(run_two_step):
    __all__.append("run_two_step")

