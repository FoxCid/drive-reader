# app_backend_20250713.py — backend Flask robuste (RRF/rerank prêts côté moteur)

import os
import json
import ctypes
import signal
import psutil
import traceback
from typing import Any, Dict, List, Union

from flask import Flask, request, jsonify
from two_step.auto_restore import record_feedback_event
from two_step.api import semantic_search_phase2_only
from prompt_templates import get_role_prompt
from flask_cors import CORS
from two_step.auto_restore import get_auto_restore_candidate

# --- DLL GTK (logs vus à l’écran) ---
os.environ["PATH"] = r"C:\GTK3\bin;" + os.environ.get("PATH", "")
for dll in [
    "libgobject-2.0-0.dll", "libglib-2.0-0.dll", "libgio-2.0-0.dll",
    "libpango-1.0-0.dll", "libpangocairo-1.0-0.dll", "libcairo-2.dll",
    "libgdk_pixbuf-2.0-0.dll", "libfontconfig-1.dll",
    "libfreetype-6.dll", "libharfbuzz-0.dll"
]:
    try:
        ctypes.CDLL(os.path.join(r"C:\GTK3\bin", dll))
        print(f"✅ Chargé depuis backend : {dll}")
    except Exception as e:
        print(f"❌ Erreur DLL ({dll}): {e}")




# --- Imports projet ---
from pdf_generator_pdfkit import generate_structured_pdf_from_answer
from structured_output_generator import generate_structured_output_from_answer, group_chunks_by_theme

from feedback_learning_20250713 import save_feedback
from feedback_engine import reload_validated_feedback, load_validated_feedback
from build_feedback_index_by_role import build_feedback_index_for_role

# moteur v1
from semantic_search_20250713 import (
    semantic_search_20250713 as semantic_search,
    load_faiss_indexes,
    hash_question
)



# moteur v2 (two-step, RRF/rerank/boosts côté moteur)
from two_step.auto_restore import get_auto_restore_candidate, record_feedback_event, hash_answer, evidence_signature
 

print("📄 BACKEND FILE:", os.path.abspath(__file__))


SEMANTIC_INDEX_VERSION = "v2025-07-25+pdf"
ROLES_VALIDES = {"rh", "appel_offres", "general"}
selected_role = None

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

DEBUG_RAG = _env_bool("DEBUG_RAG", False)

def check_and_kill_port(port=5050):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"🔕 Port {port} occupé par PID {proc.info['pid']}. Suppression…")
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

if check_and_kill_port(5050):
    print("✅ Port libéré")
else:
    print("ℹ️ Port déjà libre")

app = Flask(__name__)
CORS(app)


# --- Health endpoints ---
from datetime import datetime
import platform

STARTED_AT = datetime.utcnow().isoformat() + "Z"

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "drive-reader-backend",
        "started_at": STARTED_AT,
        "now": datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "pid": os.getpid(),
    }, 200

def _readiness_checks():
    errors = []

    # 1) Import API phase2
    try:
        from two_step.api import semantic_search_phase2_only  # noqa
    except Exception as e:
        errors.append(f"two_step.api import: {e!r}")

    # 2) Dossier feedback accessible
    try:
        os.makedirs("data/feedback", exist_ok=True)
        test_path = os.path.join("data", "feedback", ".healthz.touch")
        open(test_path, "a").close()
        os.remove(test_path)
    except Exception as e:
        errors.append(f"feedback dir: {e!r}")

    return errors

@app.get("/readyz")
def readyz():
    errs = _readiness_checks()
    if errs:
        return {"ok": False, "errors": errs}, 503
    return {"ok": True}, 200





# ---------- Utils backend ----------

def _doc_to_public_dict(doc) -> Dict[str, Any]:
    """Mappe un objet chunk/doc → dict JSON-safe pour l’UI."""
    meta = getattr(doc, "metadata", {}) or {}
    return {
        "text": getattr(doc, "page_content", ""),
        "score": float(meta.get("total_score", 0.0) or 0.0),
        "source": meta.get("source", ""),
        "entity_score": float(meta.get("entity_match_score", meta.get("entity_score", 0.0)) or 0.0),
        "feedback_penalty": float(meta.get("penalty_feedback", meta.get("feedback_penalty", 0.0)) or 0.0),
        "bonus_metadata": float(meta.get("bonus_metadata", 0.0) or 0.0),
        "faiss_score": float(meta.get("faiss_score", meta.get("score", 0.0)) or 0.0),
    }

def _safe_json_loads(maybe_json: Union[str, Dict, List, None]) -> Union[Dict, List, None]:
    """Tolère une sortie texte non strictement JSON (préfixée par logs / texte) et renvoie {} si échec."""
    if maybe_json is None:
        return None
    if isinstance(maybe_json, (dict, list)):
        return maybe_json
    if not isinstance(maybe_json, str):
        return None
    s = maybe_json.strip()
    if not s:
        return None
    # heuristique: trouver le premier { ou [
    start_brace = s.find("{")
    start_brack = s.find("[")
    starts = [i for i in (start_brace, start_brack) if i >= 0]
    if starts:
        first = min(starts)
        if first > 0:
            s = s[first:]
    try:
        return json.loads(s)
    except Exception as e:
        if DEBUG_RAG:
            print(f"❌ _safe_json_loads: {e}")
            print("---- RAW BEGIN ----")
            print(maybe_json[:1000])
            print("---- RAW END ----")
        return {}

def _read_threshold_env() -> float:
    try:
        return float(os.getenv("CHUNK_MIN_SCORE", "0.0"))
    except Exception:
        return 0.0

# ---------- Routes ----------

@app.route("/version", methods=["GET"])
def version():
    return jsonify({
        "status": "ok",
        "version": SEMANTIC_INDEX_VERSION,
        "message": "Backend opérationnel"
    })

@app.route("/initialize", methods=["POST"])
def initialize():
    """Enregistre le rôle côté backend (appelé par l’UI avant /chat[_v2])."""
    global selected_role
    data = request.get_json(silent=True) or {}
    selected_role = data.get("role")
    if not selected_role:
        return jsonify({"error": "Aucun rôle fourni"}), 400
    if selected_role not in ROLES_VALIDES:
        return jsonify({"error": f"❌ Rôle invalide : {selected_role}"}), 400
    return jsonify({"message": "Rôle enregistré avec succès"})

@app.route("/check_index", methods=["GET"])
def check_index():
    try:
        vectorstore_docs, vectorstore_context = load_faiss_indexes(role=selected_role)
        if not vectorstore_docs or not vectorstore_context:
            return jsonify({"message": "❌ Index introuvable"}), 500
        return jsonify({"message": "✅ Index chargé avec succès"})
    except Exception as e:
        return jsonify({"message": f"❌ Erreur : {str(e)}"}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        save_feedback(data)

        try:
            # Enregistrement dans l’historique d’auto-restauration
            record_feedback_event(
                question=data.get("question"),
                role=data.get("role"),
                score=int(data.get("score", 0)),
                answer_text=data.get("answer"),
                answer_hash=data.get("answer_hash"),
                evidence=data.get("evidence"),
                sources=data.get("sources", []),
                meta={"mode": data.get("mode")}
            )
        except Exception as e:
            print(f"⚠️ Auto-restore non enregistré : {e}")

        return jsonify({"message": "✅ Feedback enregistré"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# -------- v1: /chat (mono-étape historique) --------

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()
        role = (data.get("role") or "general").strip()
        top_k = int(data.get("top_k") or 8)

        # --- Fast-path auto-restore : 3×5/5 consécutifs + preuves inchangées ---
        from two_step.auto_restore import get_auto_restore_candidate  # import local pour éviter cycles
        cand = None
        try:
            cand = get_auto_restore_candidate(question, role)
        except Exception:
            cand = None

        if cand:
            # les chunks/sources viennent du stockage feedback; on les renvoie tels quels
            return jsonify({
                "answer": cand.get("answer_text", ""),
                "sources": cand.get("sources", []),
                "chunks":  cand.get("chunks", []),
                "debug":   {"auto_restore": True, "count": cand.get("count", 0)},
                "version": SEMANTIC_INDEX_VERSION + "+restored"
            })

        # ---------- Phase 2 seule : recherche sur l’index général ----------
        answer, sources, top_docs, debug = semantic_search_phase2_only(
            query=question, role=role, top_k=top_k
        )


        # ---------- Prompt de rôle (métier) ----------
        role_prompt = get_role_prompt(role)

        # ---------- Contexte textuel à partir des chunks ----------
        def _as_text(d):
            if isinstance(d, dict):
                return d.get("page_content") or d.get("text") or ""
            return getattr(d, "page_content", "") or ""

        context_text = "\n\n".join(_as_text(d) for d in top_docs[:top_k])

        # ---------- Construction du prompt final ----------
        final_prompt = f"""{'{'}role_prompt{'}'}

Question:
{'{'}question{'}'}

Contexte (extraits):
{'{'}context_text{'}'}

Consignes:
- Réponds en français, de manière concise et sourcée si possible.
- Si l'information n'apparaît pas dans le contexte, dis-le explicitement.
"""

        # ---------- Génération (remplacer par ton appel LLM si nécessaire) ----------
        answer_text = ""
        try:
            from generation import generate_text  # optionnel
            answer_text = generate_text(final_prompt)
        except Exception:
            # Fallback extractif pour ne pas planter si le LLM n'est pas branché
            answer_text = (answer or "").strip()
            if not answer_text:
                answer_text = "Résumé extractif basé sur les meilleurs extraits :\n\n" + context_text[:1800]

        # ---------- Normalisation des chunks en dicts ----------
        

        chunks = [_doc_to_public_dict(d) for d in top_docs]

        return jsonify({
            "answer": answer_text,
            "sources": sources,
            "chunks": chunks,
            "debug": debug,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/chat_v2', methods=['POST'])
def chat_v2():
    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()
        role     = (data.get("role") or "general").strip()
        top_k    = int(data.get("top_k") or 8)

        res = semantic_search_phase2_only(query=question, role=role, top_k=top_k)

        # --- compat tuples / dicts ---
        extra = None
        if isinstance(res, dict):
            answer   = res.get("answer") or res.get("answer_text", "")
            sources  = res.get("sources", [])
            top_docs = res.get("top_docs", res.get("chunks", []))
            debug    = res.get("debug", {})
            # tout le reste dans extra
            extra = {k:v for k,v in res.items() if k not in {"answer","answer_text","sources","top_docs","chunks","debug"}}
        elif isinstance(res, (list, tuple)):
            if len(res) >= 4:
                answer, sources, top_docs, debug, *rest = res
                extra = {"extra": rest} if rest else None
            else:
                # tolérance si 3 éléments (ex: pas de debug)
                if len(res) == 3:
                    answer, sources, top_docs = res
                    debug = {}
                else:
                    raise ValueError(f"Unexpected return arity from semantic_search_phase2_only: {len(res)}")
        else:
            raise TypeError(f"Unexpected return type: {type(res)}")

        # normalisation chunks -> dict pour l’UI
        

        chunks = [_doc_to_public_dict(d) for d in (top_docs or [])]

        payload = {
            "answer": answer or "",
            "sources": sources or [],
            "chunks": chunks,
            "debug": debug or {},
        }
        if extra:
            payload["extra"] = extra

        return jsonify(payload)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500




if __name__ == "__main__":
    import os
    PORT = int(os.getenv("PORT", "5050"))  # ton script parle de 5050
    HOST = os.getenv("HOST", "127.0.0.1")

    # Si tu as une fonction qui “libère le port” (kill du PID), garde-la
    try:
        # free_port_if_needed(PORT)   # <- seulement si tu l’as vraiment
        pass
    except Exception as e:
        print(f"⚠️ Impossible de libérer le port {PORT}: {e}")

    # LANCE le serveur (c’est le point manquant)
    app.run(host=HOST, port=PORT)

