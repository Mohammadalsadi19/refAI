import json
import os
import re
import sys
import time
from collections import defaultdict
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_FILE = "referee_dataset.json"
EMB_CACHE_FILE = "case_embeddings_cache.json"
CLASSIFIER_FILE = "incident_classifier.pkl"
PREDICTION_LOG_FILE = "last_prediction.json"

CLASSIFIER_CONFIDENCE_THRESHOLD = 0.60
PRECEDENT_SCORE_THRESHOLD = 0.55
MODEL_NAME = "deepseek/deepseek-chat"

RAG_INDEX_FILE = "knowledge/faiss.index"
RAG_CHUNKS_FILE = "knowledge/chunks.json"
RAG_META_FILE = "knowledge/metadata.json"
RAG_TOP_K = 3

text_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================================
# RAG LAYER — retrieves the actual IFAB Laws of the Game text that
# supports a decision. Built from knowledge/pdf/IFAB_Laws_2025.pdf via
# scripts/extract_ifab_text.py -> chunk_ifab_text.py -> build_ifab_faiss.py.
#
# This is retrieval for EXPLAINABILITY only, exactly like precedent
# matching: it never feeds back into apply_law() and never changes the
# decision. It answers "which rulebook passage backs this up?", not
# "what should the decision be?".
# ==========================================================
def build_rag_keywords(incident_type, features, decision):
    """Pulls the exact IFAB terminology already present in this case's own
    features/decision — used to boost retrieval toward matching law text
    instead of relying on dense similarity alone."""
    terms = [incident_type.replace("_", " ")]
    for v in features.values():
        if isinstance(v, str):
            terms.append(v.replace("_", " "))
    for key in ("card", "restart"):
        v = decision.get(key)
        if isinstance(v, str) and v not in ("none",):
            terms.append(v.replace("_", " "))
    if decision.get("card") == "red":
        terms += ["serious foul play", "violent conduct", "sending-off"]
    if decision.get("card") == "yellow":
        terms += ["caution", "careless"]
    return list(dict.fromkeys(terms))  # de-duplicate, keep order


class IFABRetriever:
    def __init__(self, index_path=RAG_INDEX_FILE, chunks_path=RAG_CHUNKS_FILE,
                 meta_path=RAG_META_FILE):
        import faiss
        self.faiss = faiss
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_text, top_k=RAG_TOP_K, keyword_terms=None, candidate_pool=15):
        query_vec = text_model.encode(query_text).astype("float32")
        query_vec = np.expand_dims(query_vec, axis=0)
        self.faiss.normalize_L2(query_vec)  # must match the index's normalization
        similarities, indices = self.index.search(query_vec, candidate_pool)

        candidates = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                candidates.append({
                    "rank": rank + 1,
                    "similarity": float(similarities[0][rank]),
                    "text": self.chunks[idx]["text"],
                    "source": self.metadata[idx]["file"],
                })

        # Hybrid re-rank: pure dense retrieval with a general-purpose
        # embedding model often under-matches narrative scenario text
        # against dense legal/glossary text (register mismatch). Since
        # we already know the exact feature values and decision from the
        # rule engine, boost chunks that literally contain that
        # terminology (e.g. "excessive force", "DOGSO", "reckless").
        if keyword_terms:
            terms = [t.lower() for t in keyword_terms if t]
            for c in candidates:
                text_lower = c["text"].lower()
                hits = sum(1 for t in terms if t in text_lower)
                c["keyword_hits"] = hits
                c["combined_score"] = c["similarity"] + 0.15 * hits
            candidates.sort(key=lambda c: c["combined_score"], reverse=True)
        else:
            for c in candidates:
                c["keyword_hits"] = 0
                c["combined_score"] = c["similarity"]

        return candidates[:top_k]


_rag_retriever = None
_rag_load_attempted = False


def get_rag_retriever():
    """Lazily loads the RAG index once. Returns None (with a one-time
    warning) if faiss or the knowledge files aren't available, so the
    rest of the pipeline keeps working without it."""
    global _rag_retriever, _rag_load_attempted
    if _rag_retriever is not None:
        return _rag_retriever
    if _rag_load_attempted:
        return None
    _rag_load_attempted = True
    try:
        _rag_retriever = IFABRetriever()
        return _rag_retriever
    except Exception as e:
        print(f"⚠️  RAG knowledge base unavailable ({e}). "
              f"Continuing without law-text citations.")
        return None

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==========================
# OpenRouter client (lazy — only required for interactive mode,
# NOT for `evaluate`, so evaluation never needs an API key)
# ==========================
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY environment variable. "
                "Set it before running interactive mode.\n"
                "PowerShell: $env:OPENROUTER_API_KEY=\"your-key\"\n"
                "cmd:        set OPENROUTER_API_KEY=your-key"
            )
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


# ==========================================================
# Per incident_type feature schema — rebuilt from the FULL
# value inventory actually present in referee_dataset_v1.json
# ==========================================================
SCHEMAS = {
    "handball": {
        "player_role": ["attacker", "defender", "midfielder", "goalkeeper"],
        "deliberate": "bool",
        "arm_position": ["unnatural", "controlled", "natural", "protective", "extended"],
        "inside_penalty_area": "bool",
        "ball_speed": ["slow", "medium", "fast"],
        "distance_to_ball": ["short", "medium", "long"],
        "deflection": "bool",
        "goal_scoring_opportunity": "bool",
    },
    "tackle": {
        "player_role": ["attacker", "defender", "midfielder", "goalkeeper", "striker"],
        "tackle_type": ["challenge", "dive", "from_behind", "lunge", "push",
                        "shoulder", "slide", "standing", "two_footed"],
        "contact_with_ball": "bool",
        "contact_with_opponent": "bool",
        "force": ["careless", "normal", "reckless", "excessive"],
        "location": ["penalty_area", "midfield", "outside"],
        "endangering_safety": "bool",
    },
    "offside": {
        "position": ["beyond_last_defender", "level", "behind_ball"],
        "ball_played": "bool",
        "interfering_with_play": "bool",
        "interfering_with_opponent": "bool",
        "goal_scoring_opportunity": "bool",
        "restart_type": ["goal_kick", "throw_in", "corner_kick", "none"],
        "deflection_from_defender": "bool",
        "deliberate_play_by_defender": "bool",
    },
    "yellow_card": {
        "player_role": ["attacker", "defender", "midfielder", "goalkeeper"],
        "foul_type": ["dissent", "holding", "persistent_infringement", "reckless_tackle",
                      "simulation", "time_wasting", "unsporting_behavior"],
        "location": ["penalty_area", "midfield", "touchline"],
        "attack_promise": "bool",
        "goal_scoring_opportunity": "bool",
        "deliberate": "bool",
    },
    "red_card": {
        "player_role": ["attacker", "defender", "midfielder", "goalkeeper", "striker"],
        "foul_type": ["abusive_language", "deliberate_handball", "dogso_foul",
                      "serious_foul_play", "spitting", "violent_conduct"],
        "location": ["penalty_area", "midfield", "outside_penalty_area"],
        "dogso": "bool",
        "deliberate": "bool",
    },
    "substitution": {
        "player_role": ["attacker", "defender", "midfielder", "substitute", "team"],
        "substitution_type": ["early_entry", "exceed_limit", "half_time",
                              "illegal_entry", "illegal_exit", "normal", "refusal"],
        "time": ["active_play", "goal_celebration", "half_time", "stoppage"],
        "referee_permission": "bool",
    },
    "goal": {
        "ball_crossed_line": "bool",
        "between_posts": "bool",
        "under_crossbar": "bool",
        "no_foul": "bool",
        "own_goal": "bool",
        "referee_contact": "bool",
        "goalkeeper_contact": "bool",
        "goal_scoring_opportunity": "bool",
    },
    "cancelled_goal": {
        "ball_crossed_line": "bool",
        "handball": "bool",
        "deliberate": "bool",
        "offside": "bool",
        "interfering_with_play": "bool",
        "foul_on_goalkeeper": "bool",
        "play_stopped": "bool",
        "foul_in_attack": "bool",
    },
    "var": {
        "review_type": ["goal", "penalty", "red_card"],
        "possible_offence": ["offside", "handball", "foul", "location_of_foul",
                              "mistaken_identity", "serious_foul_play",
                              "reckless_not_excessive", "simulation",
                              "ball_not_crossed_line", "foul_in_attack"],
        "var_initiated": "bool",
    },
}
INCIDENT_TYPES = list(SCHEMAS.keys())


# ==========================================================
# RULE ENGINE — pure IFAB law logic, no ML, fully explainable.
# Every branch below was checked against the full ground-truth
# dataset (see evaluate_rule_engine) to align with real patterns:
# excessive force -> red, reckless -> yellow, careless -> foul only,
# free-kick/throw-in/corner-kick/goal-kick always cancels offside, etc.
# ==========================================================
def handball_law(f):
    role = f.get("player_role")
    deliberate = f.get("deliberate")
    arm_position = f.get("arm_position")
    inside = f.get("inside_penalty_area")
    deflection = f.get("deflection")
    gso = f.get("goal_scoring_opportunity")

    unnatural = {"unnatural", "extended", "controlled"}
    is_offense = bool(deliberate) or (arm_position in unnatural and not deflection)

    if not is_offense:
        return {"foul": False, "card": "none", "restart": "play_on", "var_review": False}

    restart = "penalty" if inside else "free_kick"
    card = "none"
    if role in ("defender", "goalkeeper") and gso:
        card = "red"
    elif role == "attacker" and arm_position == "controlled":
        card = "yellow"

    return {"foul": True, "card": card, "restart": restart, "var_review": True}


def tackle_law(f):
    contact_ball = f.get("contact_with_ball")
    contact_opp = f.get("contact_with_opponent")
    force = f.get("force")
    restart = "penalty" if f.get("location") == "penalty_area" else "free_kick"

    if force == "excessive":
        return {"foul": True, "card": "red", "restart": restart, "var_review": True}
    if force == "reckless":
        return {"foul": True, "card": "yellow", "restart": restart, "var_review": False}
    if force == "careless":
        return {"foul": True, "card": "none", "restart": restart, "var_review": False}
    if force == "normal":
        if contact_ball and not contact_opp:
            return {"foul": False, "card": "none", "restart": "play_on", "var_review": False}
        return {"foul": True, "card": "none", "restart": restart, "var_review": False}

    # Unrecognized/missing force value: fall back on ball vs. opponent contact
    if contact_ball and not contact_opp:
        return {"foul": False, "card": "none", "restart": "play_on", "var_review": False}
    return {"foul": True, "card": "none", "restart": restart, "var_review": False}


def offside_law(f):
    # Receiving the ball directly from a goal kick, throw-in, or corner kick
    # cancels offside entirely (Law 11).
    if f.get("restart_type") in ("goal_kick", "throw_in", "corner_kick"):
        return {"foul": False, "card": "none", "restart": "play_on", "var_review": False}

    # Deliberate play by a defender resets the offside phase.
    if f.get("deliberate_play_by_defender"):
        return {"foul": False, "card": "none", "restart": "play_on", "var_review": False}

    is_offside_position = f.get("position") == "beyond_last_defender"
    interfering_play = f.get("interfering_with_play")
    interfering_opp = f.get("interfering_with_opponent")
    ball_played = f.get("ball_played")

    # Interfering with an opponent is an offense on its own; interfering
    # with play additionally requires having actually played the ball.
    is_offense = is_offside_position and (
        (interfering_play and bool(ball_played)) or bool(interfering_opp)
    )

    var_review = bool(f.get("goal_scoring_opportunity")) or bool(f.get("deflection_from_defender"))

    if is_offense:
        return {"foul": True, "card": "none", "restart": "indirect_free_kick", "var_review": var_review}
    return {"foul": False, "card": "none", "restart": "play_on", "var_review": False}


NON_CONTACT_YELLOW_TYPES = {"time_wasting", "dissent", "simulation"}
CONTACT_YELLOW_TYPES = {"holding", "persistent_infringement", "reckless_tackle", "unsporting_behavior"}
SERIOUS_RED_TYPES = {"violent_conduct", "serious_foul_play", "deliberate_handball", "dogso_foul"}
NON_CONTACT_RED_TYPES = {"abusive_language", "spitting"}


def card_offense_law(f):
    """Generic disciplinary fouls (yellow_card / red_card dataset categories)."""
    foul_type = f.get("foul_type")
    dogso = f.get("dogso")
    location = f.get("location")
    attack_promise = f.get("attack_promise")

    restart = "penalty" if location == "penalty_area" else "free_kick"

    # Off-the-ball misconduct: no free kick, play continues from where it was
    if foul_type in NON_CONTACT_RED_TYPES:
        return {"foul": False, "card": "red", "restart": "play_on", "var_review": False}

    if dogso or foul_type in SERIOUS_RED_TYPES:
        return {"foul": True, "card": "red", "restart": restart, "var_review": True}

    if foul_type in NON_CONTACT_YELLOW_TYPES:
        return {"foul": False, "card": "yellow", "restart": "play_on", "var_review": False}

    if foul_type in CONTACT_YELLOW_TYPES or attack_promise:
        return {"foul": True, "card": "yellow", "restart": restart, "var_review": False}

    return {"foul": True, "card": "none", "restart": restart, "var_review": False}


def goal_law(f):
    """Covers both 'goal' and 'cancelled_goal' incident types."""

    def default_true(key):
        # These fields are frequently absent when not relevant to the
        # scenario — absence should mean "assume valid", not "assume false".
        v = f.get(key)
        return True if v is None else v

    crossed = default_true("ball_crossed_line")
    between_posts = default_true("between_posts")
    under_crossbar = default_true("under_crossbar")
    handball = f.get("handball")
    deliberate = f.get("deliberate")
    offside = f.get("offside")
    interfering = f.get("interfering_with_play")
    foul_on_gk = f.get("foul_on_goalkeeper")
    play_stopped = f.get("play_stopped")
    foul_in_attack = f.get("foul_in_attack")

    if not crossed or not between_posts or not under_crossbar:
        return {"goal_awarded": False, "restart": "goal_kick"}
    if play_stopped:
        return {"goal_awarded": False, "restart": "free_kick"}
    if handball and deliberate:
        return {"goal_awarded": False, "restart": "free_kick"}
    if offside and interfering:
        return {"goal_awarded": False, "restart": "indirect_free_kick"}
    if foul_on_gk or foul_in_attack:
        return {"goal_awarded": False, "restart": "free_kick"}

    # own_goal / referee_contact / goalkeeper_contact do not cancel a goal
    return {"goal_awarded": True, "restart": "kick_off"}


def var_law(f):
    review_type = f.get("review_type")
    offence = f.get("possible_offence")

    if review_type == "goal":
        if offence == "offside":
            return {"goal_awarded": False, "restart": "indirect_free_kick"}
        if offence in ("handball", "foul_in_attack"):
            return {"goal_awarded": False, "restart": "free_kick"}
        return {"goal_awarded": True, "restart": "kick_off"}

    if review_type == "penalty":
        if offence == "simulation":
            return {"penalty_cancelled": True, "restart": "indirect_free_kick"}
        return {"penalty_awarded": True, "restart": "penalty"}

    if review_type == "red_card":
        if offence == "mistaken_identity":
            return {"card_corrected": True, "restart": "free_kick"}
        if offence == "reckless_not_excessive":
            return {"card_downgraded": True, "restart": "free_kick"}
        return {"card_upheld": True, "restart": "free_kick"}

    return {"result": "undetermined"}


def substitution_law(f):
    sub_type = f.get("substitution_type")
    permission = f.get("referee_permission")
    time = f.get("time")

    if sub_type == "normal":
        return {"foul": False, "card": "none", "restart": "play_on"}
    if sub_type == "illegal_entry":
        if time == "goal_celebration":
            return {"foul": False, "card": "yellow", "restart": "kick_off"}
        return {"foul": True, "card": "yellow", "restart": "indirect_free_kick"}
    if sub_type == "early_entry":
        return {"foul": True, "card": "yellow", "restart": "indirect_free_kick"}
    if sub_type == "half_time":
        return {"foul": False, "card": "none", "restart": "kick_off"}
    if sub_type == "illegal_exit":
        return {"foul": False, "card": "yellow", "restart": "play_on"}
    if sub_type == "exceed_limit":
        return {"foul": False, "card": "none", "restart": "play_on"}
    if sub_type == "refusal":
        return {"foul": False, "card": "yellow", "restart": "play_on"}

    if permission:
        return {"foul": False, "card": "none", "restart": "play_on"}
    return {"foul": True, "card": "yellow", "restart": "indirect_free_kick"}


LAW_DISPATCH = {
    "handball": handball_law,
    "tackle": tackle_law,
    "offside": offside_law,
    "yellow_card": card_offense_law,
    "red_card": card_offense_law,
    "goal": goal_law,
    "cancelled_goal": goal_law,
    "var": var_law,
    "substitution": substitution_law,
}


def apply_law(incident_type, features):
    fn = LAW_DISPATCH.get(incident_type)
    if fn is None:
        return {"error": f"unknown_incident_type:{incident_type}"}
    return fn(features)


# ==========================================================
# EXPLANATION GENERATOR
# ==========================================================
def explanation_generator(incident_type, features, decision):
    if incident_type == "handball":
        if decision.get("foul"):
            base = "Handball offense: "
            base += "deliberate contact" if features.get("deliberate") else \
                    f"arm in a {features.get('arm_position')} position making the body bigger"
            loc = "inside the penalty area (penalty)" if decision.get("restart") == "penalty" else "outside the area (free kick)"
            base += f", {loc}."
            if decision.get("card") == "red":
                base += " Red card: denied an obvious goal-scoring opportunity."
            elif decision.get("card") == "yellow":
                base += " Yellow card: gained an unfair advantage with a controlled arm."
            return base
        return "No handball offense: contact was accidental with the arm in a natural position."

    if incident_type == "tackle":
        if not decision.get("foul"):
            return "Fair challenge: the ball was played cleanly with no dangerous contact."
        if decision.get("card") == "red":
            return "Excessive force endangering the opponent's safety — red card."
        if decision.get("card") == "yellow":
            return "Reckless challenge, unnecessary force used — cautioned."
        return "Careless foul, direct free kick awarded, no card warranted."

    if incident_type == "offside":
        if decision.get("foul"):
            return "Offside: player was in an offside position and interfered with play or an opponent."
        return "No offside: not interfering, not beyond the last defender, or the phase was reset."

    if incident_type in ("yellow_card", "red_card"):
        if decision.get("card") == "red":
            return "Red card: serious foul play, violent conduct, or denial of an obvious goal-scoring opportunity."
        if decision.get("card") == "yellow":
            return "Yellow card: reckless or tactical foul, or misconduct stopping a promising attack."
        return "Foul committed, no disciplinary card warranted."

    if incident_type in ("goal", "cancelled_goal"):
        if decision.get("goal_awarded"):
            return "Goal awarded: ball fully crossed the line with no infringement."
        return "Goal disallowed due to an offense before or during the goal (handball, offside, foul, or stoppage)."

    if incident_type == "var":
        return "VAR review outcome based on the reviewed incident."

    if incident_type == "substitution":
        if decision.get("foul"):
            return "Improper substitution procedure without referee permission."
        return "Substitution handled correctly under the current procedure."

    return "Decision generated from extracted features."


# ==========================================================
# ML LAYER 1: incident-type classifier (LogisticRegression)
# ==========================================================
def get_case_embedding_cache():
    if os.path.exists(EMB_CACHE_FILE):
        with open(EMB_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_case_embedding_cache(cache):
    with open(EMB_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_embeddings(dataset):
    """Returns (X, y, emb_cache) — encodes+caches every case's embedding."""
    emb_cache = get_case_embedding_cache()
    changed = False

    X, y = [], []
    for case in dataset:
        cid = str(case["id"])
        if cid in emb_cache:
            emb = emb_cache[cid]
        else:
            text = case["scenario"] + " " + case.get("explanation", "")
            emb = text_model.encode(text).tolist()
            emb_cache[cid] = emb
            changed = True
        X.append(emb)
        y.append(case["incident_type"])

    if changed:
        save_case_embedding_cache(emb_cache)

    return X, y, emb_cache


def make_classifier_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ])


def train_or_load_classifier(dataset):
    X, y, emb_cache = build_embeddings(dataset)

    if os.path.exists(CLASSIFIER_FILE):
        clf = joblib.load(CLASSIFIER_FILE)
    else:
        clf = make_classifier_pipeline()
        clf.fit(X, y)
        joblib.dump(clf, CLASSIFIER_FILE)

    return clf, emb_cache


def classify_incident_type_ml(clf, embedding):
    prediction = clf.predict([embedding])[0]
    probabilities = clf.predict_proba([embedding])[0]
    confidence = float(np.max(probabilities))
    return prediction, confidence


def nearest_neighbour_fallback(user_embedding, dataset, emb_cache):
    best_sim, best_type = -1.0, None
    for case in dataset:
        emb = emb_cache[str(case["id"])]
        sim = cosine_similarity([user_embedding], [emb])[0][0]
        if sim > best_sim:
            best_sim, best_type = sim, case["incident_type"]
    return best_type


# ==========================================================
# ML LAYER 2: LLM feature extractor (per incident_type schema)
# ==========================================================
def ask_model(prompt):
    client = get_client()
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            error = str(e)
            if "429" in error or "503" in error:
                wait = (2 ** attempt) + 3
                print(f"\nModel busy... Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise e
    return "{}"


def clean_json(text):
    try:
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception:
        print("⚠️  WARNING: Model returned invalid JSON, using safe defaults.")
        print("Raw response was:", text[:200])
        return {}


def build_extraction_prompt(incident_type, scenario_text):
    schema = SCHEMAS[incident_type]
    lines = []
    for key, allowed in schema.items():
        if allowed == "bool":
            lines.append(f'"{key}": true/false')
        else:
            lines.append(f'"{key}": "{"/".join(allowed)}"')
    schema_str = "{\n  " + ",\n  ".join(lines) + "\n}"
    return f"""Analyze this football incident and extract features.

Return ONLY JSON matching this exact schema, no extra text:

{schema_str}

Incident: {scenario_text}"""


def validate_features(incident_type, features, case_id=None):
    schema = SCHEMAS[incident_type]
    clean = {}
    for key, allowed in schema.items():
        val = features.get(key)
        if allowed == "bool":
            clean[key] = val if isinstance(val, bool) else None
            if clean[key] is None and case_id:
                print(f"⚠️  Case {case_id}: '{key}' missing/invalid bool -> None")
        else:
            v = str(val).strip().lower() if val is not None else None
            if v not in allowed:
                if case_id:
                    print(f"⚠️  Case {case_id}: '{key}'='{val}' not in {allowed} -> None")
                v = None
            clean[key] = v
    return clean


def extract_features(scenario_text, incident_type):
    prompt = build_extraction_prompt(incident_type, scenario_text)
    raw = ask_model(prompt)
    features = clean_json(raw)
    validated = validate_features(incident_type, features)

    if all(v is None for v in validated.values()):
        print("⚠️  Fallback triggered: extraction returned nothing usable, using safe defaults")
        schema = SCHEMAS[incident_type]
        validated = {k: (False if v == "bool" else None) for k, v in schema.items()}

    missing = sum(v is None for v in validated.values())
    quality = 1 - (missing / len(validated))

    if quality < 0.70:
        print(f"Warning: feature extraction quality is low ({round(quality * 100, 2)}%)")

    return validated


# ==========================================================
# Feature similarity for precedent matching (weighted, type-scoped)
# ==========================================================
FEATURE_WEIGHTS = {
    "player_role": 1,
    "location": 3,
    "inside_penalty_area": 4,
    "deliberate": 5,
    "dogso": 5,
    "force": 4,
    "arm_position": 4,
    "goal_scoring_opportunity": 4,
    "contact_with_ball": 3,
    "contact_with_opponent": 3,
    "ball_played": 4,
    "interfering_with_play": 5,
    "interfering_with_opponent": 4,
    "offside": 5,
    "review_type": 3,
    "possible_offence": 4,
}


def feature_similarity(user_features, case_features):
    score, total = 0, 0
    for key in set(user_features) | set(case_features):
        uv, cv = user_features.get(key), case_features.get(key)
        if uv is None or cv is None:
            continue
        weight = FEATURE_WEIGHTS.get(key, 1)
        total += weight
        if uv == cv:
            score += weight
    return score / total if total else 0.0


def build_case_features_cache(dataset):
    """Ground-truth features already exist in the dataset — reuse them
    directly instead of paying for LLM re-extraction on known cases."""
    return {str(c["id"]): c["features"] for c in dataset}


def find_precedents(user_embedding, incident_type, user_features,
                     dataset, emb_cache, gt_features_cache, top_n=5):
    candidates = []
    for case in dataset:
        if case["incident_type"] != incident_type:
            continue

        emb = emb_cache[str(case["id"])]
        text_score = cosine_similarity([user_embedding], [emb])[0][0]
        feature_score = feature_similarity(user_features, gt_features_cache[str(case["id"])])
        final_score = (0.45 * text_score) + (0.55 * feature_score)

        candidates.append((case, final_score, text_score, feature_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def matched_feature_keys(user_features, case_features):
    return [k for k in user_features if user_features.get(k) == case_features.get(k)]


def save_prediction_log(incident_type, confidence, features, decision, explanation):
    prediction = {
        "incident_type": incident_type,
        "classifier_confidence": confidence,
        "features": features,
        "decision": decision,
        "explanation": explanation,
    }
    with open(PREDICTION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)


# ==========================================================
# EVALUATION — the missing piece for a defensible graduation project
# ==========================================================
def evaluate_rule_engine(dataset):
    """
    Runs the rule engine on every case's GROUND-TRUTH features (no ML
    involved) and compares its output to the dataset's own recorded
    decision, field by field. This answers: "is the law logic itself
    correct?", independent of any extraction or classification error.
    """
    per_type_total = defaultdict(int)
    per_type_full_match = defaultdict(int)
    per_type_field_scores = defaultdict(list)
    mismatches = []

    for case in dataset:
        t = case["incident_type"]
        predicted = apply_law(t, case["features"])
        expected = case["decision"]

        keys = set(expected.keys())
        matched = sum(1 for k in keys if predicted.get(k) == expected.get(k))
        field_score = matched / len(keys) if keys else 0.0
        full_match = field_score == 1.0

        per_type_total[t] += 1
        per_type_full_match[t] += int(full_match)
        per_type_field_scores[t].append(field_score)

        if not full_match:
            mismatches.append({
                "id": case["id"], "incident_type": t,
                "features": case["features"], "predicted": predicted, "expected": expected,
            })

    print("\n===== RULE ENGINE EVALUATION (vs. dataset ground truth) =====")
    print(f"{'Type':<16}{'Exact match':<14}{'Field-level match'}")
    total_full, total_n = 0, 0
    for t in per_type_total:
        n = per_type_total[t]
        full = per_type_full_match[t]
        avg_field = sum(per_type_field_scores[t]) / n
        print(f"{t:<16}{full}/{n:<10}{round(avg_field * 100, 1)}%")
        total_full += full
        total_n += n

    print(f"\nOVERALL exact match: {total_full}/{total_n} ({round(total_full / total_n * 100, 1)}%)")
    print(f"Mismatched cases: {len(mismatches)} (see below)")

    if mismatches:
        print("\n--- Sample mismatches (first 5) ---")
        for m in mismatches[:5]:
            print(f"[{m['incident_type']}] id={m['id']}")
            print("  features:", m["features"])
            print("  predicted:", m["predicted"])
            print("  expected :", m["expected"])

    return total_full / total_n if total_n else 0.0


def evaluate_classifier(dataset, n_splits=5):
    """
    Proper stratified k-fold cross-validation for the incident-type
    classifier — NOT trained-and-tested-on-the-same-data. This gives an
    honest accuracy number instead of an overfit one.
    """
    X, y, _ = build_embeddings(dataset)
    X = np.array(X)
    y = np.array(y)

    class_counts = defaultdict(int)
    for label in y:
        class_counts[label] += 1
    min_class_count = min(class_counts.values())
    splits = min(n_splits, min_class_count) if min_class_count >= 2 else 2

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    clf = make_classifier_pipeline()

    y_pred = cross_val_predict(clf, X, y, cv=skf)

    print(f"\n===== INCIDENT-TYPE CLASSIFIER EVALUATION ({splits}-fold CV) =====")
    print(classification_report(y, y_pred, zero_division=0))

    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("Labels:", labels)
    print(cm)

    accuracy = float((y_pred == y).mean())
    print(f"\nOverall cross-validated accuracy: {round(accuracy * 100, 2)}%")
    return accuracy


def run_evaluation():
    print("Running full evaluation (no API key required)...\n")
    rule_acc = evaluate_rule_engine(data)
    clf_acc = evaluate_classifier(data)
    print("\n===== SUMMARY =====")
    print(f"Rule engine exact-match accuracy vs ground truth: {round(rule_acc * 100, 1)}%")
    print(f"Incident-type classifier cross-validated accuracy: {round(clf_acc * 100, 1)}%")


# ==========================================================
# MAIN — interactive mode
# ==========================================================
def main():
    print("\nTraining/loading incident-type classifier...")
    clf, emb_cache = train_or_load_classifier(data)
    gt_features_cache = build_case_features_cache(data)
    print("Ready ✅")

    user_text = input("\nEnter case: ").strip()
    user_embedding = text_model.encode(user_text)

    # --- ML layer 1: classify incident type ---
    incident_type, confidence = classify_incident_type_ml(clf, user_embedding.tolist())

    if confidence < CLASSIFIER_CONFIDENCE_THRESHOLD:
        print("\nClassifier confidence is low — falling back to semantic nearest neighbour...")
        incident_type = nearest_neighbour_fallback(user_embedding, data, emb_cache)

    print(f"\nIncident Type: {incident_type}")
    print(f"Classifier Confidence: {round(confidence * 100, 2)}%")

    # --- ML layer 2: extract structured features via LLM ---
    user_features = extract_features(user_text, incident_type)
    print("\nExtracted Features:")
    print(json.dumps(user_features, indent=2))

    # --- Rule engine: official, explainable decision ---
    decision = apply_law(incident_type, user_features)
    if "error" in decision:
        print("Rule Engine Error:", decision["error"])
        return

    explanation = explanation_generator(incident_type, user_features, decision)

    print("\n===== REFEREE DECISION (Rule Engine) =====")
    print(json.dumps(decision, indent=2))
    print("\nExplanation:", explanation)

    # --- RAG: retrieve the actual IFAB Law text backing this decision ---
    retriever = get_rag_retriever()
    if retriever is not None:
        rag_keywords = build_rag_keywords(incident_type, user_features, decision)
        law_hits = retriever.search(user_text + " " + explanation, top_k=RAG_TOP_K,
                                     keyword_terms=rag_keywords)
        if law_hits:
            print("\n===== LEGAL BASIS (IFAB Laws of the Game) =====")
            for hit in law_hits:
                print(f"[{hit['source']} | similarity={round(hit['similarity'], 3)} "
                      f"| keyword_hits={hit['keyword_hits']}]")
                snippet = hit["text"][:400].strip()
                print(snippet + ("..." if len(hit["text"]) > 400 else ""))
                print("-" * 60)

    # --- Precedent matching (explainability only, never overrides the decision) ---
    combined_embedding = text_model.encode(user_text + " " + explanation)
    precedents = find_precedents(combined_embedding, incident_type, user_features,
                                  data, emb_cache, gt_features_cache)
    precedents = [p for p in precedents if p[1] >= PRECEDENT_SCORE_THRESHOLD]

    if precedents:
        print(f"\nBest Similar Case Confidence: {round(precedents[0][1] * 100, 2)}%")
        print("\n===== CLOSEST PRECEDENTS =====")
        for case, final_score, text_score, feature_score in precedents:
            print("=" * 60)
            print("CASE:", case["id"])
            print(case["scenario"])
            print("\nDecision:", case["decision"])
            print(f"\nFinal Score: {round(final_score * 100, 2)}%")
            print(f"Text Similarity: {round(text_score * 100, 2)}%")
            print(f"Feature Similarity: {round(feature_score * 100, 2)}%")
            print("\nMatched Features:", matched_feature_keys(user_features, case["features"]))
    else:
        print("\nNo precedent cases met the similarity threshold.")

    save_prediction_log(incident_type, confidence, user_features, decision, explanation)
    print(f"\nSaved prediction to {PREDICTION_LOG_FILE} ✅")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        run_evaluation()
    else:
        main()