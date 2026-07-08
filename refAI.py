from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import cv2
import time
import re
import os
from openai import OpenAI

# ==========================
# Client (OpenRouter)
# ==========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

text_model = SentenceTransformer('all-MiniLM-L6-v2')

with open("referee_dataset_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings_store = []

model_name = "deepseek/deepseek-chat"

FEATURE_SCHEMA = """{
"contact": true,
"location": "penalty_area/outside",
"body_part": "hand/leg/head/body",
"intensity": "low/medium/high",
"ball_played": true,
"deliberate": true,
"attacker": false,
"defender": true
}"""

# ==========================
# Ask model (retry)
# ==========================
def ask_model(prompt):
    retries = 5
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
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

# ==========================
# JSON cleaner + validator
# ==========================
def clean_json(text):
    try:
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except Exception:
        print("⚠️  WARNING: Model returned invalid JSON, using safe defaults.")
        print("Raw response was:", text[:200])
        return {"error": "invalid_json"}

def validate_features(features, case_id=None):
    def norm(v):
        if v is None:
            return None
        return str(v).strip().lower()

    location_synonyms = {
        "penalty area": "penalty_area",
        "penalty_area": "penalty_area",
        "box": "penalty_area",
        "penalty box": "penalty_area",
        "in the box": "penalty_area",
        "18-yard box": "penalty_area",
        "outside": "outside",
        "outside the box": "outside",
        "midfield": "outside",
    }
    body_synonyms = {
        "hand": "hand",
        "arm": "hand",
        "handball": "hand",
        "leg": "leg",
        "foot": "leg",
        "head": "head",
        "body": "body",
        "chest": "body",
        "shoulder": "body",
    }

    valid_location = {"penalty_area", "outside"}
    valid_body = {"hand", "leg", "head", "body"}
    valid_intensity = {"low", "medium", "high"}

    raw_location = norm(features.get("location"))
    raw_body = norm(features.get("body_part"))
    raw_intensity = norm(features.get("intensity"))

    location = location_synonyms.get(raw_location, raw_location)
    body_part = body_synonyms.get(raw_body, raw_body)

    if location not in valid_location:
        location = "outside"

    if body_part not in valid_body:
        body_part = "body"

    if raw_intensity not in valid_intensity:
        raw_intensity = "medium"

    features["location"] = location
    features["body_part"] = body_part
    features["intensity"] = raw_intensity
    features["contact"] = features.get("contact") if isinstance(features.get("contact"), bool) else None
    features["deliberate"] = features.get("deliberate") if isinstance(features.get("deliberate"), bool) else None
    features["ball_played"] = features.get("ball_played") if isinstance(features.get("ball_played"), bool) else None

    return features

# ==========================
# Feature Similarity
# ==========================
def feature_similarity(user, case):
    weights = {
        "deliberate": 3,
        "body_part": 2,
        "location": 2,
        "ball_played": 2,
        "contact": 1,
        "intensity": 1,
    }

    case_features = case["features"]
    score = 0
    total = 0

    for key, w in weights.items():
        user_val = user.get(key)
        case_val = case_features.get(key)
        if user_val is None or case_val is None:
            continue
        total += w
        if user_val == case_val:
            score += w

    return score / total if total > 0 else 0.0

# ==========================
# Rule Engine
# ==========================
def rule_engine(features):
    location = features.get("location")
    body_part = features.get("body_part")
    intensity = features.get("intensity")
    contact = features.get("contact")
    deliberate = features.get("deliberate")
    ball_played = features.get("ball_played")

    if ball_played is True and intensity != "high":
        return "play_on"

    if body_part == "hand":
        is_offense = deliberate is True
        if is_offense and location == "penalty_area":
            return "penalty"
        elif is_offense:
            return "free_kick"
        else:
            return "play_on"

    elif body_part in ["leg", "head", "body"]:
        is_foul = contact is True and intensity == "high"
        if is_foul and location == "penalty_area":
            return "penalty"
        elif is_foul:
            return "foul"
        else:
            return "play_on"

    return "undecided"

# ==========================
# Process Dataset + Cache
# ==========================
CACHE_FILE = "features_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

print("\nProcessing Dataset...")

for case in data:
    text = case.get("scenario", "") + " " + case.get("explanation", "")
    text_embedding = text_model.encode(text)
    case_id = str(case["id"])

    if case_id in cache:
        structured_features = cache[case_id]
    else:
        prompt = f"""
Analyze football incident.

Return ONLY JSON, no extra text:

{FEATURE_SCHEMA}

Text:

{case.get("scenario", "")}
"""
        response_text = ask_model(prompt)
        structured_features = clean_json(response_text)
        structured_features = validate_features(structured_features, case_id=case_id)
        cache[case_id] = structured_features

    decision_rule = rule_engine(structured_features)

    embeddings_store.append({
        "id": case["id"],
        "scenario": case.get("scenario", ""),
        "text_embedding": text_embedding.tolist(),
        "features": structured_features,
        "decision": decision_rule,
        "media": case.get("media", {})
    })

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print("Dataset Ready ✅")

# ==========================
# USER INPUT
# ==========================
user_text = input("\nEnter case: ")

response_text = ask_model(f"""
Analyze football incident.

Return ONLY JSON, no extra text:

{FEATURE_SCHEMA}

Text:

{user_text}
""")

user_features = clean_json(response_text)
user_features = validate_features(user_features, case_id="user_input")
cache["last_user"] = user_features

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

decision_rule = rule_engine(user_features)

print("\nUser Text Analysis:")
print(json.dumps(user_features, indent=2))
print("\nRule Engine Decision:", decision_rule)

# ==========================
# User Embedding
# ==========================
user_embedding = text_model.encode(user_text)

embeddings_store.append({
    "id": f"user_{len(embeddings_store) + 1}",
    "text_embedding": user_embedding.tolist(),
    "features": user_features,
    "decision": decision_rule,
    "media": {}
})

# ==========================
# Similarity Search
# ==========================
all_embeddings = [x["text_embedding"] for x in embeddings_store[:-1]]
similarities = cosine_similarity([user_embedding], all_embeddings)[0]

scores = []
for i, case in enumerate(embeddings_store[:-1]):
    text_score = similarities[i]
    feat_score = feature_similarity(user_features, case)
    final_score = (0.4 * text_score) + (0.6 * feat_score)
    scores.append((i, final_score))

scores.sort(key=lambda x: x[1], reverse=True)
top5 = scores[:5]

best_index = top5[0][0]
score = top5[0][1]
