from pathlib import Path
import json
import math

TRACKS_DIR = Path("tracks")
FEATURES_DIR = Path("features")

# ----------------------------------------------------------------
# Only these fields are realistically derivable from bounding-box
# tracking alone. Everything else in RefAI.py's SCHEMAS (deliberate,
# force, arm_position, tackle_type, foul_type, dogso, ...) requires
# judgment about INTENT or SEVERITY that plain object detection cannot
# provide — that needs either a trained action-recognition/pose model,
# or manual/LLM-assisted labeling. Left as None on purpose, not a bug:
# it plugs straight into RefAI.py's SCHEMAS without breaking anything,
# since every law function already handles missing (None) fields.
# ----------------------------------------------------------------

# Tunable thresholds — calibrate against your actual frame resolution
# and FRAME_STEP once you have a few labeled clips to check against.
CONTACT_IOU_THRESHOLD = 0.02
DISTANCE_SHORT_PX = 60
DISTANCE_MEDIUM_PX = 150
BALL_SPEED_SLOW_PX = 8
BALL_SPEED_FAST_PX = 25

# Optional: set this to 4 (x, y) pixel points if your camera is static
# and you've calibrated the penalty area corners. Leave as None to skip
# location detection entirely (safer default than guessing wrong).
PENALTY_AREA_POLYGON = None


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def centroid_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def point_in_polygon(point, polygon):
    if polygon is None:
        return None
    x, y = point
    n = len(polygon)
    inside = False
    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2) and y <= max(py1, py2) and x <= max(px1, px2):
            if py1 != py2:
                xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
            else:
                xinters = px1
            if px1 == px2 or x <= xinters:
                inside = not inside
        px1, py1 = px2, py2
    return inside


def build_base_features(incident_type):
    """Returns every SCHEMA field for this incident_type set to None.
    Keep this list in sync with RefAI.py's SCHEMAS."""
    base = {
        "handball": ["player_role", "deliberate", "arm_position", "inside_penalty_area",
                     "ball_speed", "distance_to_ball", "deflection", "goal_scoring_opportunity"],
        "tackle": ["player_role", "tackle_type", "contact_with_ball", "contact_with_opponent",
                   "force", "location", "endangering_safety"],
        "offside": ["position", "ball_played", "interfering_with_play",
                    "interfering_with_opponent", "goal_scoring_opportunity",
                    "restart_type", "deflection_from_defender", "deliberate_play_by_defender"],
    }.get(incident_type, [])
    return {k: None for k in base}


def load_tracks(track_file):
    with open(track_file, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_track_id(tracks, class_name):
    groups = {}
    for t in tracks:
        if t["class_name"] != class_name:
            continue
        groups.setdefault(t["track_id"], []).append(t)
    for tid in groups:
        groups[tid].sort(key=lambda x: x["frame_id"])
    return groups


def pick_ball_track(ball_groups):
    """Picks the ball track_id with the most detections (most stable)."""
    if not ball_groups:
        return None
    return max(ball_groups.items(), key=lambda kv: len(kv[1]))[1]


def pick_primary_player(person_groups, ball_points):
    """Picks the player whose average distance to the ball is smallest —
    the one most likely involved in the incident."""
    if not person_groups or not ball_points:
        return None, None

    ball_by_frame = {p["frame_id"]: p["centroid"] for p in ball_points}
    best_id, best_avg = None, float("inf")

    for tid, points in person_groups.items():
        dists = [
            centroid_distance(p["centroid"], ball_by_frame[p["frame_id"]])
            for p in points if p["frame_id"] in ball_by_frame
        ]
        if not dists:
            continue
        avg = sum(dists) / len(dists)
        if avg < best_avg:
            best_avg, best_id = avg, tid

    return best_id, person_groups.get(best_id)


def compute_geometric_features(tracks, incident_type):
    features = build_base_features(incident_type)

    person_groups = group_by_track_id(tracks, "person")
    ball_groups = group_by_track_id(tracks, "ball")
    ball_points = pick_ball_track(ball_groups)

    if not ball_points:
        return features  # nothing usable without a ball track

    primary_id, primary_points = pick_primary_player(person_groups, ball_points)
    if not primary_points:
        return features

    ball_by_frame = {p["frame_id"]: p for p in ball_points}
    primary_by_frame = {p["frame_id"]: p for p in primary_points}
    shared_frames = sorted(set(ball_by_frame) & set(primary_by_frame))
    if not shared_frames:
        return features

    # --- contact_with_ball ---
    contact = any(
        bbox_iou(primary_by_frame[f]["bbox"], ball_by_frame[f]["bbox"]) > CONTACT_IOU_THRESHOLD
        for f in shared_frames
    )
    if "contact_with_ball" in features:
        features["contact_with_ball"] = contact

    # --- contact_with_opponent (any OTHER person overlapping the primary player) ---
    if "contact_with_opponent" in features:
        opponent_contact = False
        for tid, points in person_groups.items():
            if tid == primary_id:
                continue
            other_by_frame = {p["frame_id"]: p for p in points}
            for f in shared_frames:
                if f in other_by_frame and bbox_iou(
                        primary_by_frame[f]["bbox"], other_by_frame[f]["bbox"]) > CONTACT_IOU_THRESHOLD:
                    opponent_contact = True
                    break
            if opponent_contact:
                break
        features["contact_with_opponent"] = opponent_contact

    # --- distance_to_ball (bucketed pixel distance) ---
    if "distance_to_ball" in features:
        avg_dist = sum(
            centroid_distance(primary_by_frame[f]["centroid"], ball_by_frame[f]["centroid"])
            for f in shared_frames
        ) / len(shared_frames)
        if avg_dist <= DISTANCE_SHORT_PX:
            features["distance_to_ball"] = "short"
        elif avg_dist <= DISTANCE_MEDIUM_PX:
            features["distance_to_ball"] = "medium"
        else:
            features["distance_to_ball"] = "long"

    # --- ball_speed (bucketed pixel displacement per frame) ---
    if "ball_speed" in features and len(ball_points) >= 2:
        displacements = [
            centroid_distance(ball_points[i]["centroid"], ball_points[i - 1]["centroid"])
            for i in range(1, len(ball_points))
        ]
        avg_speed = sum(displacements) / len(displacements)
        if avg_speed <= BALL_SPEED_SLOW_PX:
            features["ball_speed"] = "slow"
        elif avg_speed <= BALL_SPEED_FAST_PX:
            features["ball_speed"] = "medium"
        else:
            features["ball_speed"] = "fast"

    # --- location (only if you've calibrated PENALTY_AREA_POLYGON) ---
    if "location" in features and PENALTY_AREA_POLYGON is not None:
        last_point = primary_by_frame[shared_frames[-1]]["centroid"]
        inside = point_in_polygon(last_point, PENALTY_AREA_POLYGON)
        features["location"] = "penalty_area" if inside else "outside"

    if "inside_penalty_area" in features and PENALTY_AREA_POLYGON is not None:
        last_point = primary_by_frame[shared_frames[-1]]["centroid"]
        features["inside_penalty_area"] = bool(point_in_polygon(last_point, PENALTY_AREA_POLYGON))

    return features


def main():
    if not TRACKS_DIR.exists():
        raise FileNotFoundError(f"{TRACKS_DIR} not found — run tracker.py first.")

    all_cases = []
    next_id = 1

    for category_dir in sorted(TRACKS_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        incident_type = category_dir.name

        for track_file in sorted(category_dir.glob("*.json")):
            video_name = track_file.stem
            tracks = load_tracks(track_file)
            features = compute_geometric_features(tracks, incident_type)

            case = {
                "id": next_id,
                "incident_type": incident_type,
                "video": video_name,
                "features": features,
            }
            next_id += 1

            out_path = FEATURES_DIR / incident_type / f"{video_name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(case, f, indent=2)

            all_cases.append(case)
            filled = sum(1 for v in features.values() if v is not None)
            print(f"{incident_type}/{video_name}: {filled}/{len(features)} fields derived from video")

    with open(FEATURES_DIR / "vision_features_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2)

    print(f"\nSaved {len(all_cases)} cases -> {FEATURES_DIR / 'vision_features_dataset.json'}")


if __name__ == "__main__":
    main()