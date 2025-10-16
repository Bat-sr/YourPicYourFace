import cv2
import numpy as np
from deepface import DeepFace

# Config
SAMPLE_COUNT = 5
SIMILARITY_THRESHOLD = 0.85

# In-memory DB (list of np.ndarray embeddings)
known_faces: list[np.ndarray] = []

# --------------------------
# Utilities
# --------------------------
def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n

def rotate_image(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def _is_point(p) -> bool:
    return isinstance(p, (tuple, list, np.ndarray)) and len(p) == 2

def _has_required_landmarks(facial_area: dict) -> bool:
    le = facial_area.get("left_eye")
    re = facial_area.get("right_eye")
    nose = facial_area.get("nose")
    mouth = facial_area.get("mouth")
    ml = facial_area.get("mouth_left")
    mr = facial_area.get("mouth_right")
    has_eyes = _is_point(le) and _is_point(re)
    has_nose = _is_point(nose)
    has_mouth = _is_point(mouth) or (_is_point(ml) and _is_point(mr))
    return has_eyes and has_nose and has_mouth

# --------------------------
# Detection + Embedding
# --------------------------
def _detect_one_face(frame_bgr: np.ndarray):
    try:
        faces = DeepFace.extract_faces(
            frame_bgr, detector_backend="retinaface", enforce_detection=False, align=True
        )
        f0 = faces[0]
        fa = f0.get("facial_area", {})
        print(f"[INFO] detector=retinaface area={fa}")
        if not _has_required_landmarks(fa):
            print("[INFO] no valid (missing eyes/nose/mouth) returning None")
            return None
        return f0
    except Exception as e:
        print(f"[WARN] retinaface detection failed: {e}")

def get_embedding_from_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Detect one face, apply small rotations, average FaceNet embeddings, L2-normalize."""
    face_data = _detect_one_face(frame_bgr)
    if face_data is None:
        print("[WARN] No face detected by any backend.")
        return None

    face_rgb = (face_data["face"] * 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

    embs = []
    for angle in (-5, 0, 5):
        rot = rotate_image(face_bgr, angle)
        rep = DeepFace.represent(
            rot, model_name="Facenet", detector_backend="skip", enforce_detection=False
        )
        embs.append(np.asarray(rep[0]["embedding"], dtype=np.float64))

    return _norm(np.mean(embs, axis=0))

# --------------------------
# Similarity
# --------------------------
def compare_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity on L2-normalized vectors (higher is better)."""
    if a is None or b is None:
        return 0.0
    a = _norm(a)
    b = _norm(b)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

# --------------------------
# Matching / Enrollment
# --------------------------
def register_or_get_known_face(frame_bgr: np.ndarray | None = None):
    """
    If frame_bgr provided, compute one embedding from it; else sample camera frames and average.
    Returns an embedding (np.ndarray) or None. Maintains a simple known_faces cache.
    """
    global known_faces

    def _match_or_register(avg_emb: np.ndarray) -> np.ndarray:
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-12)
        avg_emb = avg_emb.copy()

        best_idx = -1
        best_sim = -1.0
        for idx, known in enumerate(known_faces):
            sim = compare_embeddings(avg_emb, known)
            if sim > best_sim:
                best_sim, best_idx = sim, idx
        print("Best sim = ", best_sim)

        if best_idx >= 0 and best_sim >= SIMILARITY_THRESHOLD:
            print("Existing face Match")
            return known_faces[best_idx]

        known_faces.append(avg_emb.copy())
        return avg_emb

    # Path 1: provided frame
    if frame_bgr is not None:
        emb = get_embedding_from_frame(frame_bgr)
        if emb is None:
            print("[WARN] No valid embedding from provided frame.")
            return None
        return _match_or_register(emb)

    # Path 2: camera capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return None

    # warm-up
    for _ in range(5):
        cap.read()

    collected = []
    tries = SAMPLE_COUNT * 3
    while len(collected) < SAMPLE_COUNT and tries > 0:
        ok, frame = cap.read()
        if not ok:
            tries -= 1
            continue
        emb = get_embedding_from_frame(frame)
        if emb is not None:
            collected.append(emb.copy())
        tries -= 1
        cv2.waitKey(80)

    cap.release()

    if not collected:
        print("[WARN] No valid face samples collected from camera.")
        return None

    avg_emb = np.mean(np.stack(collected, axis=0), axis=0)
    return _match_or_register(avg_emb)
