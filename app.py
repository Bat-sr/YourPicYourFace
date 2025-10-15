from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import base64
import face_scan as fc
import os
import time
import struct

# Config
SIMILARITY_THRESHOLD = 0.85       # face-embedding cosine gate before decrypt
VERIFY_COSINE_THRESHOLD = 0.995   # image signature cosine must be ≥ this
SIG_SIZE = (64, 64)               # downscale size for image signature

def _cos_to_l2_thr(c: float) -> float:
    c = float(max(-1.0, min(1.0, c)))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * c)))

VERIFY_L2_THRESHOLD = _cos_to_l2_thr(VERIFY_COSINE_THRESHOLD)

app = Flask(__name__)
socketio = SocketIO(app)

GALLERY_DIR = "encrypted_gallery"
os.makedirs(GALLERY_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Encryption / Decryption
# -------------------------------------------------------------------
def encrypt_image_pixels(image_pixel_array: np.ndarray, face_embedding: np.ndarray) -> np.ndarray:
    print("--- Encrypting image using custom multiplication logic ---")
    original_shape = image_pixel_array.shape
    image_vector = image_pixel_array.flatten().astype(np.float64)

    key_vector = face_embedding.astype(np.float64)
    key_len = len(key_vector)
    if key_len == 0:
        return image_pixel_array

    num_repeats = (len(image_vector) // key_len) + 1
    resized_key = np.tile(key_vector, num_repeats)[: len(image_vector)]
    cipher_vector = image_vector * (100.0 * resized_key)
    return cipher_vector.reshape(original_shape)

def decrypt_image_pixels(encrypted_pixel_array: np.ndarray, face_embedding: np.ndarray) -> np.ndarray:
    print("--- Decrypting image by reversing multiplication ---")
    original_shape = encrypted_pixel_array.shape
    image_vector = encrypted_pixel_array.flatten().astype(np.float64)
    key_vector = face_embedding.astype(np.float64)

    key_len = len(key_vector)
    if key_len == 0:
        return encrypted_pixel_array

    num_repeats = (len(image_vector) // key_len) + 1
    resized_key = np.tile(key_vector, num_repeats)[: len(image_vector)]

    with np.errstate(divide='ignore', invalid='ignore'):
        decrypted_vector = np.divide(image_vector, (100.0 * resized_key))
        decrypted_vector[np.isinf(decrypted_vector)] = 0
        decrypted_vector = np.nan_to_num(decrypted_vector)

    return np.clip(decrypted_vector, 0, 255).reshape(original_shape).astype(np.uint8)

# -------------------------------------------------------------------
# Image Signature + Verification
# -------------------------------------------------------------------
def _unit_signature(img_bgr: np.ndarray, size=SIG_SIZE) -> np.ndarray | None:
    """Return unit-normalized grayscale signature vector for similarity checks."""
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    vec = small.flatten()
    vec -= vec.mean()
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)

def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _images_match(sig_ref: np.ndarray | None, sig_dec: np.ndarray | None) -> bool:
    if sig_ref is None or sig_dec is None:
        return False
    d = _l2(sig_ref, sig_dec)
    print(f"[VERIFY] L2={d:.6f} (thr≤{VERIFY_L2_THRESHOLD:.6f})")
    return d <= VERIFY_L2_THRESHOLD

# -------------------------------------------------------------------
# Routes / Socket Handlers
# -------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected!')

@socketio.on('get_gallery')
def handle_get_gallery():
    images = []
    for filename in sorted(os.listdir(GALLERY_DIR)):
        if filename.endswith(".bin"):
            file_path = os.path.join(GALLERY_DIR, filename)
            with open(file_path, "rb") as f:
                shape_bytes = f.read(12)
                if len(shape_bytes) < 12:
                    continue
                shape = struct.unpack('>III', shape_bytes)
                encrypted_data = f.read()
                encrypted_array = np.frombuffer(encrypted_data, dtype=np.float64).reshape(shape)

                display_array = np.mod(encrypted_array, 256)
                _, buffer = cv2.imencode('.png', display_array.astype(np.uint8))
                b64_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
                images.append({'id': filename, 'data': b64_data})

    emit('gallery_data', {'images': images})

@socketio.on('upload_and_encrypt')
def handle_upload(data):
    face_image_b64 = data['face_image'].split(',')[1]
    upload_image_b64 = data['upload_image'].split(',')[1]

    face_bytes = base64.b64decode(face_image_b64)
    face_frame = cv2.imdecode(np.frombuffer(face_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    embedding = fc.register_or_get_known_face(face_frame)
    if embedding is None:
        emit('encryption_status', {'status': 'error', 'message': 'Could not detect face for key.'})
        return

    upload_bytes = base64.b64decode(upload_image_b64)
    image_pixel_array = cv2.imdecode(np.frombuffer(upload_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_pixel_array is None:
        emit('encryption_status', {'status': 'error', 'message': 'Could not decode uploaded image.'})
        return

    ref_sig = _unit_signature(image_pixel_array, SIG_SIZE)
    encrypted_array = encrypt_image_pixels(image_pixel_array, embedding)

    original_shape = encrypted_array.shape
    shape_header = struct.pack('>III', *original_shape)
    final_data_to_save = shape_header + encrypted_array.tobytes()

    timestamp = int(time.time() * 1000)
    base_filename = f"image_{timestamp}"
    encrypted_path = os.path.join(GALLERY_DIR, f"{base_filename}.bin")
    embedding_path = os.path.join(GALLERY_DIR, f"{base_filename}.npy")
    sig_path = os.path.join(GALLERY_DIR, f"{base_filename}.sig.npy")

    with open(encrypted_path, "wb") as f:
        f.write(final_data_to_save)
    np.save(embedding_path, embedding)
    if ref_sig is not None:
        np.save(sig_path, ref_sig.astype(np.float32))

    print(f"Processed and saved {base_filename}")
    emit('encryption_status', {'status': 'success', 'message': 'Image processed successfully!'})
    handle_get_gallery()

@socketio.on('decrypt_with_face')
def handle_decrypt(data):
    face_image_b64 = data['face_image'].split(',')[1]
    face_bytes = base64.b64decode(face_image_b64)
    face_frame = cv2.imdecode(np.frombuffer(face_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    live_embedding = fc.register_or_get_known_face(face_frame)
    if live_embedding is None:
        emit('decryption_result', {'status': 'error', 'message': 'Could not detect face for decryption.'})
        return

    processed_images = []
    blocked = 0
    face_blocked = 0

    for filename in sorted(os.listdir(GALLERY_DIR)):
        if not filename.endswith(".bin"):
            continue

        base_filename = filename[:-4]
        embedding_path = os.path.join(GALLERY_DIR, f"{base_filename}.npy")
        sig_path = os.path.join(GALLERY_DIR, f"{base_filename}.sig.npy")
        encrypted_path = os.path.join(GALLERY_DIR, filename)

        if not os.path.exists(embedding_path):
            print(f"[SKIP] No stored embedding for {filename}")
            continue

        stored_embedding = np.load(embedding_path)
        a = stored_embedding / (np.linalg.norm(stored_embedding) + 1e-12)
        b = live_embedding / (np.linalg.norm(live_embedding) + 1e-12)
        face_cos = float(np.dot(a, b))
        print(f"[FACE-GATE] {filename}: cosine={face_cos:.6f} (threshold={SIMILARITY_THRESHOLD:.6f})")

        if face_cos < SIMILARITY_THRESHOLD:
            face_blocked += 1
            continue

        with open(encrypted_path, "rb") as f:
            shape_bytes = f.read(12)
            if len(shape_bytes) < 12:
                continue
            shape = struct.unpack('>III', shape_bytes)
            encrypted_data_bytes = f.read()
            encrypted_pixel_array = np.frombuffer(encrypted_data_bytes, dtype=np.float64).reshape(shape)

        decrypted_array = decrypt_image_pixels(encrypted_pixel_array, live_embedding)

        if not os.path.exists(sig_path):
            print(f"[SKIP] No reference signature for {filename}; cannot verify image. Skipping.")
            blocked += 1
            continue

        ref_sig = np.load(sig_path)
        dec_sig = _unit_signature(decrypted_array, SIG_SIZE)

        if not _images_match(ref_sig, dec_sig):
            print(f"[BLOCK] Image verification failed for {filename} (mismatch).")
            blocked += 1
            continue

        print(f"[PASS] {filename} verified. Returning decrypted image.")
        _, decrypted_buffer = cv2.imencode('.png', decrypted_array)
        b64_data = base64.b64encode(decrypted_buffer.tobytes()).decode('utf-8')
        processed_images.append({'id': filename, 'data': b64_data})

    status_msg = f"Decryption finished. Returned={len(processed_images)}, face_blocked={face_blocked}, image_blocked={blocked}"
    emit('decryption_result', {'status': 'success', 'images': processed_images, 'message': status_msg})

@socketio.on('clear_gallery')
def handle_clear_gallery():
    print("--- Clearing the gallery ---")
    for filename in os.listdir(GALLERY_DIR):
        file_path = os.path.join(GALLERY_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    handle_get_gallery()

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000)