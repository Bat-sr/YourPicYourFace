from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import base64
import face_scan as fc # Assumes you have your face_scan.py file
import os
import time
import struct
import hashlib

# --- Basic Flask App Setup ---
app = Flask(__name__)
socketio = SocketIO(app)

# --- Ensure Directories Exist ---
GALLERY_DIR = "encrypted_gallery"
os.makedirs(GALLERY_DIR, exist_ok=True)

# -------------------------------------------------------------------
# --- PRE-EXISTING LOGIC FUNCTIONS ---
# -------------------------------------------------------------------

def encrypt_image_pixels(image_pixel_array, face_embedding):
    print("--- Encrypting image using custom multiplication logic ---")
    original_shape = image_pixel_array.shape
    image_vector = image_pixel_array.flatten().astype(np.float64)

    key_vector = face_embedding
    image_len = len(image_vector)
    key_len = len(key_vector)

    print(key_vector)

    if key_len == 0:
        return image_pixel_array

    num_repeats = (image_len // key_len) + 1
    tiled_key = np.tile(key_vector, num_repeats)
    
    resized_key = tiled_key[:image_len]

    cipher_vector = image_vector * 100*resized_key

    encrypted_array = cipher_vector.reshape(original_shape)
    return encrypted_array

def decrypt_image_pixels(encrypted_pixel_array, face_embedding):
    print("--- Decrypting image by reversing multiplication ---")
    original_shape = encrypted_pixel_array.shape
    image_vector = encrypted_pixel_array.flatten().astype(np.float64)
    key_vector = face_embedding

    image_len = len(image_vector)
    key_len = len(key_vector)

    if key_len == 0:
        return encrypted_pixel_array
    
    num_repeats = (image_len // key_len) + 1
    tiled_key = np.tile(key_vector, num_repeats)
    resized_key = tiled_key[:image_len]

    with np.errstate(divide='ignore', invalid='ignore'):
        decrypted_vector = np.divide(image_vector, 100*resized_key)
        decrypted_vector[np.isinf(decrypted_vector)] = 0 # Replace infinity with 0
        decrypted_vector = np.nan_to_num(decrypted_vector) # Replace NaN with 0

    # Clip values to be in the valid range for an image [0, 255] and convert type
    decrypted_array = np.clip(decrypted_vector, 0, 255)
    return decrypted_array.reshape(original_shape).astype(np.uint8)


# -------------------------------------------------------------------
# --- CORE APPLICATION LOGIC ---
# -------------------------------------------------------------------

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected!')

@socketio.on('get_gallery')
def handle_get_gallery():
    """Sends all currently encrypted images to the client."""
    images = []
    for filename in sorted(os.listdir(GALLERY_DIR)):
        if filename.endswith(".bin"):
            file_path = os.path.join(GALLERY_DIR, filename)
            with open(file_path, "rb") as f:
                shape_bytes = f.read(12)
                if len(shape_bytes) < 12: continue
                shape = struct.unpack('>III', shape_bytes)
                encrypted_data = f.read()
                encrypted_array = np.frombuffer(encrypted_data, dtype=np.float64).reshape(shape)
                
                # Encode to PNG for UI display, clipping values to be displayable
                display_array = np.mod(encrypted_array, 256)
                _, buffer = cv2.imencode('.png', display_array.astype(np.uint8))
                b64_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
                images.append({'id': filename, 'data': b64_data})
                
    emit('gallery_data', {'images': images})

@socketio.on('upload_and_encrypt')
def handle_upload(data):
    """Handles the upload, encryption, and saving as a raw binary file."""
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

    encrypted_array = encrypt_image_pixels(image_pixel_array, embedding)
    
    original_shape = encrypted_array.shape
    shape_header = struct.pack('>III', *original_shape)
    encrypted_data_bytes = encrypted_array.tobytes()
    final_data_to_save = shape_header + encrypted_data_bytes

    timestamp = int(time.time() * 1000)
    base_filename = f"image_{timestamp}"
    encrypted_path = os.path.join(GALLERY_DIR, f"{base_filename}.bin")
    embedding_path = os.path.join(GALLERY_DIR, f"{base_filename}.npy")

    with open(encrypted_path, "wb") as f:
        f.write(final_data_to_save)
    np.save(embedding_path, embedding)

    print(f"Processed and saved {base_filename}")
    emit('encryption_status', {'status': 'success', 'message': 'Image processed successfully!'})
    handle_get_gallery()

@socketio.on('decrypt_with_face')
def handle_decrypt(data):
    """Handles decryption by reading raw binary files."""
    face_image_b64 = data['face_image'].split(',')[1]
    face_bytes = base64.b64decode(face_image_b64)
    face_frame = cv2.imdecode(np.frombuffer(face_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    live_embedding = fc.register_or_get_known_face(face_frame)
    if live_embedding is None:
        emit('decryption_result', {'status': 'error', 'message': 'Could not detect face for decryption.'})
        return

    processed_images = []
    
    for filename in sorted(os.listdir(GALLERY_DIR)):
        if filename.endswith(".bin"):
            base_filename = filename.replace(".bin", "")
            embedding_path = os.path.join(GALLERY_DIR, f"{base_filename}.npy")
            encrypted_path = os.path.join(GALLERY_DIR, filename)

            if not os.path.exists(embedding_path): continue

            print(f"MATCH for {filename} - Decrypting.")
            with open(encrypted_path, "rb") as f:
                shape_bytes = f.read(12)
                if len(shape_bytes) < 12: continue
                shape = struct.unpack('>III', shape_bytes)
                encrypted_data_bytes = f.read()
                encrypted_pixel_array = np.frombuffer(encrypted_data_bytes, dtype=np.float64).reshape(shape)

            decrypted_array = decrypt_image_pixels(encrypted_pixel_array, live_embedding)
            _, decrypted_buffer = cv2.imencode('.png', decrypted_array)
            b64_data = base64.b64encode(decrypted_buffer.tobytes()).decode('utf-8')
            processed_images.append({'id': filename, 'data': b64_data})

    emit('decryption_result', {'status': 'success', 'images': processed_images})

@socketio.on('clear_gallery')
def handle_clear_gallery():
    """Deletes all files from the encrypted gallery directory."""
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