"""
VocalIsolator — Flask Backend con lalal.ai API
"""

import os
import io
import time
import uuid
import tempfile
import traceback
import requests

from pydub import AudioSegment
from flask import Flask, request, jsonify, send_file, send_from_directory

app = Flask(__name__, static_folder=".")

LALAL_API_KEY  = os.environ.get("LALAL_API_KEY", "")
LALAL_UPLOAD   = "https://www.lalal.ai/api/upload/"
LALAL_SPLIT    = "https://www.lalal.ai/api/split/"
LALAL_CHECK    = "https://www.lalal.ai/api/check/"

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def lalal_upload(file_bytes: bytes, filename: str) -> str:
    """Sube el archivo a lalal.ai y devuelve el file_id."""
    headers = {"X-License-Key": LALAL_API_KEY}
    files   = {"file": (filename, io.BytesIO(file_bytes), "audio/mpeg")}
    resp = requests.post(LALAL_UPLOAD, headers=headers, files=files, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Upload error: {data.get('error', data)}")
    return data["id"]


def lalal_split(file_id: str) -> None:
    """Inicia la separación de voces en lalal.ai."""
    headers = {"X-License-Key": LALAL_API_KEY, "Content-Type": "application/json"}
    payload = {
        "id": file_id,
        "stem": "vocals",          # vocals vs accompaniment
        "splitter": "phoenix",     # modelo más reciente y preciso
    }
    resp = requests.post(LALAL_SPLIT, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Split error: {data.get('error', data)}")


def lalal_wait(file_id: str, max_wait: int = 300) -> dict:
    """Espera hasta que el procesamiento termine y devuelve las URLs."""
    headers = {"X-License-Key": LALAL_API_KEY, "Content-Type": "application/json"}
    payload = {"id": file_id}
    deadline = time.time() + max_wait

    while time.time() < deadline:
        resp = requests.post(LALAL_CHECK, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            raise RuntimeError(f"Check error: {data.get('error', data)}")

        task = data.get("task", {})
        state = task.get("state", "")

        if state == "success":
            return task   # contiene stem_track y back_track URLs
        elif state in ("error", "cancelled"):
            raise RuntimeError(f"Task failed with state: {state}")

        # Todavía procesando — esperar un poco
        time.sleep(4)

    raise TimeoutError("lalal.ai tardó demasiado en procesar.")


def download_audio(url: str) -> AudioSegment:
    """Descarga un audio desde una URL y lo devuelve como AudioSegment."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    return AudioSegment.from_file(buf)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/process", methods=["POST"])
def process():
    if not LALAL_API_KEY:
        return jsonify({"error": "LALAL_API_KEY no configurada en el servidor."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    file = request.files["file"]
    if not allowed(file.filename):
        return jsonify({"error": "Formato no soportado. Usá MP3, WAV, FLAC u OGG."}), 400

    try:
        start = float(request.form.get("start", 0))
        end   = float(request.form.get("end",   0))
    except ValueError:
        return jsonify({"error": "Los tiempos deben ser números."}), 400

    if end <= start:
        return jsonify({"error": "El tiempo de fin debe ser mayor al de inicio."}), 400

    try:
        # ── 1. Leer archivo completo ──
        file_bytes = file.read()
        audio_full = AudioSegment.from_file(io.BytesIO(file_bytes))
        duration_ms = len(audio_full)

        start_ms = max(0, min(int(start * 1000), duration_ms))
        end_ms   = max(start_ms + 100, min(int(end * 1000), duration_ms))

        # ── 2. Extraer solo el segmento y subirlo a lalal.ai ──
        segment = audio_full[start_ms:end_ms]
        seg_buf = io.BytesIO()
        segment.export(seg_buf, format="wav")
        seg_bytes = seg_buf.getvalue()

        print(f"[lalal] Subiendo segmento {start:.1f}s–{end:.1f}s ({len(seg_bytes)//1024} KB)…")
        file_id = lalal_upload(seg_bytes, "segment.wav")

        # ── 3. Iniciar separación ──
        print(f"[lalal] Iniciando split para file_id={file_id}…")
        lalal_split(file_id)

        # ── 4. Esperar resultado ──
        print("[lalal] Esperando resultado…")
        task = lalal_wait(file_id)

        vocals_url = task.get("stem_track")
        if not vocals_url:
            return jsonify({"error": "lalal.ai no devolvió la URL de voces."}), 500

        # ── 5. Descargar voces ──
        print(f"[lalal] Descargando voces desde {vocals_url}…")
        vocals = download_audio(vocals_url)

        # ── 6. Ajustar volumen ──
        orig_dbfs   = segment.dBFS if segment.dBFS > -100 else -20
        vocals_dbfs = vocals.dBFS  if vocals.dBFS  > -100 else -20
        vocals = vocals.apply_gain(orig_dbfs - vocals_dbfs)

        # ── 7. Ensamblar resultado final ──
        before = audio_full[:start_ms]
        after  = audio_full[end_ms:]
        result = before + vocals + after

        # ── 8. Exportar y devolver ──
        out_buf = io.BytesIO()
        result.export(out_buf, format="wav")
        out_buf.seek(0)

        return send_file(
            out_buf,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="resultado_vocal.wav"
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
