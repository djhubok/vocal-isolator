"""
VocalIsolator — Flask Backend con lalal.ai API
"""

import os
import io
import time
import traceback
import requests

from pydub import AudioSegment
from flask import Flask, request, jsonify, send_file, send_from_directory

app = Flask(__name__, static_folder=".")

LALAL_API_KEY = os.environ.get("LALAL_API_KEY", "")
LALAL_UPLOAD  = "https://www.lalal.ai/api/upload/"
LALAL_SPLIT   = "https://www.lalal.ai/api/split/"
LALAL_CHECK   = "https://www.lalal.ai/api/check/"

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def auth_headers() -> dict:
    """Header de autenticación correcto para lalal.ai"""
    return {"Authorization": f"license {LALAL_API_KEY}"}


def lalal_upload(file_bytes: bytes, filename: str) -> str:
    """Sube el archivo a lalal.ai con Content-Disposition y devuelve el file_id."""
    headers = {
        **auth_headers(),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "audio/wav",
    }
    resp = requests.post(LALAL_UPLOAD, headers=headers, data=file_bytes, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Upload error: {data.get('error', data)}")
    return data["id"]


def lalal_split(file_id: str) -> None:
    """Inicia la separación usando form-urlencoded (formato correcto de lalal.ai)."""
    headers = auth_headers()
    # lalal.ai espera form-urlencoded con params como JSON string
    import json
    params = json.dumps([{"id": file_id, "stem": "vocals", "filter": 2}])
    resp = requests.post(
        LALAL_SPLIT,
        headers=headers,
        data={"params": params},
        timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Split error: {data.get('error', data)}")


def lalal_wait(file_id: str, max_wait: int = 300) -> dict:
    """Espera hasta que el procesamiento termine y devuelve el resultado."""
    headers  = auth_headers()
    deadline = time.time() + max_wait

    while time.time() < deadline:
        resp = requests.post(
            LALAL_CHECK,
            headers=headers,
            data={"id": file_id},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            raise RuntimeError(f"Check error: {data.get('error', data)}")

        result = data.get("result", {})
        file_result = result.get(file_id, {})

        if file_result.get("status") != "success":
            raise RuntimeError(f"File error: {file_result.get('error', file_result)}")

        task = file_result.get("task", {})
        split = file_result.get("split")

        if split:
            return split  # contiene stem_track y back_track

        state = task.get("state", "")
        if state == "error":
            raise RuntimeError(f"Task error: {task.get('error', 'unknown')}")
        elif state == "cancelled":
            raise RuntimeError("Task cancelled by lalal.ai")

        time.sleep(4)

    raise TimeoutError("lalal.ai tardó demasiado en procesar.")


def download_audio(url: str) -> AudioSegment:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return AudioSegment.from_file(io.BytesIO(resp.content))


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
        # 1. Leer y decodificar audio completo
        file_bytes  = file.read()
        audio_full  = AudioSegment.from_file(io.BytesIO(file_bytes))
        duration_ms = len(audio_full)

        start_ms = max(0, min(int(start * 1000), duration_ms))
        end_ms   = max(start_ms + 100, min(int(end * 1000), duration_ms))

        # 2. Extraer segmento
        segment   = audio_full[start_ms:end_ms]
        seg_buf   = io.BytesIO()
        segment.export(seg_buf, format="wav")
        seg_bytes = seg_buf.getvalue()

        # 3. Subir segmento a lalal.ai
        print(f"[lalal] Subiendo segmento {start:.1f}s–{end:.1f}s ({len(seg_bytes)//1024} KB)…")
        file_id = lalal_upload(seg_bytes, "segment.wav")
        print(f"[lalal] file_id={file_id}")

        # 4. Iniciar separación
        lalal_split(file_id)
        print("[lalal] Split iniciado, esperando…")

        # 5. Esperar resultado
        split_result = lalal_wait(file_id)
        vocals_url   = split_result.get("stem_track")
        if not vocals_url:
            return jsonify({"error": "lalal.ai no devolvió la URL de voces."}), 500

        # 6. Descargar voces
        print(f"[lalal] Descargando voces desde {vocals_url}…")
        vocals = download_audio(vocals_url)

        # 7. Ajustar volumen para que coincida con el original
        orig_dbfs   = segment.dBFS if segment.dBFS > -100 else -20
        vocals_dbfs = vocals.dBFS  if vocals.dBFS  > -100 else -20
        vocals      = vocals.apply_gain(orig_dbfs - vocals_dbfs)

        # 8. Ensamblar: antes + voces + después
        result  = audio_full[:start_ms] + vocals + audio_full[end_ms:]
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
