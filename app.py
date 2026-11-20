"""Flask interface for an enrolled face-recognition gallery."""

import argparse
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from recognize import FaceRecognizer


def create_app(gallery=Path("embeddings.pt"), threshold=0.9, engine=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app.config["FACE_ENGINE"] = engine
    lock = Lock()

    @app.get("/")
    def index():
        return """<!doctype html><title>Celebrity Face Recognition</title>
        <h1>Celebrity Face Recognition</h1>
        <form action="/recognize" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Recognize</button></form>"""

    @app.post("/recognize")
    def recognize():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Provide an image file."), 400
        try:
            with Image.open(upload.stream) as source:
                image = source.convert("RGB")
            with lock:
                if app.config["FACE_ENGINE"] is None:
                    recognizer = FaceRecognizer()
                    recognizer.load_gallery(gallery)
                    app.config["FACE_ENGINE"] = recognizer
                records = app.config["FACE_ENGINE"].recognize(image, threshold)
            return jsonify(faces=records)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            return jsonify(error=str(error)), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, default=Path("embeddings.pt"))
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if not args.gallery.is_file():
        parser.error("Create the gallery with recognize.py enroll first.")
    create_app(args.gallery).run(host="127.0.0.1", port=args.port)
