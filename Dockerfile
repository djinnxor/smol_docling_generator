# ── Labeler – Dockerfile ───────────────────────────────────────────────────
# Runs the Flask-based labeler tool so that annotators with only Docker
# installed can use it without setting up a Python environment locally.
# ───────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System packages required by the labeler's Python dependencies:
#   poppler-utils  – pdf2image needs pdfinfo / pdftoppm
#   libgl1 libglib2.0-0 libsm6 libxext6 – OpenCV runtime libraries (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the labeler-relevant packages (skip heavy training deps).
# Versions are pinned for reproducible builds.
# easyocr is the offline OCR engine used by the labeler.
# gunicorn is used as the production WSGI server (multi-worker, multi-user safe).
RUN pip install --no-cache-dir \
        "flask>=3.0,<4" \
        "pillow>=10.0,<12" \
        "pdf2image>=1.17,<2" \
        "docling>=2.0,<3" \
        "opencv-python-headless>=4.8,<5" \
        "easyocr>=1.7,<2" \
        "numpy>=1.24,<3" \
        "requests>=2.31,<3" \
        "werkzeug>=3.0,<4" \
        "scikit-learn>=1.3,<2" \
        "gunicorn>=21.0,<23"

# Copy application source
COPY labeler.py google_ocr.py table_detection.py prepare_dataset.py converter.py ./
COPY templates/ ./templates/

# Persistent data (uploads, annotations, xml_annotations, SQLite DB) is
# mounted as a volume at runtime so files survive container restarts.
RUN mkdir -p labeler_data/uploads \
              labeler_data/annotations \
              labeler_data/xml_annotations

# Run as a non-root user for security
RUN useradd -m labeler && chown -R labeler:labeler /app
USER labeler

EXPOSE 5000

# Healthcheck so orchestrators can verify the app is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/', timeout=5)" || exit 1

# gunicorn: 4 worker processes, 2 threads each → handles concurrent users.
# Adjust WEB_CONCURRENCY at runtime (e.g. -e WEB_CONCURRENCY=2) if needed.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 --workers ${WEB_CONCURRENCY:-4} --threads 2 --timeout 120 labeler:app"]
