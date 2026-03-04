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
# rapidocr-onnxruntime is an optional OCR engine used by the labeler.
RUN pip install --no-cache-dir \
        "flask>=3.0,<4" \
        "pillow>=10.0,<12" \
        "pdf2image>=1.17,<2" \
        "docling>=2.0,<3" \
        "opencv-python-headless>=4.8,<5" \
        "rapidocr-onnxruntime>=1.3,<2" \
        "numpy>=1.24,<3" \
        "requests>=2.31,<3" \
        "werkzeug>=3.0,<4" \
        "scikit-learn>=1.3,<2"

# Copy application source
COPY labeler.py google_ocr.py table_detection.py prepare_dataset.py converter.py ./
COPY templates/ ./templates/

# Persistent data (uploads, annotations, xml_annotations) is mounted as a
# volume at runtime so files survive container restarts.
RUN mkdir -p labeler_data/uploads \
              labeler_data/annotations \
              labeler_data/xml_annotations

# Run as a non-root user for security
RUN useradd -m labeler && chown -R labeler:labeler /app
USER labeler

EXPOSE 5000

# Bind to 0.0.0.0 so the port is reachable from the host
CMD ["python", "labeler.py", "--host", "0.0.0.0", "--port", "5000"]
