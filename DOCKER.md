# Running the Labeler with Docker

This guide lets **any annotator** run the labeler on their laptop with just [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed — no Python or package setup required.

The container uses **gunicorn** (a production-grade WSGI server) so multiple users can annotate simultaneously without interfering with each other.  Annotations can be saved to an **SQLite database** (inside the container's `labeler_data/` volume) so each user's work is tracked separately by session ID.

---

## Prerequisites

| Tool | Minimum version | Download |
|------|----------------|---------|
| Docker Desktop (Windows / macOS) or Docker Engine + Compose (Linux) | Docker 24 · Compose v2 | https://docs.docker.com/get-docker/ |
| Git (to clone the repo) | any | https://git-scm.com |

> **Windows users**: enable *WSL 2 backend* in Docker Desktop settings for best performance.

---

## Step 1 – Clone the repository

```bash
git clone https://github.com/minje1112/smol_docling_generator.git
cd smol_docling_generator
```

---

## Step 2 – (Optional) Configure environment variables

Create a `.env` file in the project root to customise the container:

```bash
# .env  (never commit this file)

# Google Cloud Vision API key – required only for Google OCR
GOOGLE_OCR_API_KEY=AIzaSy...your-key-here...

# Number of gunicorn worker processes (default 4).
# Increase for teams with many concurrent annotators.
WEB_CONCURRENCY=4
```

Both variables are optional.  Skip this step if you only need Docling's built-in text extraction or EasyOCR.

---

## Step 3 – Build and start the container

```bash
docker compose up --build
```

What this does:
1. Builds a Docker image with Python 3.11, all labeler dependencies, and poppler (for PDF → image conversion).
2. Starts the labeler via **gunicorn** (4 worker processes by default) inside a container.
3. Maps port **5000** on your machine to the container.

The first build downloads packages and may take **3–5 minutes**. Subsequent starts (without `--build`) are instant.

---

## Step 4 – Open the labeler

Open your browser and go to:

```
http://localhost:5000
```

You should see the Docling Tag Labeler interface ready to use.

---

## Step 5 – Upload and annotate PDFs

1. Click **Upload PDF** and choose a PDF file.
2. Select an OCR model if needed (Docling / EasyOCR / Google Vision).
3. Review and correct the detected elements on each page.
4. Click **Save to DB** to store annotations in the SQLite database under a unique session ID (safe for multi-user setups), or **Export XML** to save traditional XML files.

Annotations are saved to the **`labeler_data/`** folder on your machine (via the Docker volume mount), so they persist after the container is stopped.

---

## Step 6 – Multi-user annotation (SQLite database)

The labeler exposes database-backed endpoints that keep each user's annotations separate:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/db/save_annotations` | POST | Save annotations; returns a `session_id` |
| `/db/list_sessions` | GET | List all saved sessions |
| `/db/load_annotations/<session_id>` | GET | Load a specific session |
| `/db/delete_session/<session_id>` | DELETE | Remove a session permanently |

Example – save and retrieve with `curl`:

```bash
# Save (returns a session_id)
curl -X POST http://localhost:5000/db/save_annotations \
     -H "Content-Type: application/json" \
     -d '{"document_name":"my_doc","pages":{"1":{"image_name":"page_1","width":1240,"height":1754,"elements":[]}}}'

# List all sessions
curl http://localhost:5000/db/list_sessions

# Load a specific session
curl http://localhost:5000/db/load_annotations/<session_id>
```

The database file is stored at `labeler_data/annotations.db` and is included in the Docker volume mount.

---

## Step 7 – Prepare the training dataset

After annotating, run `prepare_dataset.py` either inside the container or locally:

```bash
# Inside the running container
docker compose exec labeler python prepare_dataset.py \
    --xml_dir labeler_data/xml_annotations \
    --images_dir labeler_data/uploads \
    --output training_data.jsonl

# Or locally (if you have Python set up)
python prepare_dataset.py \
    --xml_dir labeler_data/xml_annotations \
    --images_dir labeler_data/uploads \
    --output training_data.jsonl
```

---

## Stopping the labeler

```bash
docker compose down
```

Your files in `labeler_data/` are **not deleted** when the container stops.

---

## Useful commands

| Command | Description |
|---------|-------------|
| `docker compose up --build` | Rebuild the image and start |
| `docker compose up -d` | Start in the background (detached) |
| `docker compose down` | Stop and remove the container |
| `docker compose logs -f` | Stream live logs |
| `docker compose exec labeler bash` | Open a shell inside the container |

---

## Changing the port

If port 5000 is already in use on your machine, edit `docker-compose.yml` and change the left side of the port mapping:

```yaml
ports:
  - "8080:5000"   # now accessible at http://localhost:8080
```

Then restart with `docker compose up`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `port is already allocated` | Change the host port in `docker-compose.yml` |
| Page images are blank after upload | Check that `labeler_data/uploads/` exists and is writable |
| `permission denied` on `labeler_data/` | Run `chmod -R 777 labeler_data/` on the host |
| Google OCR returns empty text | Verify `GOOGLE_OCR_API_KEY` in `.env` and that the Vision API is enabled |
| Slow first run | Normal — docling models are downloaded on first use inside the container |
| `db/save_annotations` returns 500 | Check container logs (`docker compose logs labeler`) for SQLite errors |
