# Running the Labeler with Docker

This guide lets **any annotator** run the labeler on their laptop with just [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed — no Python or package setup required.

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

## Step 2 – (Optional) Set a Google Cloud Vision API key

If you want to use **Google OCR** inside the labeler, create a file called `.env` in the project root:

```bash
# .env  (never commit this file)
GOOGLE_API_KEY=...your-key-here...
```

Skip this step if you only need Docling's built-in text extraction or RapidOCR.

---

## Step 3 – Build and start the container

```bash
docker compose up --build
```

What this does:
1. Builds a Docker image with Python 3.11, all labeler dependencies, and poppler (for PDF → image conversion).
2. Starts the labeler Flask server inside a container.
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
2. Select an OCR model if needed (Docling / RapidOCR / Google).
3. Review and correct the detected elements on each page.
4. Click **Export XML** to save annotations to `labeler_data/xml_annotations/`.

Annotations are saved directly to the **`labeler_data/`** folder on your machine (via the Docker volume mount), so they persist after the container is stopped.

---

## Step 6 – Prepare the training dataset

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
| Google OCR returns empty text | Verify `GOOGLE_API_KEY` in `.env` and that the Vision API is enabled |
| Slow first run | Normal — docling models are downloaded on first use inside the container |
