FROM python:3.11-slim

WORKDIR /app

# Dependencies first so the layer caches across source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Corpus is generated at build time rather than shipped, so the image carries
# the generator and not 200 text files. Seed is pinned, so the image and the
# README's numbers describe the same corpus.
RUN python data/generate_corpus.py --n 200 --seed 7

# PaaS platforms (Render, Railway, Fly, Cloud Run) inject the listen port via
# $PORT and health-check the bound address. Hardcoding 8000 makes the container
# start and then fail its health check, which presents as a deploy timeout with
# no error in the app logs. The default keeps `docker run -p 8000:8000` working.
ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} is expanded at runtime; exec form would pass it literally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
