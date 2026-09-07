FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal on purpose -- this image should stay small and fast to pull in CI.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY prompts/ prompts/
COPY golden_dataset/ golden_dataset/
COPY run_eval.py .

# Persisted at runtime via a mounted volume so eval history survives across container runs.
VOLUME ["/app/data", "/app/reports"]

# Configured via env vars at `docker run` time:
#   OPENAI_API_KEY      - required
#   SLACK_WEBHOOK_URL    - optional, alerts are skipped if unset
#   PROMPT_VERSION        - optional, defaults to the latest prompt file
ENTRYPOINT ["python", "run_eval.py"]
