FROM python:3.10-slim

# --- system deps ------------------------------------------------------------
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# --- project ---------------------------------------------------------------
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip \
 && pip install poetry \
 && poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --only main

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
