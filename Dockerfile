FROM python:3.9-slim

WORKDIR /app

COPY . .

# Install system dependencies (optional, if you need them for numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pinned versions
RUN pip install --no-cache-dir \
    "tensorflow==2.15.0" \
    "numpy<2.0" \
    fastapi \
    uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
