FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Run as non-root user for security
RUN useradd --create-home appuser
USER appuser

CMD ["python", "-u", "main.py"]
