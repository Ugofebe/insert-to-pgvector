# Multi-stage build for optimized production image
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # poppler-utils
    # libgl1-mesa-dev 
    # libglib2.0-dev
    # tesseract-ocr
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser *.py ./

# Create directories for data with proper permissions
RUN mkdir -p /app/research_db /app/ocr_data_output /app/.cache/huggingface /app/.cache/sentence_transformers /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables for HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["python", "insert_pg_3.py"]