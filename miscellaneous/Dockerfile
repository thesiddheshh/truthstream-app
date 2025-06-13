# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy local code to the container
COPY . /app/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgl1 \
        libsm6 \
        ffmpeg \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy language model
RUN python -m spacy download en_core_web_sm

# Expose port for Streamlit
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address=0.0.0.0"]
