# Use Python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy FastAPI backend code into container
COPY ./server /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]