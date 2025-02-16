# Use Python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /ml

# Copy everything into container
COPY . /ml

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg

# Keep container running with Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]