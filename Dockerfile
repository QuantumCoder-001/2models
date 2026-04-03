# 1. Use a lightweight Python image
FROM python:3.9-slim

# 2. Install Tesseract and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your project files into the container
COPY . .

# 6. Set the command to run your app
# Ensure "app" matches your Flask variable name and "app.py" matches your filename
CMD ["python", "app.py"]
