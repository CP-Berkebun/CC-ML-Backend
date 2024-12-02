# Gunakan base image Python
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Salin semua file ke container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port untuk aplikasi FastAPI
EXPOSE 8000

# Jalankan aplikasi menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
