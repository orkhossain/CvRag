FROM python:3.11-slim

WORKDIR /app

# system deps (for pdfminer + dateutil)
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# copy deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# expose default HF port
EXPOSE 7860

# start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]