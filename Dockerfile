FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt



RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*


COPY --chown=user . /app
CMD ["uvicorn", "cvrag.main:app", "--host", "0.0.0.0", "--port", "7860"]
