FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

ENV PYTHONPATH=/app/src
ENV EXECUTION_MODE=PAPER

EXPOSE 8501
