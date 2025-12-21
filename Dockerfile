FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# kod aplikacji
COPY src ./src

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
