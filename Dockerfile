FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей для LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY app.py .

# Создаем непривилегированного пользователя (Best Practice для ИБ-проектов)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Поддержка динамического порта Render.com ($PORT) с фоллбэком на 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
