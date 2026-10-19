import os
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel
import requests
import scipy.sparse as sp
import tldextract

app = FastAPI(title="ML Anti-Phishing Guard")

# Загрузка сохраненного бандла модели
MODEL_PATH = "models/phishing_pipeline.joblib"
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
tfidf = bundle["tfidf"]
rank_dict = bundle["rank_dict"]
feature_cols = bundle["feature_cols"]
THRESHOLD = bundle["threshold"]


def normalize_url(raw_url: str) -> str:
  """Приводит пользовательский ввод к стандартному формату URL."""
  url = raw_url.strip()
  # Если пользователь не ввел протокол, по умолчанию добавляем https://
  if not re.match(r"^https?://", url, re.IGNORECASE):
    url = "https://" + url
  return url


def extract_features_from_url_and_page(url: str):
  # 1. Сбор признаков со страницы (HTML) с коротким таймаутом
  html_feats = {
      "html_length": 0,
      "iframe_count": 0,
      "password_input": 0,
      "hidden_input": 0,
      "form_count": 0,
      "script_count": 0,
      "eval_count": 0,
  }

  try:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    # Таймаут 1.5 секунды, чтобы микросервис не вис на мертвых сайтах
    response = requests.get(
        url, headers=headers, timeout=1.5, allow_redirects=True
    )
    content = response.text.lower()

    html_feats["html_length"] = len(content)
    html_feats["iframe_count"] = content.count("<iframe")
    html_feats["password_input"] = content.count('type="password"')
    html_feats["hidden_input"] = content.count('type="hidden"')
    html_feats["form_count"] = content.count("<form")
    html_feats["script_count"] = content.count("<script")
    html_feats["eval_count"] = len(re.findall(r"eval\(", content))
  except Exception:
    # Если ресурс недоступен или заблокирован — эвристика опирается на URL и репутацию
    pass

  # 2. Извлечение признаков структуры URL и OSINT (Majestic Million)
  ext = tldextract.extract(url)
  root_domain = f"{ext.domain}.{ext.suffix}".lower()
  global_rank = rank_dict.get(root_domain, 2000000)

  has_punycode = 1 if "xn--" in url.lower() else 0
  is_https = 1 if url.lower().startswith("https://") else 0
  has_ip = 1 if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0

  free_hosts = [
      "docs.google.com",
      "storage.googleapis.com",
      "firebaseapp.com",
      "vk.com",
  ]
  is_free_hosting = 1 if any(h in url.lower() for h in free_hosts) else 0

  brands = ["apple", "sberbank", "paypal", "microsoft", "google", "yandex"]
  brand_squatting = 1 if any(b in url.lower() for b in brands) else 0

  url_feats = {
      "global_rank": global_rank,
      "has_punycode": has_punycode,
      "is_https": is_https,
      "has_ip": has_ip,
      "is_free_hosting": is_free_hosting,
      "brand_squatting": brand_squatting,
  }

  # Формирование числового вектора в строгом соответствии с порядком при обучении
  combined_dict = {**html_feats, **url_feats}
  numeric_vector = np.array([[combined_dict[col] for col in feature_cols]])

  # Char TF-IDF векторизация URL
  tfidf_vector = tfidf.transform([url])

  # Объединение в разреженную матрицу для подачи в модель
  X_final = sp.hstack([tfidf_vector, numeric_vector]).tocsr()
  return X_final, combined_dict


class URLCheckRequest(BaseModel):
  url: str


@app.post("/predict")
def predict_endpoint(req: URLCheckRequest):
  clean_url = normalize_url(req.url)
  X_matrix, feats = extract_features_from_url_and_page(clean_url)

  prob = float(model.predict_proba(X_matrix)[:, 1][0])
  is_phishing = prob >= THRESHOLD

  return {
      "input_url": req.url,
      "normalized_url": clean_url,
      "is_phishing": is_phishing,
      "verdict": "🚨 ФИШИНГ" if is_phishing else "✅ БЕЗОПАСНЫЙ РЕСУРС",
      "phishing_probability": round(prob, 4),
      "threshold": round(THRESHOLD, 4),
      "details": {
          "domain_rank": (
              feats["global_rank"]
              if feats["global_rank"] < 2000000
              else "Не входит в Топ-1М"
          ),
          "is_https": bool(feats["is_https"]),
          "password_fields": feats["password_input"],
          "punycode_detected": bool(feats["has_punycode"]),
          "brand_squatting": bool(feats["brand_squatting"]),
      },
  }


@app.get("/", response_class=HTMLResponse)
def index_page():
  return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Anti-Phishing Guard</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #0b0f19;
                color: #f1f5f9;
                display: flex;
                justify-content: center;
                padding-top: 50px;
                margin: 0;
            }
            .container {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                width: 90%;
                max-width: 620px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            }
            .badge {
                background: #0284c7;
                color: #e0f2fe;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            h2 {
                margin: 15px 0 8px 0;
                color: #38bdf8;
            }
            p.sub {
                color: #94a3b8;
                font-size: 14px;
                margin-top: 0;
                margin-bottom: 20px;
                line-height: 1.4;
            }
            input {
                width: 100%;
                box-sizing: border-box;
                padding: 13px 15px;
                background: #0f172a;
                border: 1px solid #475569;
                border-radius: 8px;
                color: #fff;
                font-size: 15px;
                outline: none;
                transition: 0.2s;
            }
            input:focus {
                border-color: #38bdf8;
                box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
            }
            button {
                width: 100%;
                margin-top: 12px;
                padding: 13px;
                background: #0284c7;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-weight: 700;
                font-size: 16px;
                cursor: pointer;
                transition: 0.2s;
            }
            button:hover {
                background: #0369a1;
            }
            #resultBox {
                margin-top: 22px;
                padding: 18px;
                border-radius: 8px;
                display: none;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">TOP-10 ХАКАТОН • КИБЕРБЕЗОПАСНОСТЬ</span>
            <h2>🛡️ ML Детектор Фишинга (LightGBM)</h2>
            <p class="sub">Комплексный анализ URL: n-граммы, лексические маркеры, Majestic Million и HTML-инспекция.</p>
            
            <input type="text" id="targetUrl" placeholder="Введите адрес (например: ya.ru или https://sber-verify.online)" onkeydown="if(event.key==='Enter') checkUrl()" />
            <button onclick="checkUrl()">Проверить URL</button>

            <div id="resultBox"></div>
        </div>

        <script>
            async function checkUrl() {
                const inputField = document.getElementById('targetUrl');
                const rawUrl = inputField.value.trim();
                if (!rawUrl) return;

                const box = document.getElementById('resultBox');
                box.style.display = 'block';
                box.style.background = '#1e293b';
                box.style.border = '1px solid #475569';
                box.innerHTML = '⏳ Выполняется инспекция структуры URL и контента...';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url: rawUrl })
                    });
                    const data = await response.json();

                    const isPhish = data.is_phishing;
                    const color = isPhish ? '#ef4444' : '#22c55e';
                    box.style.border = `1px solid ${color}`;
                    box.style.background = isPhish ? 'rgba(239, 68, 68, 0.12)' : 'rgba(34, 197, 94, 0.12)';

                    box.innerHTML = `
                        <div style="font-size: 19px; font-weight: bold; color: ${color}; margin-bottom: 6px;">${data.verdict}</div>
                        <div><b>Проверенный URL:</b> <code>${data.normalized_url}</code></div>
                        <div><b>Вероятность фишинга:</b> ${(data.phishing_probability * 100).toFixed(1)}%</div>
                        <div><b>Рабочий порог:</b> ${data.threshold}</div>
                        <hr style="border: 0; border-top: 1px solid #334155; margin: 12px 0;">
                        <div style="font-size: 13px; color: #cbd5e1;">
                            • Ранг Majestic Million: <b>${data.details.domain_rank}</b><br>
                            • Наличие HTTPS: <b>${data.details.is_https ? 'Да' : 'Нет'}</b><br>
                            • Полей пароля в форме: <b>${data.details.password_fields}</b><br>
                            • Punycode (IDN spoofing): <b>${data.details.punycode_detected ? 'Обнаружен (Опасно)' : 'Нет'}</b><br>
                            • Мимикрия под бренд: <b>${data.details.brand_squatting ? 'Обнаружена' : 'Нет'}</b>
                        </div>
                    `;
                } catch (err) {
                    box.style.border = '1px solid #ef4444';
                    box.style.background = 'rgba(239, 68, 68, 0.1)';
                    box.innerHTML = '❌ Ошибка связи с микросервисом. Попробуйте еще раз.';
                }
            }
        </script>
    </body>
    </html>
    """
