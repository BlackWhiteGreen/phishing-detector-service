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

app = FastAPI(title="Anti-Phishing ML Service")

# Загружаем сохраненный бандл
bundle = joblib.load("models/phishing_pipeline.joblib")
model = bundle["model"]
tfidf = bundle["tfidf"]
rank_dict = bundle["rank_dict"]
feature_cols = bundle["feature_cols"]
THRESHOLD = bundle["threshold"]


def analyze_url_and_html(url: str):
  # 1. Попытка легкого сбора HTML (с таймаутом 1.5 сек)
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
        )
    }
    resp = requests.get(url, headers=headers, timeout=1.5)
    content = resp.text.lower()
    html_feats["html_length"] = len(content)
    html_feats["iframe_count"] = content.count("<iframe")
    html_feats["password_input"] = content.count('type="password"')
    html_feats["hidden_input"] = content.count('type="hidden"')
    html_feats["form_count"] = content.count("<form")
    html_feats["script_count"] = content.count("<script")
    html_feats["eval_count"] = len(re.findall(r"eval\(", content))
  except Exception:
    # Если сайт мертв/недоступен — оставляем нули, сработают эвристики URL
    pass

  # 2. Признаки самого URL
  ext = tldextract.extract(url)
  root_domain = f"{ext.domain}.{ext.suffix}".lower()
  global_rank = rank_dict.get(root_domain, 2000000)

  has_punycode = 1 if "xn--" in url.lower() else 0
  is_https = 1 if url.startswith("https") else 0
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

  # Собираем числовой вектор в том же порядке, что и при обучении
  all_feats = {**html_feats, **url_feats}
  num_vector = np.array([[all_feats[col] for col in feature_cols]])

  # TF-IDF по URL
  tfidf_vec = tfidf.transform([url])

  # Объединяем разреженную матрицу
  X_input = sp.hstack([tfidf_vec, num_vector]).tocsr()
  return X_input, all_feats


class CheckRequest(BaseModel):
  url: str


@app.post("/predict")
def predict(req: CheckRequest):
  X_input, feats = analyze_url_and_html(req.url)
  prob = float(model.predict_proba(X_input)[:, 1][0])
  is_phish = prob >= THRESHOLD

  return {
      "url": req.url,
      "is_phishing": is_phish,
      "verdict": "🚨 ФИШИНГ" if is_phish else "✅ БЕЗОПАСНЫЙ РЕСУРС",
      "phishing_probability": round(prob, 4),
      "threshold": round(THRESHOLD, 4),
      "details": {
          "domain_rank": feats["global_rank"],
          "password_fields": feats["password_input"],
          "punycode_detected": bool(feats["has_punycode"]),
          "brand_squatting": bool(feats["brand_squatting"]),
      },
  }


@app.get("/", response_class=HTMLResponse)
def home():
  return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>ML Anti-Phishing Guard</title>
        <style>
            body { font-family: system-ui, sans-serif; background: #0b0f19; color: #f1f5f9; display: flex; justify-content: center; padding-top: 50px; }
            .box { background: #1e293b; border: 1px solid #334155; border-radius: 12px; width: 620px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            h2 { margin-top: 0; color: #38bdf8; }
            .badge { background: #0369a1; color: #e0f2fe; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; }
            input { width: 100%; box-sizing: border-box; padding: 12px; margin: 18px 0 10px 0; background: #0f172a; border: 1px solid #475569; border-radius: 6px; color: #fff; font-size: 15px; }
            button { width: 100%; padding: 12px; background: #0284c7; border: none; border-radius: 6px; color: white; font-weight: bold; font-size: 16px; cursor: pointer; transition: 0.2s; }
            button:hover { background: #0369a1; }
            #res { margin-top: 20px; padding: 16px; border-radius: 8px; display: none; line-height: 1.6; }
        </style>
    </head>
    <body>
        <div class="box">
            <span class="badge">TOP-10 ХАКАТОН • КИБЕРБЕЗОПАСНОСТЬ</span>
            <h2>🛡️ ML Детектор Фишинга (LightGBM)</h2>
            <p style="color: #94a3b8; font-size: 14px;">Комплексный анализ URL: n-граммы, лексические маркеры, Majestic Million и HTML-инспекция.</p>
            <input type="text" id="url" placeholder="Вставьте ссылку, например: https://sber-verify.online/login" />
            <button onclick="check()">Проверить URL</button>
            <div id="res"></div>
        </div>
        <script>
            async function check() {
                const u = document.getElementById('url').value.trim();
                if(!u) return;
                const r = document.getElementById('res');
                r.style.display = 'block';
                r.style.background = '#1e293b';
                r.style.border = '1px solid #475569';
                r.innerHTML = '⏳ Выполняется анализ параметров URL и контента...';
                try {
                    const resp = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({url: u})
                    });
                    const d = await resp.json();
                    const color = d.is_phishing ? '#ef4444' : '#22c55e';
                    r.style.border = `1px solid ${color}`;
                    r.style.background = d.is_phishing ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)';
                    r.innerHTML = `
                        <div style="font-size: 18px; font-weight: bold; color: ${color}; margin-bottom: 8px;">${d.verdict}</div>
                        <div><b>Вероятность фишинга:</b> ${(d.phishing_probability * 100).toFixed(1)}%</div>
                        <div><b>Рабочий порог:</b> ${d.threshold}</div>
                        <hr style="border: 0; border-top: 1px solid #334155; margin: 10px 0;">
                        <div style="font-size: 13px; color: #cbd5e1;">
                            • Ранг домена в Majestic: <b>${d.details.domain_rank > 1000000 ? 'Не входит в топ (Подозрительно)' : d.details.domain_rank}</b><br>
                            • Полей пароля на странице: <b>${d.details.password_fields}</b><br>
                            • Punycode (IDN spoofing): <b>${d.details.punycode_detected ? 'Да (Опасно)' : 'Нет'}</b><br>
                            • Мимикрия под бренд: <b>${d.details.brand_squatting ? 'Обнаружена' : 'Нет'}</b>
                        </div>
                    `;
                } catch(e) {
                    r.innerHTML = '❌ Ошибка при обращении к микросервису.';
                }
            }
        </script>
    </body>
    </html>
    """
