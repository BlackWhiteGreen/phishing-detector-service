import ipaddress
import os
import re
import socket
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
import urllib3

# Отключаем предупреждения об отсутствии SSL-верификации при проверке подозрительных узлов
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="ML Anti-Phishing Guard", version="2.0.0")

# Инициализируем tldextract без внешних сетевых запросов при холодном старте
tld_extractor = tldextract.TLDExtract(suffix_list_urls=())

# Загрузка сохраненного бандла модели
MODEL_PATH = os.getenv("MODEL_PATH", "models/phishing_pipeline.joblib")
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
tfidf = bundle["tfidf"]
rank_dict = bundle["rank_dict"]
feature_cols = bundle["feature_cols"]
THRESHOLD = float(bundle["threshold"])

# Официальные домены брендов для точного отделения легитимного ресурса от фишинга
OFFICIAL_BRAND_DOMAINS = {
    "apple": {"apple.com", "icloud.com"},
    "sberbank": {"sberbank.ru", "sberbank.com", "sber.ru"},
    "sber": {"sberbank.ru", "sber.ru", "sbermegamarket.ru", "sbermarket.ru"},
    "paypal": {"paypal.com", "paypal.me"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "azure.com"},
    "google": {"google.com", "google.ru", "youtube.com", "gmail.com", "googleapis.com"},
    "yandex": {"yandex.ru", "ya.ru", "yandex.com", "yandex.net", "clck.ru"},
    "gosuslugi": {"gosuslugi.ru"},
    "tbank": {"tbank.ru", "tinkoff.ru"},
    "tinkoff": {"tbank.ru", "tinkoff.ru"},
    "vtb": {"vtb.ru"},
    "alfabank": {"alfabank.ru"},
    "vk": {"vk.com", "vk.ru", "vkvideo.ru"},
    "telegram": {"telegram.org", "t.me"},
    "ozon": {"ozon.ru"},
    "wildberries": {"wildberries.ru", "wb.ru"},
    "steam": {"steampowered.com", "steamcommunity.com"},
}

# Доверенные домены (защита от False Positive при блокировке зарубежных IP хостинга Render)
TRUSTED_ROOT_DOMAINS = (
    set().union(*OFFICIAL_BRAND_DOMAINS.values())
    | {
        "github.com",
        "gitlab.com",
        "habr.com",
        "rbc.ru",
        "mail.ru",
        "dzen.ru",
        "avito.ru",
        "hh.ru",
        "rutube.ru",
        "kinopoisk.ru",
        "mos.ru",
        "nalog.gov.ru",
        "cbr.ru",
        "rzd.ru",
        "aeroflot.ru",
        "pochta.ru",
        "wikipedia.org",
        "stackoverflow.com",
        "python.org",
        "render.com",
        "mirea.ru",
        "bmstu.ru",
        "hse.ru",
        "msu.ru",
        "spbu.ru",
        "itmo.ru",
        "urfu.ru",
        "mpt.ru",
    }
)

# Бесплатные облачные/статические хостинги, часто используемые для размещения фишинга
FREE_HOSTS = (
    "docs.google.com/forms",
    "storage.googleapis.com",
    "firebaseapp.com",
    "web.app",
    "pages.dev",
    "vercel.app",
    "netlify.app",
    "github.io",
    "ngrok-free.app",
    "ngrok.io",
    "000webhostapp.com",
    "weebly.com",
    "blogspot.com",
)

# Маркеры социальной инженерии в URL
PHISHING_KEYWORDS = re.compile(
    r"(verify|secure|login|signin|auth|account|update|confirm|kabinet|support|wallet|bonus|claim|recover|unlock|password|pay|bank)",
    re.IGNORECASE,
)

# Доменные зоны верхнего уровня с повышенной концентрацией дешевого фишинга
SUSPICIOUS_TLDS = {
    "xyz", "top", "online", "site", "club", "info", "pw",
    "icu", "buzz", "cfd", "sbs", "tk", "ml", "ga", "cf", "gq", "shop", "live"
}


def normalize_url(raw_url: str) -> str:
    """Приводит пользовательский ввод к стандартному формату URL."""
    url = raw_url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def is_safe_public_host(hostname: str) -> bool:
    """Защита от SSRF: проверяет, что хост не указывает на локальную/внутреннюю сеть."""
    if not hostname:
        return False
    try:
        ip_str = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip_str)
        return not (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
        )
    except Exception:
        return False


def detect_idn_and_homograph(hostname: str, suffix: str) -> tuple[int, bool]:
    """
    Определяет наличие Punycode и идеографических атак (смешение кириллицы и латиницы).
    Возвращает (has_punycode_flag, is_homograph_attack).
    """
    if not hostname:
        return 0, False

    try:
        idna_host = hostname.encode("idna").decode("ascii").lower()
    except Exception:
        idna_host = hostname.lower()

    has_xn = "xn--" in idna_host

    # Проверяем смешение латиницы и кириллицы внутри одной метки домена (например, sberbаnk.ru)
    labels = hostname.lower().split(".")
    mixed_script = False
    for label in labels:
        has_latin = bool(re.search(r"[a-z]", label))
        has_cyrillic = bool(re.search(r"[а-яё]", label))
        if has_latin and has_cyrillic:
            mixed_script = True
            break

    # Чисто кириллические домены в зонах .рф / .рус / .su без смешения алфавитов легитимны
    is_pure_cyrillic_zone = suffix.lower() in {"рф", "xn--p1ai", "рус", "xn--p1acf"}
    if mixed_script:
        return 1, True
    if has_xn and not is_pure_cyrillic_zone:
        return 1, True

    return (1 if has_xn else 0), False


def detect_brand_squatting(url_lower: str, root_domain: str) -> int:
    """
    Возвращает 1 только если имя бренда упоминается в URL,
    но сам корневой домен НЕ принадлежит этому бренду.
    """
    for brand, official_domains in OFFICIAL_BRAND_DOMAINS.items():
        if brand in url_lower:
            if root_domain not in official_domains:
                return 1
    return 0


def fetch_html_features(url: str, hostname: str) -> tuple[dict, bool]:
    """Безопасно скачивает до 256 КБ HTML и извлекает признаки через BeautifulSoup."""
    html_feats = {
        "html_length": 0,
        "iframe_count": 0,
        "password_input": 0,
        "hidden_input": 0,
        "form_count": 0,
        "script_count": 0,
        "eval_count": 0,
    }
    page_reachable = False

    if not is_safe_public_host(hostname):
        return html_feats, False

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        with requests.get(
            url,
            headers=headers,
            timeout=2.0,
            allow_redirects=True,
            stream=True,
            verify=False,
        ) as response:
            page_reachable = response.status_code < 500
            # Ограничиваем чтение первыми 256 КБ для защиты от переполнения памяти (OOM)
            chunks = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bytes_read >= 262144:
                    break

            raw_bytes = b"".join(chunks)
            content = raw_bytes.decode(response.encoding or "utf-8", errors="ignore")
            content_lower = content.lower()

            soup = BeautifulSoup(content, "html.parser")

            html_feats["html_length"] = len(content)
            html_feats["iframe_count"] = len(soup.find_all("iframe"))
            html_feats["form_count"] = len(soup.find_all("form"))
            html_feats["script_count"] = len(soup.find_all("script"))
            html_feats["password_input"] = len(
                soup.find_all(
                    "input",
                    attrs={"type": re.compile(r"^password$", re.IGNORECASE)},
                )
            )
            html_feats["hidden_input"] = len(
                soup.find_all(
                    "input",
                    attrs={"type": re.compile(r"^hidden$", re.IGNORECASE)},
                )
            )
            html_feats["eval_count"] = len(re.findall(r"\beval\s*\(", content_lower))
    except Exception:
        pass

    return html_feats, page_reachable


def extract_features_from_url_and_page(url: str):
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    url_lower = url.lower()

    # 1. Сбор признаков со страницы (HTML)
    html_feats, page_reachable = fetch_html_features(url, hostname)

    # 2. Извлечение признаков структуры URL и OSINT (Majestic Million)
    ext = tld_extractor(hostname)
    root_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else hostname

    # Проверяем также ASCII/IDNA представление для корректного поиска в словаре рангов
    try:
        idna_root = root_domain.encode("idna").decode("ascii").lower()
    except Exception:
        idna_root = root_domain

    global_rank = min(
        rank_dict.get(root_domain, 2000000),
        rank_dict.get(idna_root, 2000000),
    )
    if root_domain in TRUSTED_ROOT_DOMAINS and global_rank == 2000000:
        global_rank = 1500  # Гарантируем высокий ранг для проверенных доменов

    has_punycode, is_homograph = detect_idn_and_homograph(hostname, ext.suffix)
    is_https = 1 if url_lower.startswith("https://") else 0

    # Проверка IP строго в хосте, а не в пути URL
    has_ip = (
        1
        if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", hostname)
        or ":" in hostname
        else 0
    )

    is_free_hosting = 1 if any(h in url_lower for h in FREE_HOSTS) else 0
    brand_squatting = detect_brand_squatting(url_lower, root_domain)

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

    meta = {
        "root_domain": root_domain,
        "subdomain": ext.subdomain.lower(),
        "suffix": ext.suffix.lower(),
        "is_homograph": is_homograph,
        "page_reachable": page_reachable,
        "has_phishing_keywords": bool(PHISHING_KEYWORDS.search(url_lower)),
    }
    return X_final, combined_dict, meta


def calibrate_phishing_probability(
    raw_prob: float, feats: dict, meta: dict
) -> tuple[float, str]:
    """
    Двухконтурная калибровка (ML + Threat Intelligence Guardrails).
    Защищает бизнес-процессы от False Positive и блокирует явный фишинг при недоступном HTML.
    """
    prob = raw_prob
    root_domain = meta["root_domain"]
    suffix = meta["suffix"]

    is_gov_or_edu = suffix in {"gov.ru", "edu.ru", "mil.ru", "gov", "edu"} or root_domain.endswith(
        (".gov.ru", ".edu.ru")
    )
    is_trusted_domain = (
        root_domain in TRUSTED_ROOT_DOMAINS
        or is_gov_or_edu
        or feats["global_rank"] <= 100000
    )

    # 1. Контур защиты от False Negative (явные индикаторы компрометации)
    if meta["is_homograph"]:
        return max(prob, max(THRESHOLD + 0.35, 0.96)), "IDN Homograph (подмена символов)"

    if feats["brand_squatting"] == 1 and not is_trusted_domain:
        return max(prob, max(THRESHOLD + 0.30, 0.92)), "Мимикрия под известный бренд"

    if feats["has_ip"] == 1 and (
        feats["is_https"] == 0 or feats["password_input"] > 0 or meta["has_phishing_keywords"]
    ):
        return max(prob, max(THRESHOLD + 0.25, 0.88)), "Авторизация по прямому IP-адресу"

    if not is_trusted_domain and feats["global_rank"] == 2000000:
        # Если домен неизвестен и использует фишинговые лексемы или подозрительную TLD-зону
        if meta["has_phishing_keywords"] and (
            suffix in SUSPICIOUS_TLDS or "-" in root_domain or feats["password_input"] > 0
        ):
            return max(prob, max(THRESHOLD + 0.20, 0.85)), "Фишинговые маркеры на неизвестном домене"
        if feats["is_free_hosting"] == 1 and (
            feats["password_input"] > 0 or meta["has_phishing_keywords"]
        ):
            return max(prob, max(THRESHOLD + 0.20, 0.84)), "Подозрительная форма на бесплатном хостинге"

    # 2. Контур защиты бизнеса от False Positive (подтвержденная репутация домена)
    if (
        is_trusted_domain
        and feats["is_free_hosting"] == 0
        and feats["brand_squatting"] == 0
        and not meta["is_homograph"]
    ):
        # Демпфируем шум символьного TF-IDF на легитимных сайтах из Топ-100К и белого списка
        safe_cap = min(THRESHOLD * 0.45, 0.15)
        prob = min(prob, safe_cap)
        return prob, "Подтвержденная доменная репутация"

    return prob, "Оценка ансамбля LightGBM + TF-IDF"


class URLCheckRequest(BaseModel):
    url: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(req: URLCheckRequest):
    clean_url = normalize_url(req.url)
    X_matrix, feats, meta = extract_features_from_url_and_page(clean_url)

    raw_prob = float(model.predict_proba(X_matrix)[:, 1][0])
    final_prob, decision_source = calibrate_phishing_probability(raw_prob, feats, meta)
    is_phishing = final_prob >= THRESHOLD

    return {
        "input_url": req.url,
        "normalized_url": clean_url,
        "is_phishing": is_phishing,
        "verdict": "🚨 ФИШИНГ" if is_phishing else "✅ БЕЗОПАСНЫЙ РЕСУРС",
        "phishing_probability": round(final_prob, 4),
        "raw_ml_probability": round(raw_prob, 4),
        "threshold": round(THRESHOLD, 4),
        "decision_source": decision_source,
        "details": {
            "domain_rank": (
                feats["global_rank"]
                if feats["global_rank"] < 2000000
                else "Не входит в Топ-1М"
            ),
            "is_https": bool(feats["is_https"]),
            "password_fields": feats["password_input"],
            "punycode_detected": bool(meta["is_homograph"]),
            "brand_squatting": bool(feats["brand_squatting"]),
            "page_reachable": meta["page_reachable"],
        },
    }


@app.get("/", response_class=HTMLResponse)
async def index_page():
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
                padding: 45px 15px;
                margin: 0;
            }
            .container {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                width: 100%;
                max-width: 640px;
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
                margin-bottom: 18px;
                line-height: 1.4;
            }
            .presets {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 14px;
            }
            .preset-btn {
                background: #0f172a;
                border: 1px solid #334155;
                color: #cbd5e1;
                padding: 5px 10px;
                border-radius: 6px;
                font-size: 12px;
                cursor: pointer;
                width: auto;
                margin-top: 0;
                font-weight: 500;
            }
            .preset-btn:hover {
                border-color: #38bdf8;
                background: #1e293b;
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
            button.main-btn {
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
            button.main-btn:hover:not(:disabled) {
                background: #0369a1;
            }
            button.main-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            #resultBox {
                margin-top: 22px;
                padding: 18px;
                border-radius: 8px;
                display: none;
                line-height: 1.6;
                word-break: break-word;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">TOP-10 ХАКАТОН • КИБЕРБЕЗОПАСНОСТЬ</span>
            <h2>🛡️ ML Детектор Фишинга (LightGBM)</h2>
            <p class="sub">Комплексный анализ URL: символьные n-граммы, лексические маркеры, репутация Majestic Million и HTML-инспекция.</p>

            <div class="presets">
                <button class="preset-btn" onclick="fillAndCheck('https://ya.ru')">✅ ya.ru</button>
                <button class="preset-btn" onclick="fillAndCheck('https://sberbank.ru')">✅ sberbank.ru</button>
                <button class="preset-btn" onclick="fillAndCheck('https://sber-verify.online/login')">🚨 sber-verify.online</button>
                <button class="preset-btn" onclick="fillAndCheck('https://sberbаnk.ru')">🚨 IDN Homograph</button>
            </div>

            <input type="text" id="targetUrl" placeholder="Введите адрес (например: ya.ru или https://sber-verify.online)" onkeydown="if(event.key==='Enter') checkUrl()" />
            <button id="checkBtn" class="main-btn" onclick="checkUrl()">Проверить URL</button>

            <div id="resultBox"></div>
        </div>

        <script>
            function escapeHtml(str) {
                return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }

            function fillAndCheck(sampleUrl) {
                document.getElementById('targetUrl').value = sampleUrl;
                checkUrl();
            }

            async function checkUrl() {
                const inputField = document.getElementById('targetUrl');
                const btn = document.getElementById('checkBtn');
                const rawUrl = inputField.value.trim();
                if (!rawUrl) return;

                const box = document.getElementById('resultBox');
                btn.disabled = true;
                box.style.display = 'block';
                box.style.background = '#1e293b';
                box.style.border = '1px solid #475569';
                box.innerHTML = '⏳ Выполняется инспекция структуры URL, репутации домена и HTML-контента...';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url: rawUrl })
                    });
                    if (!response.ok) throw new Error('HTTP error');
                    const data = await response.json();

                    const isPhish = data.is_phishing;
                    const color = isPhish ? '#ef4444' : '#22c55e';
                    box.style.border = `1px solid ${color}`;
                    box.style.background = isPhish ? 'rgba(239, 68, 68, 0.12)' : 'rgba(34, 197, 94, 0.12)';

                    box.innerHTML = `
                        <div style="font-size: 19px; font-weight: bold; color: ${color}; margin-bottom: 6px;">${escapeHtml(data.verdict)}</div>
                        <div><b>Проверенный URL:</b> <code>${escapeHtml(data.normalized_url)}</code></div>
                        <div><b>Вероятность фишинга:</b> ${(data.phishing_probability * 100).toFixed(1)}% <span style="color:#94a3b8; font-size:13px;">(порог: ${escapeHtml(data.threshold)})</span></div>
                        <div><b>Драйвер решения:</b> ${escapeHtml(data.decision_source)}</div>
                        <hr style="border: 0; border-top: 1px solid #334155; margin: 12px 0;">
                        <div style="font-size: 13px; color: #cbd5e1;">
                            • Ранг Majestic Million: <b>${escapeHtml(data.details.domain_rank)}</b><br>
                            • Наличие HTTPS: <b>${data.details.is_https ? 'Да' : 'Нет'}</b><br>
                            • Доступность HTML-страницы: <b>${data.details.page_reachable ? 'Проинспектирована онлайн' : 'Недоступна / Оценка по OSINT и URL'}</b><br>
                            • Полей пароля в форме: <b>${escapeHtml(data.details.password_fields)}</b><br>
                            • Punycode / IDN-спуфинг: <b>${data.details.punycode_detected ? 'Обнаружен (Опасно)' : 'Нет'}</b><br>
                            • Мимикрия под бренд: <b>${data.details.brand_squatting ? 'Обнаружена' : 'Нет'}</b>
                        </div>
                    `;
                } catch (err) {
                    box.style.border = '1px solid #ef4444';
                    box.style.background = 'rgba(239, 68, 68, 0.1)';
                    box.innerHTML = '❌ Ошибка связи с микросервисом. Попробуйте еще раз.';
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
