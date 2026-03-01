from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import os
import joblib
import json
import secrets
import httpx
from datetime import date

app = FastAPI(title="SERP Intent Classification API", version="1.2.0")

# -----------------------------
# Key store (JSON-based)
# -----------------------------
KEYS_FILE = os.getenv("KEYS_FILE", "keys.json")


def save_key_store(store: dict) -> None:
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def load_key_store() -> dict:
    if not os.path.exists(KEYS_FILE):
        store = {"admin_key": "CHANGE-ME-ADMIN-KEY", "keys": {}}
        save_key_store(store)
        return store

    with open(KEYS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def today_str() -> str:
    return date.today().isoformat()


def get_key_record(store: dict, api_key: str):
    return store.get("keys", {}).get(api_key)


def increment_usage_or_block(api_key: str) -> None:
    store = load_key_store()
    rec = get_key_record(store, api_key)
    if rec is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    day = today_str()
    if rec.get("usage_date") != day:
        rec["usage_date"] = day
        rec["usage_count"] = 0

    limit = int(rec.get("daily_limit", 100))
    if rec.get("usage_count", 0) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (daily limit)")

    rec["usage_count"] = int(rec.get("usage_count", 0)) + 1
    store["keys"][api_key] = rec
    save_key_store(store)


def require_and_get_key_record(x_api_key: str) -> dict:
    store = load_key_store()
    rec = get_key_record(store, x_api_key)
    if rec is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    day = today_str()
    if rec.get("usage_date") != day:
        rec["usage_date"] = day
        rec["usage_count"] = 0
        store["keys"][x_api_key] = rec
        save_key_store(store)

    return rec


def require_api_key(x_api_key: str):
    increment_usage_or_block(x_api_key)


# -----------------------------
# ML model loading
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "intent_model.joblib")
ml_model = None
if os.path.exists(MODEL_PATH):
    ml_model = joblib.load(MODEL_PATH)

    DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN", "")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD", "")
DATAFORSEO_BASE_URL = "https://api.dataforseo.com/v3"


def require_dataforseo_credentials():
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        raise HTTPException(status_code=500, detail="DataForSEO credentials not configured")


async def fetch_dataforseo_search_volume_live(
    keywords: List[str],
    location_code: int,
    language_code: str,
) -> dict:
    """
    Calls:
    POST https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live
    Provides search volume, CPC, competition, etc. for up to 1000 keywords/request.
    """
    require_dataforseo_credentials()

    url = f"{DATAFORSEO_BASE_URL}/keywords_data/google_ads/search_volume/live"
    payload = [{
        "keywords": keywords,
        "location_code": location_code,
        "language_code": language_code
    }]

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD))
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"DataForSEO error: {r.status_code} {r.text}")

        return r.json()

# -----------------------------
# Pydantic models
# -----------------------------
class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1)
    daily_limit: int = Field(default=60, ge=1, le=100000)


class IntentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    serp_features: Optional[List[str]] = None


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    buying_stage: str
    conversion_score: float
    recommended_content: str
    reasons: List[str]


class BatchIntentRequest(BaseModel):
    # Removed min_items for max compatibility across Pydantic versions
    queries: List[str]
    serp_features: Optional[List[str]] = None

class KeywordAnalyzeRequest(BaseModel):
    keywords: List[str] = Field(..., min_items=1, max_items=1000)
    location_code: int = Field(..., description="DataForSEO location_code, e.g. 2528 for Netherlands")
    language_code: str = Field(default="nl", description="DataForSEO language_code, e.g. 'nl' or 'en'")
    serp_features: Optional[List[str]] = None  # optional global SERP signals if you want to influence your ML


class KeywordAnalyzeItem(BaseModel):
    keyword: str
    intent: str
    confidence: float
    buying_stage: str
    conversion_score: float
    recommended_content: str
    priority_score: int
    priority_reason: str
    # External metrics
    search_volume: Optional[int] = None
    cpc: Optional[float] = None
    competition: Optional[float] = None


class KeywordAnalyzeResponse(BaseModel):
    items: List[KeywordAnalyzeItem]
    count: int
    
class UsageResponse(BaseModel):
    name: str
    daily_limit: int
    usage_date: str
    usage_count: int
    remaining_today: int


# -----------------------------
# Intent rules + enrichment
# -----------------------------
TRANSACTIONAL = {
    "buy", "order", "deal", "discount", "coupon", "price", "pricing",
    "kopen", "bestellen", "aanbieding", "korting", "prijs", "prijzen"
}
COMMERCIAL = {
    "best", "review", "reviews", "compare", "comparison", "top", "vs", "alternative",
    "beste", "vergelijk", "vergelijking", "alternatief"
}
NAVIGATIONAL = {
    "login", "sign in", "contact", "address", "opening hours", "customer service",
    "inloggen", "adres", "openingstijden", "klantenservice"
}
LOCAL = {"near me", "in de buurt", "dichtbij", "omgeving"}


def enrich_decision_data(intent: str, serp_features: Optional[List[str]]):
    features = set([f.lower() for f in (serp_features or [])])

    stage_map = {
        "informational": "top",
        "commercial": "mid",
        "transactional": "bottom",
        "navigational": "bottom",
        "local": "bottom"
    }
    buying_stage = stage_map.get(intent, "unknown")

    base_scores = {
        "informational": 0.2,
        "commercial": 0.6,
        "transactional": 0.85,
        "navigational": 0.9,
        "local": 0.8
    }
    score = base_scores.get(intent, 0.5)

    if "shopping_ads" in features:
        score += 0.05
    if "product_listings" in features:
        score += 0.05
    if "featured_snippet" in features:
        score -= 0.05
    if "people_also_ask" in features:
        score -= 0.05

    score = max(0.0, min(score, 0.98))

    content_map = {
        "informational": "blog_guide",
        "commercial": "comparison_article",
        "transactional": "product_page",
        "local": "location_page",
        "navigational": "landing_page"
    }
    recommended_content = content_map.get(intent, "blog_guide")

    return buying_stage, score, recommended_content


def classify_intent(query: str, serp_features: Optional[List[str]]) -> Tuple[str, float, List[str]]:
    q = query.lower().strip()
    features = set([f.lower() for f in (serp_features or [])])
    reasons: List[str] = []

    if any(w in q for w in NAVIGATIONAL):
        reasons.append("Found navigational trigger (login/contact/address/etc.)")
        return "navigational", 0.80, reasons

    if any(w in q for w in TRANSACTIONAL):
        reasons.append("Found transactional trigger (buy/price/discount/etc.)")
        confidence = 0.78
        if "shopping_ads" in features or "product_listings" in features:
            confidence += 0.08
            reasons.append("SERP indicates strong shopping signals (shopping_ads/product_listings)")
        return "transactional", min(confidence, 0.95), reasons

    if any(w in q for w in COMMERCIAL):
        reasons.append("Found commercial investigation trigger (best/review/compare/etc.)")
        confidence = 0.74
        if "reviews" in features or "people_also_ask" in features:
            confidence += 0.06
            reasons.append("SERP indicates research signals (reviews/people_also_ask)")
        return "commercial", min(confidence, 0.92), reasons

    if any(w in q for w in LOCAL) or "local_pack" in features:
        reasons.append("Found local intent signal (near me/local_pack)")
        return "local", 0.72, reasons

    reasons.append("No strong buy/navigation signals; defaulting to informational")
    confidence = 0.65
    if "people_also_ask" in features or "featured_snippet" in features:
        confidence += 0.05
        reasons.append("SERP indicates informational signals (people_also_ask/featured_snippet)")
    return "informational", min(confidence, 0.85), reasons


def features_to_tokens_list(serp_features: Optional[List[str]]) -> str:
    if not serp_features:
        return ""
    feats = [f.strip().lower() for f in serp_features if isinstance(f, str) and f.strip()]
    return " ".join([f"__feat_{f}__" for f in feats])


def predict_with_ml(query: str, serp_features: Optional[List[str]] = None):
    if ml_model is None:
        return None

    text = query.strip() + " " + features_to_tokens_list(serp_features)

    label = ml_model.predict([text])[0]
    proba = ml_model.predict_proba([text])[0]
    classes = list(ml_model.classes_)
    confidence = float(proba[classes.index(label)])
    reasons = ["Predicted by ML model (SERP-aware: query + feature tokens)"]
    return label, confidence, reasons


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/admin/create_key")
def admin_create_key(req: CreateKeyRequest, x_admin_key: str = Header(default="")):
    store = load_key_store()
    if x_admin_key != store.get("admin_key"):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    new_key = secrets.token_urlsafe(32)
    store["keys"][new_key] = {
        "name": req.name,
        "daily_limit": int(req.daily_limit),
        "usage_date": today_str(),
        "usage_count": 0
    }
    save_key_store(store)

    return {"api_key": new_key, "daily_limit": req.daily_limit, "name": req.name}


@app.get("/health")
def health():
    return {"ok": True, "ml_loaded": ml_model is not None}


@app.get("/v1/usage", response_model=UsageResponse)
def usage(x_api_key: str = Header(default="")):
    rec = require_and_get_key_record(x_api_key)

    daily_limit = int(rec.get("daily_limit", 0))
    usage_count = int(rec.get("usage_count", 0))
    remaining = max(daily_limit - usage_count, 0)

    return UsageResponse(
        name=str(rec.get("name", "")),
        daily_limit=daily_limit,
        usage_date=str(rec.get("usage_date", today_str())),
        usage_count=usage_count,
        remaining_today=remaining
    )


@app.post("/v1/intent", response_model=IntentResponse)
def intent(req: IntentRequest, x_api_key: str = Header(default="")):
    require_api_key(x_api_key)

    ml_result = predict_with_ml(req.query, req.serp_features)

    if ml_result:
        label, confidence, reasons = ml_result

        if confidence < 0.55:
            rule_label, rule_conf, rule_reasons = classify_intent(req.query, req.serp_features)
            label = rule_label
            confidence = max(rule_conf, confidence)
            reasons = reasons + [f"ML confidence < 0.55; fallback to rules: {rule_label}"] + rule_reasons
    else:
        label, confidence, reasons = classify_intent(req.query, req.serp_features)

    buying_stage, conversion_score, recommended_content = enrich_decision_data(label, req.serp_features)

    return IntentResponse(
        intent=label,
        confidence=confidence,
        buying_stage=buying_stage,
        conversion_score=conversion_score,
        recommended_content=recommended_content,
        reasons=reasons
    )


@app.post("/v1/intent/batch")
def batch_intent(req: BatchIntentRequest, x_api_key: str = Header(default="")):
    require_api_key(x_api_key)

    results = []

    for q in req.queries:
        q = (q or "").strip()
        if not q:
            continue

        ml_result = predict_with_ml(q, req.serp_features)

        if ml_result:
            label, confidence, reasons = ml_result

            if confidence < 0.55:
                rule_label, rule_conf, rule_reasons = classify_intent(q, req.serp_features)
                label = rule_label
                confidence = max(rule_conf, confidence)
                reasons = reasons + [f"ML confidence < 0.55; fallback to rules: {rule_label}"] + rule_reasons
        else:
            label, confidence, reasons = classify_intent(q, req.serp_features)

        buying_stage, conversion_score, recommended_content = enrich_decision_data(label, req.serp_features)

        results.append({
            "query": q,
            "intent": label,
            "confidence": confidence,
            "buying_stage": buying_stage,
            "conversion_score": conversion_score,
            "recommended_content": recommended_content,
            "reasons": reasons
        })

        @app.post("/v1/keyword/analyze", response_model=KeywordAnalyzeResponse)
async def keyword_analyze(req: KeywordAnalyzeRequest, x_api_key: str = Header(default="")):
    require_api_key(x_api_key)

    # 1) Fetch external metrics from DataForSEO
    dfs = await fetch_dataforseo_search_volume_live(
        keywords=req.keywords,
        location_code=req.location_code,
        language_code=req.language_code,
    )

    # 2) Build a lookup from DataForSEO response
    # DataForSEO returns tasks -> result -> items (structure may include nested arrays)
    # We'll parse defensively.
    keyword_metrics = {}
    tasks = (dfs or {}).get("tasks", [])
    for t in tasks:
        results = t.get("result", []) or []
        for res in results:
            items = res.get("items", []) or []
            for it in items:
                kw = it.get("keyword")
                if not kw:
                    continue
                keyword_metrics[kw.lower()] = {
                    "search_volume": it.get("search_volume"),
                    "cpc": it.get("cpc"),
                    "competition": it.get("competition"),
                }

    # 3) Enrich with your decision engine
    out_items: List[KeywordAnalyzeItem] = []
    for kw in req.keywords:
        serp_features = req.serp_features or []

        ml_result = predict_with_ml(kw, serp_features)
        if ml_result:
            label, confidence, reasons = ml_result
            if confidence < 0.55:
                rule_label, rule_conf, rule_reasons = classify_intent(kw, serp_features)
                label = rule_label
                confidence = max(rule_conf, confidence)
        else:
            label, confidence, _ = classify_intent(kw, serp_features)

        buying_stage, conversion_score, recommended_content = enrich_decision_data(label, serp_features)
        priority_score, priority_reason = compute_priority_score(kw, label, conversion_score, serp_features)

        m = keyword_metrics.get(kw.lower(), {})
        out_items.append(KeywordAnalyzeItem(
            keyword=kw,
            intent=label,
            confidence=confidence,
            buying_stage=buying_stage,
            conversion_score=conversion_score,
            recommended_content=recommended_content,
            priority_score=priority_score,
            priority_reason=priority_reason,
            search_volume=m.get("search_volume"),
            cpc=m.get("cpc"),
            competition=m.get("competition"),
        ))

    return KeywordAnalyzeResponse(items=out_items, count=len(out_items))

    return {"results": results, "count": len(results)}
