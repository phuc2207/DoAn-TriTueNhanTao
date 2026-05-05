"""
Hệ thống Dự báo Giá Nhà Đất TP.HCM  ─  FastAPI Backend v3.0
=============================================================

Cải tiến v3:
  ✅ Endpoint /ors-route  → proxy ORS API (giải quyết CORS hoàn toàn)
  ✅ /predict hỗ trợ TF-IDF vectorizer + 3 feature mới
       (So_tang, Mat_tien_m, Huong_nha)
  ✅ Inference nhất quán với train_model.py v3
  ✅ httpx async client cho ORS proxy (thay requests sync)
  ✅ Giữ nguyên /recommend, /market-analysis, /route (Dijkstra)
"""

import heapq, logging, re
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# 1. LIFESPAN – LOAD MODEL
# ══════════════════════════════════════════════════════════
MODEL_PATH = "house_price_model.pkl"
model_data: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        feats = len(model_data.get("feature_columns", []))
        name  = model_data.get("model_name", "unknown")
        logger.info(f"✅ Model '{name}' loaded – {feats} struct features + TF-IDF")
    except FileNotFoundError:
        logger.warning(f"⚠️  '{MODEL_PATH}' not found. /predict uses heuristic fallback.")
    yield
    model_data.clear()
    logger.info("Shutdown – resources released.")

# ══════════════════════════════════════════════════════════
# 2. APP
# ══════════════════════════════════════════════════════════
app = FastAPI(
    title="🏘️ BĐS TP.HCM – Dự báo Giá Nhà",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# 3. PYDANTIC MODELS
# ══════════════════════════════════════════════════════════
VALID_DISTRICTS = [
    "Quan_1","Quan_2","Quan_3","Quan_4","Quan_5","Quan_6",
    "Quan_7","Quan_8","Quan_10","Quan_11","Quan_12",
    "Quan_Binh_Thanh","Quan_Tan_Binh","Quan_Phu_Nhuan",
    "Quan_Go_Vap","Quan_Binh_Tan","Quan_Thu_Duc",
    "Huyen_Binh_Chanh","Huyen_Nha_Be","Huyen_Hoc_Mon",
]
VALID_HUONG = ["Đông","Tây","Nam","Bắc","Đông Nam","Đông Bắc","Tây Nam","Tây Bắc"]

class HouseInput(BaseModel):
    Dien_tich:       float = Field(..., gt=0,  le=10_000, description="Diện tích m²")
    So_phong:        int   = Field(..., ge=1,  le=20,     description="Số phòng ngủ")
    So_tang:         int   = Field(2,   ge=1,  le=30,     description="Số tầng")
    Mat_tien_m:      float = Field(4.0, ge=2,  le=30,     description="Chiều rộng mặt tiền (m)")
    Huong_nha:       str   = Field("Nam",                 description="Hướng nhà")
    Vi_tri:          str   = Field(...,                   description="Quận/Huyện")
    Noi_dung_mo_ta:  str   = Field(..., min_length=5,     description="Mô tả nhà")

    @field_validator("Vi_tri")
    @classmethod
    def chk_district(cls, v):
        if v not in VALID_DISTRICTS:
            raise ValueError(f"Vi_tri '{v}' không hợp lệ.")
        return v

    @field_validator("Huong_nha")
    @classmethod
    def chk_huong(cls, v):
        if v not in VALID_HUONG:
            raise ValueError(f"Huong_nha '{v}' không hợp lệ.")
        return v

class RecommendRequest(BaseModel):
    Dien_tich: float = Field(..., gt=0)
    So_phong:  int   = Field(..., ge=1, le=20)
    Vi_tri:    str
    Gia:       float = Field(..., gt=0)

class MarketQueryRequest(BaseModel):
    Vi_tri:    Optional[str]   = None
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)

class RouteRequest(BaseModel):
    start: str
    end:   str

class ORSRouteRequest(BaseModel):
    """Proxy tới OpenRouteService – nhận lng/lat theo thứ tự ORS"""
    from_lat: float = Field(..., description="Vĩ độ điểm xuất phát")
    from_lng: float = Field(..., description="Kinh độ điểm xuất phát")
    to_lat:   float = Field(..., description="Vĩ độ điểm đến")
    to_lng:   float = Field(..., description="Kinh độ điểm đến")

# ══════════════════════════════════════════════════════════
# 4. DIJKSTRA – 20 QUẬN TP.HCM
# ══════════════════════════════════════════════════════════
DISTRICT_GRAPH: dict[str, dict[str, float]] = {
    "Quan_1":           {"Quan_3": 1.5, "Quan_4": 2.0, "Quan_5": 2.5, "Quan_Binh_Thanh": 3.0},
    "Quan_2":           {"Quan_1": 3.5, "Quan_Binh_Thanh": 4.0, "Quan_7": 5.0, "Quan_Thu_Duc": 6.0},
    "Quan_3":           {"Quan_1": 1.5, "Quan_10": 2.0, "Quan_Tan_Binh": 3.5, "Quan_Phu_Nhuan": 2.5},
    "Quan_4":           {"Quan_1": 2.0, "Quan_7": 3.0, "Quan_8": 3.5},
    "Quan_5":           {"Quan_1": 2.5, "Quan_6": 2.0, "Quan_10": 2.5, "Quan_11": 2.0},
    "Quan_6":           {"Quan_5": 2.0, "Quan_8": 2.5, "Quan_11": 2.5, "Quan_Binh_Tan": 3.0},
    "Quan_7":           {"Quan_4": 3.0, "Quan_8": 4.0, "Huyen_Nha_Be": 6.0},
    "Quan_8":           {"Quan_4": 3.5, "Quan_6": 2.5, "Quan_7": 4.0, "Huyen_Binh_Chanh": 7.0},
    "Quan_10":          {"Quan_3": 2.0, "Quan_5": 2.5, "Quan_11": 1.5, "Quan_Tan_Binh": 3.0},
    "Quan_11":          {"Quan_5": 2.0, "Quan_6": 2.5, "Quan_10": 1.5, "Quan_Go_Vap": 3.5},
    "Quan_12":          {"Quan_Go_Vap": 3.0, "Huyen_Hoc_Mon": 5.0, "Quan_Binh_Thanh": 5.5},
    "Quan_Binh_Thanh":  {"Quan_1": 3.0, "Quan_3": 2.5, "Quan_Phu_Nhuan": 2.0, "Quan_Go_Vap": 3.5, "Quan_2": 4.0},
    "Quan_Tan_Binh":    {"Quan_3": 3.5, "Quan_10": 3.0, "Quan_Phu_Nhuan": 2.5, "Quan_Go_Vap": 4.0},
    "Quan_Phu_Nhuan":   {"Quan_3": 2.5, "Quan_Binh_Thanh": 2.0, "Quan_Tan_Binh": 2.5},
    "Quan_Go_Vap":      {"Quan_11": 3.5, "Quan_12": 3.0, "Quan_Binh_Thanh": 3.5, "Quan_Tan_Binh": 4.0},
    "Quan_Binh_Tan":    {"Quan_6": 3.0, "Huyen_Binh_Chanh": 5.0, "Quan_Tan_Binh": 5.5},
    "Quan_Thu_Duc":     {"Quan_2": 6.0, "Quan_Binh_Thanh": 5.0, "Quan_12": 6.0},
    "Huyen_Binh_Chanh": {"Quan_8": 7.0, "Quan_Binh_Tan": 5.0, "Huyen_Hoc_Mon": 8.0},
    "Huyen_Nha_Be":     {"Quan_7": 6.0},
    "Huyen_Hoc_Mon":    {"Quan_12": 5.0, "Huyen_Binh_Chanh": 8.0},
}

def dijkstra(graph: dict, start: str, end: str) -> tuple[float, list[str]]:
    if start not in graph or end not in graph:
        return float("inf"), []
    dist = {n: float("inf") for n in graph}
    dist[start] = 0.0
    prev: dict[str, Optional[str]] = {n: None for n in graph}
    pq = [(0.0, start)]
    while pq:
        cd, cu = heapq.heappop(pq)
        if cu == end: break
        if cd > dist[cu]: continue
        for nb, w in graph.get(cu, {}).items():
            nd = cd + w
            if nd < dist[nb]:
                dist[nb] = nd; prev[nb] = cu
                heapq.heappush(pq, (nd, nb))
    path, n = [], end
    while n:
        path.append(n); n = prev[n]
    path.reverse()
    if not path or path[0] != start:
        return float("inf"), []
    return dist[end], path

# ══════════════════════════════════════════════════════════
# 5. THANH KHOẢN
# ══════════════════════════════════════════════════════════
def decision_tree_liquidity(area, km, mat_tien, so_hong) -> dict:
    if km <= 3:
        if mat_tien:  return {"nhan": "Rất cao – Đầu tư tuyệt vời", "diem": 10}
        elif so_hong: return {"nhan": "Cao – Dễ thanh khoản",        "diem": 8}
        else:         return {"nhan": "Khá cao – Tiềm năng tốt",     "diem": 7}
    elif km <= 6:
        if mat_tien:  return {"nhan": "Khá cao – Vị trí vệ tinh",    "diem": 7}
        elif area<=80:return {"nhan": "Trung bình – Diện tích nhỏ dễ bán","diem": 5}
        else:         return {"nhan": "Trung bình – Phù hợp ở thực", "diem": 5}
    else:
        if area > 150:return {"nhan": "Thấp – Kén khách mua",         "diem": 3}
        else:         return {"nhan": "Thấp – Ngoại ô xa trung tâm", "diem": 3}

# ══════════════════════════════════════════════════════════
# 6. NLP
# ══════════════════════════════════════════════════════════
NLP_KEYWORDS = [
    "sổ hồng","sổ đỏ","hẻm xe hơi","ô tô vào","nở hậu",
    "mặt tiền","nội thất cao cấp","hồ bơi","hầm xe",
    "biệt thự","chính chủ","nội thất đầy đủ",
]

def extract_nlp(text: str) -> dict[str, int]:
    t = (text or "").lower()
    return {kw: int(bool(re.search(re.escape(kw), t))) for kw in NLP_KEYWORDS}

# ══════════════════════════════════════════════════════════
# 7. ML INFERENCE HELPER
# ══════════════════════════════════════════════════════════
def build_inference_row(data: HouseInput, nlp: dict) -> dict:
    """Tạo dict features khớp với feature_columns của model."""
    row = {
        "Dien_tich":          data.Dien_tich,
        "So_phong":           data.So_phong,
        "So_tang":            data.So_tang,
        "Mat_tien_m":         data.Mat_tien_m,
        "dien_tich_x_phong":  data.Dien_tich * data.So_phong,
        "dien_tich_x_tang":   data.Dien_tich * data.So_tang,
        "mat_tien_x_tang":    data.Mat_tien_m * data.So_tang,
        **nlp,
    }
    # One-Hot Vi_tri
    for d in VALID_DISTRICTS:
        row[f"Vi_tri_{d}"] = int(data.Vi_tri == d)
    # One-Hot Huong_nha
    huong_list = ["Đông","Tây","Nam","Bắc","Đông Nam","Đông Bắc","Tây Nam","Tây Bắc"]
    for h in huong_list:
        row[f"Huong_nha_{h}"] = int(data.Huong_nha == h)
    return row


def predict_ml(data: HouseInput, nlp: dict) -> tuple[float, str]:
    """Dự báo bằng ML model (struct + TF-IDF). Trả về (giá, tên phương pháp)."""
    feature_cols: list[str] = model_data["feature_columns"]
    model        = model_data["model"]
    scaler       = model_data["scaler"]
    tfidf_vec    = model_data["tfidf_vec"]
    model_name   = model_data.get("model_name", "ML")

    # Struct features
    row = build_inference_row(data, nlp)
    X_struct = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0).values

    # TF-IDF features
    X_tfidf = tfidf_vec.transform([data.Noi_dung_mo_ta]).toarray()

    # Ghép
    X_full = np.hstack([X_struct, X_tfidf])

    # Scale + predict
    X_scaled = scaler.transform(X_full)
    price = float(model.predict(X_scaled)[0])
    return price, f"ML – {model_name}"

# ══════════════════════════════════════════════════════════
# 8. HEURISTIC FALLBACK
# ══════════════════════════════════════════════════════════
BASE_PRICE = {
    "Quan_1":0.30,"Quan_2":0.22,"Quan_3":0.25,"Quan_4":0.14,"Quan_5":0.18,
    "Quan_6":0.10,"Quan_7":0.17,"Quan_8":0.09,"Quan_10":0.16,"Quan_11":0.12,
    "Quan_12":0.08,"Quan_Binh_Thanh":0.14,"Quan_Tan_Binh":0.13,
    "Quan_Phu_Nhuan":0.17,"Quan_Go_Vap":0.09,"Quan_Binh_Tan":0.07,
    "Quan_Thu_Duc":0.08,"Huyen_Binh_Chanh":0.04,"Huyen_Nha_Be":0.045,
    "Huyen_Hoc_Mon":0.035,
}
HUONG_BONUS = {"Nam":0.07,"Đông Nam":0.08,"Đông":0.05,"Đông Bắc":0.03,
               "Bắc":0.00,"Tây Bắc":0.01,"Tây":0.02,"Tây Nam":0.02}

def heuristic_price(data: HouseInput, nlp: dict) -> float:
    pm2 = BASE_PRICE.get(data.Vi_tri, 0.10)
    p = data.Dien_tich * pm2
    p *= (1 + HUONG_BONUS.get(data.Huong_nha, 0))
    p += (data.So_tang - 1) * 0.8
    p += max(0, (data.Mat_tien_m - 4)) * 0.5
    if nlp.get("mặt tiền"):         p *= 1.42
    if nlp.get("biệt thự"):         p *= 1.55
    if nlp.get("nội thất cao cấp"): p *= 1.10
    if nlp.get("sổ hồng"):          p *= 1.05
    if nlp.get("hầm xe"):           p *= 1.12
    if nlp.get("nở hậu"):           p *= 1.08
    p += data.So_phong * 0.2
    return p

# ══════════════════════════════════════════════════════════
# 9. ORS KEY
# ══════════════════════════════════════════════════════════
ORS_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQ1OWJmM2MxNGU2ZDRmOTJhYWFhYTNkZWNmYmMzZmJjIiwiaCI6Im11cm11cjY0In0="
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

# ══════════════════════════════════════════════════════════
# 10. ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": bool(model_data),
        "model_name":   model_data.get("model_name", "none"),
        "version": "3.0.0",
    }

# ── PREDICT ──────────────────────────────────────────────
@app.post("/predict", tags=["Dự báo"])
async def predict_price(data: HouseInput):
    nlp = extract_nlp(data.Noi_dung_mo_ta)
    has_mat_tien = bool(nlp.get("mặt tiền"))
    has_so_hong  = bool(nlp.get("sổ hồng"))

    if model_data:
        try:
            price, method = predict_ml(data, nlp)
        except Exception as e:
            logger.error(f"ML predict error: {e}")
            price, method = heuristic_price(data, nlp), "Heuristic (model lỗi)"
    else:
        price, method = heuristic_price(data, nlp), "Heuristic (chưa có model)"

    price = max(0.5, round(price, 2))

    km, path = dijkstra(DISTRICT_GRAPH, data.Vi_tri, "Quan_1")
    km = round(km, 1)
    path_display = " ➔ ".join(n.replace("_"," ") for n in path)
    liq = decision_tree_liquidity(data.Dien_tich, km, has_mat_tien, has_so_hong)

    return {
        "status": "success",
        "du_bao": {"gia_ty_vnd": price, "phuong_phap": method},
        "vi_tri": {
            "quan": data.Vi_tri.replace("_"," "),
            "khoang_cach_quan1_km": km,
            "duong_di": path_display,
        },
        "dac_trung_nlp": {k: v for k, v in nlp.items() if v},
        "thanh_khoan": liq,
    }

# ── ORS PROXY ────────────────────────────────────────────
@app.post("/ors-route", tags=["GIS"])
async def ors_proxy(req: ORSRouteRequest):
    """
    Proxy tới OpenRouteService Directions API.
    Frontend gọi endpoint này thay vì ORS trực tiếp → giải quyết CORS.
    """
    payload = {
        "coordinates": [
            [req.from_lng, req.from_lat],
            [req.to_lng,   req.to_lat],
        ],
        "instructions":      True,
        "geometry":          True,
        "geometry_simplify": False,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                ORS_URL,
                json=payload,
                headers={
                    "Authorization": ORS_KEY,
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
            )
        if resp.status_code != 200:
            err_text = resp.text[:300]
            logger.error(f"ORS error {resp.status_code}: {err_text}")
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"ORS lỗi {resp.status_code}: {err_text}",
            )
        return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ORS timeout – thử lại sau.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Không kết nối được ORS: {e}")

# ── RECOMMEND ─────────────────────────────────────────────
@app.post("/recommend", tags=["Gợi ý"])
def recommend_properties(req: RecommendRequest):
    try:
        df = pd.read_csv("nha_dat_full.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("nha_dat.csv")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Không tìm thấy file CSV dữ liệu.")

    feature_cols = ["Dien_tich","So_phong","Gia_ty_vnd"]
    df_feat = df[feature_cols].copy()
    input_vec = pd.DataFrame([{"Dien_tich": req.Dien_tich, "So_phong": req.So_phong, "Gia_ty_vnd": req.Gia}])

    sc = StandardScaler()
    sc.fit(df_feat)
    sims = cosine_similarity(sc.transform(input_vec), sc.transform(df_feat))[0]

    results = []
    for idx in sims.argsort()[::-1]:
        score = round(float(sims[idx]) * 100, 1)
        if score >= 99.9: continue
        row = df.iloc[idx]
        results.append({
            "id":           int(row.get("id", idx+1)),
            "dien_tich_m2": float(row["Dien_tich"]),
            "so_phong":     int(row["So_phong"]),
            "vi_tri":       str(row.get("Vi_tri","?")).replace("_"," "),
            "gia_ban_ty":   float(row["Gia_ty_vnd"]),
            "mo_ta":        str(row.get("Noi_dung_mo_ta",""))[:120],
            "do_tuong_dong_pct": score,
        })
        if len(results) == 5: break

    return {"total_found": len(results), "recommendations": results}

# ── MARKET ANALYSIS ───────────────────────────────────────
@app.post("/market-analysis", tags=["Phân tích"])
def market_analysis(req: MarketQueryRequest):
    try:
        df = pd.read_csv("nha_dat_full.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("nha_dat.csv")
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Không tìm thấy file CSV.")

    if req.Vi_tri:
        df = df[df["Vi_tri"] == req.Vi_tri]
    if req.min_price is not None:
        df = df[df["Gia_ty_vnd"] >= req.min_price]
    if req.max_price is not None:
        df = df[df["Gia_ty_vnd"] <= req.max_price]
    if df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu khớp bộ lọc.")

    by_district = (
        df.groupby("Vi_tri")["Gia_ty_vnd"]
        .agg(["mean","min","max","count"]).round(2).reset_index()
        .rename(columns={"mean":"gia_tb","min":"gia_min","max":"gia_max","count":"so_tin"})
        .sort_values("gia_tb", ascending=False)
        .to_dict(orient="records")
    )
    return {
        "tong_tin":          int(len(df)),
        "gia_trung_binh":    round(float(df["Gia_ty_vnd"].mean()), 2),
        "gia_trung_vi":      round(float(df["Gia_ty_vnd"].median()), 2),
        "gia_thap_nhat":     round(float(df["Gia_ty_vnd"].min()), 2),
        "gia_cao_nhat":      round(float(df["Gia_ty_vnd"].max()), 2),
        "dien_tich_tb_m2":   round(float(df["Dien_tich"].mean()), 1),
        "phan_tich_theo_quan": by_district,
    }

# ── DIJKSTRA ROUTE (locations) ────────────────────────────
LOCATIONS = {
    "nha":        {"name":"Bất động sản hiện tại","lat":10.7769,"lng":106.7009},
    "cong_vien":  {"name":"Công viên 23/9",        "lat":10.7684,"lng":106.6943},
    "sieu_thi":   {"name":"Siêu thị Co.opmart",    "lat":10.7705,"lng":106.6876},
    "benh_vien":  {"name":"Bệnh viện Từ Dũ",       "lat":10.7634,"lng":106.6841},
    "truong_hoc": {"name":"ĐH Sư Phạm TP.HCM",    "lat":10.7615,"lng":106.6822},
    "cho":        {"name":"Chợ Bến Thành",         "lat":10.7726,"lng":106.6980},
    "ga_tau":     {"name":"Ga Sài Gòn",            "lat":10.7815,"lng":106.6822},
}
GRAPH_MAP = {
    "nha":       {"cong_vien":1.2,"sieu_thi":2.0,"cho":0.9},
    "cong_vien": {"nha":1.2,"sieu_thi":0.8,"benh_vien":1.5,"cho":0.6},
    "sieu_thi":  {"nha":2.0,"cong_vien":0.8,"benh_vien":1.0,"truong_hoc":1.8},
    "benh_vien": {"cong_vien":1.5,"sieu_thi":1.0,"truong_hoc":0.5},
    "truong_hoc":{"sieu_thi":1.8,"benh_vien":0.5},
    "cho":       {"nha":0.9,"cong_vien":0.6,"ga_tau":1.1},
    "ga_tau":    {"cho":1.1,"sieu_thi":1.8},
}

@app.get("/locations", tags=["GIS"])
def list_locations():
    return {"locations": {k: v["name"] for k,v in LOCATIONS.items()}}

@app.post("/route", tags=["GIS"])
def find_route_dijkstra(req: RouteRequest):
    if req.start not in GRAPH_MAP or req.end not in GRAPH_MAP:
        raise HTTPException(status_code=400, detail=f"Điểm không hợp lệ: {list(GRAPH_MAP)}")
    dist, path = dijkstra(GRAPH_MAP, req.start, req.end)
    if not path:
        raise HTTPException(status_code=404, detail="Không tìm được đường đi.")
    return {
        "path_keys":         path,
        "path_names":        [LOCATIONS[n]["name"] for n in path],
        "total_distance_km": round(dist, 2),
        "coordinates":       [LOCATIONS[n] for n in path],
        "estimated_time_min":round(dist/30*60, 0),
    }

# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)