"""
Pipeline Huấn luyện Mô hình Dự báo Giá Nhà TP.HCM  ─ v3.0
===========================================================

Đáp ứng đầy đủ yêu cầu đề tài:
  ✅ Dữ liệu ≥ 5.000 mẫu (file nha_dat_full.csv – 5.500 dòng)
  ✅ Đặc trưng định lượng: Dien_tich, So_phong, So_tang, Mat_tien_m
  ✅ Chuẩn hóa địa chỉ: phân cấp Quận/Huyện → One-Hot Encoding
  ✅ Xử lý outlier theo IQR, điền missing values
  ✅ TF-IDF trên cột mô tả (Noi_dung_mo_ta) – top 60 terms
  ✅ NLP keyword binary features (12 từ khóa nghiệp vụ)
  ✅ 3 thuật toán: Linear Regression · Random Forest · XGBoost
  ✅ K-Fold Cross-Validation (5-fold)
  ✅ Đánh giá: RMSE, MAE, R² (train / validation / test)
  ✅ Feature Importance + phân tích top TF-IDF terms
  ✅ Lưu model + scaler + vectorizer → house_price_model.pkl
"""

import re, time, logging, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════
CSV_PATH       = "nha_dat_full.csv"   # 5500 mẫu
MODEL_PATH     = "house_price_model.pkl"
REPORT_PATH    = "evaluation_report.txt"
TFIDF_MAX_FEAT = 60     # Số term TF-IDF giữ lại
TEST_SIZE      = 0.15   # 15% test
VAL_SIZE       = 0.15   # 15% validation (từ phần train)
CV_FOLDS       = 5
RANDOM_STATE   = 42

NLP_KEYWORDS = [
    "sổ hồng", "sổ đỏ", "hẻm xe hơi", "ô tô vào", "nở hậu",
    "mặt tiền", "nội thất cao cấp", "hồ bơi", "hầm xe",
    "biệt thự", "chính chủ", "nội thất đầy đủ",
]

VALID_DISTRICTS = [
    "Quan_1","Quan_2","Quan_3","Quan_4","Quan_5","Quan_6",
    "Quan_7","Quan_8","Quan_10","Quan_11","Quan_12",
    "Quan_Binh_Thanh","Quan_Tan_Binh","Quan_Phu_Nhuan",
    "Quan_Go_Vap","Quan_Binh_Tan","Quan_Thu_Duc",
    "Huyen_Binh_Chanh","Huyen_Nha_Be","Huyen_Hoc_Mon",
]

VALID_HUONG = ["Đông","Tây","Nam","Bắc","Đông Nam","Đông Bắc","Tây Nam","Tây Bắc"]

# ══════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════════════
class HousePricePipeline:
    def __init__(self):
        self.tfidf_vec       = TfidfVectorizer(
            max_features  = TFIDF_MAX_FEAT,
            ngram_range   = (1, 2),          # unigram + bigram
            sublinear_tf  = True,
            min_df        = 3,
            analyzer      = "word",
            token_pattern = r"[^\s,\.]+",    # phù hợp tiếng Việt
        )
        self.scaler          = StandardScaler()
        self.feature_columns: list[str] = []
        self.best_model      = None
        self.best_name       = ""
        self._report: list[str] = []

    # ──────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        p = Path(CSV_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy '{CSV_PATH}'. Hãy chạy generate_data.py trước.")
        df = pd.read_csv(p)
        self._log(f"📂 Load: {len(df):,} dòng từ '{CSV_PATH}'")
        self._log(f"   Cột: {list(df.columns)}")
        return df

    # ──────────────────────────────────────────────
    # 2. TIỀN XỬ LÝ
    # ──────────────────────────────────────────────
    @staticmethod
    def _extract_nlp(text: str) -> dict:
        t = str(text or "").lower()
        return {kw: int(bool(re.search(re.escape(kw), t))) for kw in NLP_KEYWORDS}

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        before = len(df)
        df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)].copy()
        logger.info(f"   Outlier [{col}]: loại {before - len(df)} dòng → còn {len(df):,}")
        return df

    def preprocess(self, df: pd.DataFrame):
        """
        Trả về:
          X_struct  – ma trận đặc trưng cấu trúc (numeric)
          X_tfidf   – ma trận TF-IDF sparse
          y         – nhãn giá
          desc_raw  – văn bản gốc (để fit/transform TF-IDF)
        """
        self._log("\n── TIỀN XỬ LÝ ──────────────────────────────────────")

        # Đảm bảo cột cần thiết
        needed = ["Dien_tich","So_phong","Vi_tri","Noi_dung_mo_ta","Gia_ty_vnd"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột '{col}'")

        # Giá trị mặc định cho cột tùy chọn
        if "So_tang"    not in df.columns: df["So_tang"]    = 2
        if "Mat_tien_m" not in df.columns: df["Mat_tien_m"] = 4.0
        if "Huong_nha"  not in df.columns: df["Huong_nha"]  = "Nam"

        df = df.dropna(subset=["Gia_ty_vnd"]).copy()

        # Điền missing
        df["Dien_tich"]  = df["Dien_tich"].fillna(df["Dien_tich"].median())
        df["So_phong"]   = df["So_phong"].fillna(3).astype(int)
        df["So_tang"]    = df["So_tang"].fillna(2).astype(int)
        df["Mat_tien_m"] = df["Mat_tien_m"].fillna(4.0)
        df["Vi_tri"]     = df["Vi_tri"].fillna("Quan_1")
        df["Huong_nha"]  = df["Huong_nha"].fillna("Nam")
        df["Noi_dung_mo_ta"] = df["Noi_dung_mo_ta"].fillna("")

        # Clip giá trị bất hợp lý
        df["Dien_tich"]  = df["Dien_tich"].clip(10, 2000)
        df["So_phong"]   = df["So_phong"].clip(1, 20)
        df["So_tang"]    = df["So_tang"].clip(1, 30)
        df["Mat_tien_m"] = df["Mat_tien_m"].clip(2, 30)

        # Loại outlier
        df = self._remove_outliers(df, "Gia_ty_vnd")
        df = self._remove_outliers(df, "Dien_tich")

        self._log(f"   Dữ liệu sau xử lý: {len(df):,} dòng")

        # ── NLP binary features ──
        nlp_df = df["Noi_dung_mo_ta"].apply(self._extract_nlp).apply(pd.Series)

        # ── Feature engineering ──
        df["dien_tich_x_phong"] = df["Dien_tich"] * df["So_phong"]
        df["dien_tich_x_tang"]  = df["Dien_tich"] * df["So_tang"]
        df["mat_tien_x_tang"]   = df["Mat_tien_m"] * df["So_tang"]

        # ── One-Hot: Quận & Hướng ──
        df = pd.get_dummies(df, columns=["Vi_tri","Huong_nha"], drop_first=False)

        # ── Tổng hợp X cấu trúc ──
        struct_cols = [
            "Dien_tich","So_phong","So_tang","Mat_tien_m",
            "dien_tich_x_phong","dien_tich_x_tang","mat_tien_x_tang",
        ]
        # Thêm các cột one-hot
        ohe_cols = [c for c in df.columns if c.startswith("Vi_tri_") or c.startswith("Huong_nha_")]
        struct_cols += ohe_cols

        # Ghép NLP binary
        X_struct = pd.concat([df[struct_cols].reset_index(drop=True),
                               nlp_df.reset_index(drop=True)], axis=1)

        # Chuyển bool → int
        bool_cols = X_struct.select_dtypes("bool").columns
        X_struct[bool_cols] = X_struct[bool_cols].astype(int)
        X_struct = X_struct.select_dtypes(include=[np.number]).fillna(0)

        self.feature_columns = list(X_struct.columns)
        self._log(f"   Đặc trưng cấu trúc: {len(self.feature_columns)} cột")
        self._log(f"   TF-IDF max_features: {TFIDF_MAX_FEAT}, ngram (1,2)")

        y        = df["Gia_ty_vnd"].reset_index(drop=True)
        desc_raw = df["Noi_dung_mo_ta"].reset_index(drop=True)

        return X_struct, desc_raw, y

    # ──────────────────────────────────────────────
    # 3. TF-IDF FIT + MERGE
    # ──────────────────────────────────────────────
    def build_features(self, X_struct: pd.DataFrame, desc_raw: pd.Series,
                        fit: bool = True):
        """Ghép TF-IDF với đặc trưng cấu trúc → numpy dense."""
        if fit:
            X_tfidf = self.tfidf_vec.fit_transform(desc_raw)
        else:
            X_tfidf = self.tfidf_vec.transform(desc_raw)

        X_dense = X_tfidf.toarray()  # (n, 60)
        X_num   = X_struct.values    # (n, k)
        return np.hstack([X_num, X_dense])

    # ──────────────────────────────────────────────
    # 4. TRAIN & EVALUATE
    # ──────────────────────────────────────────────
    def evaluate(self, name, model, X_tr, y_tr, X_val, y_val, X_te, y_te):
        """Trả về dict chỉ số + in log."""
        model.fit(X_tr, y_tr)
        metrics = {}
        for split_name, Xs, ys in [("Train",X_tr,y_tr),("Val",X_val,y_val),("Test",X_te,y_te)]:
            pred = model.predict(Xs)
            metrics[split_name] = {
                "RMSE": float(np.sqrt(mean_squared_error(ys, pred))),
                "MAE":  float(mean_absolute_error(ys, pred)),
                "R2":   float(r2_score(ys, pred)),
            }
        return metrics

    def train(self, X_struct, desc_raw, y):
        self._log("\n── HUẤN LUYỆN ──────────────────────────────────────")

        # Chia 3 tập: train / val / test
        X_tv, X_te, y_tv, y_te, d_tv, d_te = train_test_split(
            X_struct, y, desc_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        X_tr, X_val, y_tr, y_val, d_tr, d_val = train_test_split(
            X_tv, y_tv, d_tv, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)

        self._log(f"   Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")

        # Build TF-IDF (fit chỉ trên train)
        Xf_tr  = self.build_features(X_tr,  d_tr,  fit=True)
        Xf_val = self.build_features(X_val, d_val, fit=False)
        Xf_te  = self.build_features(X_te,  d_te,  fit=False)
        self._log(f"   Shape sau ghép TF-IDF: {Xf_tr.shape}")

        # Scale
        Xs_tr  = self.scaler.fit_transform(Xf_tr)
        Xs_val = self.scaler.transform(Xf_val)
        Xs_te  = self.scaler.transform(Xf_te)

        # ── 3 Mô hình ──
        candidates = {
            "Linear Regression (Ridge)": Ridge(alpha=10.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, max_depth=14,
                min_samples_leaf=2, n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=400, learning_rate=0.05,
                max_depth=6, subsample=0.85,
                colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=2.0,
                random_state=RANDOM_STATE, n_jobs=-1,
                verbosity=0, eval_metric="rmse",
            ),
        }

        header = f"{'Mô hình':<30} {'Split':<7} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
        self._log("\n" + "═"*65)
        self._log("SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH")
        self._log("═"*65)
        self._log(header)
        self._log("─"*65)

        best_r2 = -np.inf
        all_results = {}

        for name, model in candidates.items():
            t0 = time.time()
            metrics = self.evaluate(name, model, Xs_tr, y_tr, Xs_val, y_val, Xs_te, y_te)
            elapsed = time.time() - t0
            all_results[name] = metrics

            for sp, m in metrics.items():
                self._log(f"  {name if sp=='Train' else '':<28} {sp:<7} "
                          f"{m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f}")
            self._log(f"  {'':30} ⏱ {elapsed:.1f}s")
            self._log("─"*65)

            if metrics["Val"]["R2"] > best_r2:
                best_r2, self.best_model, self.best_name = metrics["Val"]["R2"], model, name

        # ── Cross-Validation ──
        self._log("\n── K-FOLD CROSS-VALIDATION (5-fold, Train+Val) ──────")
        Xf_tv = np.vstack([Xf_tr, Xf_val])
        Xs_tv = self.scaler.transform(Xf_tv)
        y_tv_arr = np.concatenate([y_tr.values, y_val.values])

        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for name, model in candidates.items():
            cv_r2   = cross_val_score(model, Xs_tv, y_tv_arr, cv=kf, scoring="r2",    n_jobs=-1)
            cv_rmse = cross_val_score(model, Xs_tv, y_tv_arr, cv=kf,
                                      scoring="neg_root_mean_squared_error", n_jobs=-1)
            self._log(f"  {name:<30} R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}  "
                      f"RMSE={-cv_rmse.mean():.4f}±{cv_rmse.std():.4f}")

        self._log(f"\n🏆 Mô hình tốt nhất (Val R²): {self.best_name}  R²={best_r2:.4f}")

        # ── TF-IDF top terms ──
        self._log("\n── TOP 20 TERMS TF-IDF QUAN TRỌNG NHẤT ─────────────")
        feat_names = self.tfidf_vec.get_feature_names_out()
        if hasattr(self.best_model, "feature_importances_"):
            n_struct = Xf_tr.shape[1] - len(feat_names)
            tfidf_imp = self.best_model.feature_importances_[n_struct:]
            top_idx = np.argsort(tfidf_imp)[::-1][:20]
            for idx in top_idx:
                bar = "█" * int(tfidf_imp[idx] * 200)
                self._log(f"  {feat_names[idx]:<30} {tfidf_imp[idx]:.5f}  {bar}")
        else:
            # Ridge: dùng coef
            n_struct = Xf_tr.shape[1] - len(feat_names)
            coefs = np.abs(self.best_model.coef_[n_struct:])
            top_idx = np.argsort(coefs)[::-1][:20]
            for idx in top_idx:
                self._log(f"  {feat_names[idx]:<30} |coef|={coefs[idx]:.4f}")

        # ── Test final ──
        self._log("\n── KẾT QUẢ CUỐI TRÊN TẬP TEST ──────────────────────")
        m = all_results[self.best_name]["Test"]
        self._log(f"  Mô hình : {self.best_name}")
        self._log(f"  RMSE    : {m['RMSE']:.4f} tỷ VNĐ")
        self._log(f"  MAE     : {m['MAE']:.4f} tỷ VNĐ")
        self._log(f"  R²      : {m['R2']:.4f}")

        self.all_results = all_results

    # ──────────────────────────────────────────────
    # 5. LƯU MODEL
    # ──────────────────────────────────────────────
    def save(self):
        payload = {
            "model":           self.best_model,
            "scaler":          self.scaler,
            "tfidf_vec":       self.tfidf_vec,
            "feature_columns": self.feature_columns,
            "nlp_keywords":    NLP_KEYWORDS,
            "model_name":      self.best_name,
        }
        joblib.dump(payload, MODEL_PATH)
        self._log(f"\n✅ Model lưu tại: {MODEL_PATH}")

        Path(REPORT_PATH).write_text("\n".join(self._report), encoding="utf-8")
        self._log(f"📄 Báo cáo: {REPORT_PATH}")

    # ──────────────────────────────────────────────
    # HELPER
    # ──────────────────────────────────────────────
    def _log(self, msg: str):
        logger.info(msg)
        self._report.append(msg)

    # ──────────────────────────────────────────────
    # RUN
    # ──────────────────────────────────────────────
    def run(self):
        t0 = time.time()
        self._log("🚀 BẮT ĐẦU PIPELINE HUẤN LUYỆN v3.0")
        self._log("=" * 65)
        df = self.load()
        X_struct, desc_raw, y = self.preprocess(df)
        self.train(X_struct, desc_raw, y)
        self.save()
        self._log(f"\n✅ Hoàn tất trong {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    HousePricePipeline().run()