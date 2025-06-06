import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import gaussian_kde, entropy, ks_2samp, wasserstein_distance

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import logging

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Make sure DataManager is importable. If running in Colab, you can either just put the DataManager code in a cell above or do:
# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# sys.path.append('/content/drive/MyDrive/your_project_folder')
# from data_manager import DataManager

from src.data_management.data_manager import DataManager

class Report:
    def __init__(
        self,
        ref_df: pd.DataFrame,
        cmp_df: pd.DataFrame,
        common_columns: list,
        ref_schema: dict,
        cmp_schema: dict
    ):
        logger.info("Initializing Report object.")
        self.ref_df = ref_df.loc[:, common_columns].reset_index(drop=True)
        self.cmp_df = cmp_df.loc[:, common_columns].reset_index(drop=True)
        self.common_columns = common_columns
        self.ref_schema = ref_schema
        self.cmp_schema = cmp_schema

        logger.info(f"Common columns: {common_columns}")
        # Determine effective column types using schema + heuristics
        self.column_types = {}
        for col in common_columns:
            declared = ref_schema.get(col)
            logger.info(f"Determining type for column '{col}' (declared: {declared}).")
            if declared == 'category':
                self.column_types[col] = 'categorical'
            elif declared in (bool, 'boolean'):
                self.column_types[col] = 'categorical'
            elif declared in (int, 'int', float, 'float', 'decimal'):
                self.column_types[col] = 'numeric'
            elif declared in (pd.Timestamp, 'datetime', 'datetime64[ns]'):
                self.column_types[col] = 'datetime'
            elif declared in (str, 'string', 'text'):
                non_na = self.ref_df[col].dropna().astype(str)
                if len(non_na) == 0:
                    self.column_types[col] = 'categorical'
                else:
                    unique_ratio = non_na.nunique() / len(non_na)
                    mean_tokens = non_na.str.split().apply(len).mean()
                    if unique_ratio <= 0.20 and mean_tokens <= 5:
                        self.column_types[col] = 'categorical'
                    else:
                        self.column_types[col] = 'text'
            else:
                dtype = self.ref_df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    self.column_types[col] = 'numeric'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    self.column_types[col] = 'datetime'
                else:
                    self.column_types[col] = 'categorical'
            logger.info(f"  → Column '{col}' determined to be '{self.column_types[col]}'.")

        # Placeholder for per‐column metrics
        self.per_column_metrics = {col: {} for col in common_columns}
        logger.info("Report initialization complete.")


    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        logger.info("Computing Jensen–Shannon divergence.")
        p = p + 1e-12
        q = q + 1e-12
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        result = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
        logger.info(f"  → JS divergence = {result:.6f}")
        return result


    def numeric_ks_similarity(self, col: str) -> float:
        logger.info(f"Computing KS similarity for numeric column '{col}'.")
        real_vals = self.ref_df[col].dropna().astype(float).values
        syn_vals  = self.cmp_df[col].dropna().astype(float).values
        logger.info(f"  → Samples: real={real_vals.size}, syn={syn_vals.size}")
        if real_vals.size < 2 or syn_vals.size < 2:
            logger.info("  → Not enough data points for KS: returning NaN.")
            return np.nan
        D_stat, _ = ks_2samp(real_vals, syn_vals)
        sim = 1.0 - D_stat
        logger.info(f"  → KS D-statistic = {D_stat:.6f}, similarity = {sim:.6f}")
        return sim


    def numeric_wasserstein(self, col: str) -> float:
        logger.info(f"Computing Wasserstein distance for numeric column '{col}'.")
        real_vals = self.ref_df[col].dropna().astype(float).values
        syn_vals  = self.cmp_df[col].dropna().astype(float).values
        logger.info(f"  → Samples: real={real_vals.size}, syn={syn_vals.size}")
        if real_vals.size < 2 or syn_vals.size < 2:
            logger.info("  → Not enough data points for Wasserstein: returning NaN.")
            return np.nan, np.nan

        Wraw = wasserstein_distance(real_vals, syn_vals)
        logger.info(f"  → Raw Wasserstein = {Wraw:.6f}")
        rmin, rmax = real_vals.min(), real_vals.max()
        rng = rmax - rmin
        if rng == 0:
            logger.info("  → Real range is zero; normalized Wasserstein is NaN.")
            return Wraw, np.nan

        Wnorm = Wraw / rng
        logger.info(f"  → Normalized Wasserstein = {Wnorm:.6f}")
        return Wraw, Wnorm


    def numeric_summary_stats(self, col: str) -> dict:
        logger.info(f"Computing detailed summary‐stats for numeric column '{col}'.")
        real_vals = self.ref_df[col].dropna().astype(float)
        syn_vals  = self.cmp_df[col].dropna().astype(float)
        if real_vals.empty or syn_vals.empty:
            logger.info("  → One side is empty: returning all NaN stats.")
            return {
                'mean_real': np.nan, 'mean_syn': np.nan, 'mean_diff': np.nan, 'mean_pct_of_range': np.nan,
                'median_real': np.nan, 'median_syn': np.nan, 'median_diff': np.nan, 'median_pct_of_range': np.nan,
                'std_real': np.nan, 'std_syn': np.nan, 'std_diff': np.nan, 'std_pct_of_std': np.nan
            }

        rmin, rmax = real_vals.min(), real_vals.max()
        data_range = rmax - rmin
        logger.info(f"  → Real min={rmin:.4f}, max={rmax:.4f}, range={data_range:.4f}")

        mean_r = real_vals.mean()
        mean_s = syn_vals.mean()
        diff_mean = mean_s - mean_r
        pct_mean = diff_mean / data_range if data_range != 0 else np.nan
        logger.info(f"  → Mean: real={mean_r:.4f}, syn={mean_s:.4f}, diff={diff_mean:.4f}, pct_range={pct_mean:.6f}")

        med_r = real_vals.median()
        med_s = syn_vals.median()
        diff_med = med_s - med_r
        pct_med = diff_med / data_range if data_range != 0 else np.nan
        logger.info(f"  → Median: real={med_r:.4f}, syn={med_s:.4f}, diff={diff_med:.4f}, pct_range={pct_med:.6f}")

        std_r = real_vals.std(ddof=1)
        std_s = syn_vals.std(ddof=1)
        diff_std = std_s - std_r
        pct_std = diff_std / std_r if std_r != 0 else np.nan
        logger.info(f"  → StdDev: real={std_r:.4f}, syn={std_s:.4f}, diff={diff_std:.4f}, pct_std={pct_std:.6f}")

        return {
            'mean_real': mean_r,
            'mean_syn': mean_s,
            'mean_diff': diff_mean,
            'mean_pct_of_range': pct_mean,
            'median_real': med_r,
            'median_syn': med_s,
            'median_diff': diff_med,
            'median_pct_of_range': pct_med,
            'std_real': std_r,
            'std_syn': std_s,
            'std_diff': diff_std,
            'std_pct_of_std': pct_std
        }


    def numeric_range_coverage(self, col: str) -> float:
        logger.info(f"Computing range coverage for numeric column '{col}'.")
        real_vals = self.ref_df[col].dropna().astype(float)
        syn_vals  = self.cmp_df[col].dropna().astype(float)
        if real_vals.empty or syn_vals.empty:
            logger.info("  → One side is empty: returning NaN.")
            return np.nan
        rmin, rmax = real_vals.min(), real_vals.max()
        rng = rmax - rmin
        if rng == 0:
            logger.info("  → Real range is zero: returning NaN.")
            return np.nan
        smin, smax = syn_vals.min(), syn_vals.max()
        srng = smax - smin
        cov = min(srng / rng, 1.0)
        logger.info(f"  → Synthetic range={srng:.4f}, coverage={cov:.6f}")
        return cov


    # ───── Univariate Methods (unchanged from earlier) ──────────────────────────

    def _compute_numeric_histogram_js(self, col: str) -> float:
        logger.info(f"Computing histogram‐based JS divergence for '{col}'.")
        ref_vals = self.ref_df[col].dropna().astype(float).values
        cmp_vals = self.cmp_df[col].dropna().astype(float).values
        if len(ref_vals) < 2 or len(cmp_vals) < 2:
            logger.info("  → Not enough data for histogram JS: returning NaN.")
            return np.nan

        n_ref = len(ref_vals)
        s_ref = np.std(ref_vals, ddof=1)
        h = 3.49 * s_ref * (n_ref ** (-1 / 3))
        if h <= 0:
            k = int(np.ceil(np.log2(n_ref) + 1))
        else:
            data_min = min(ref_vals.min(), cmp_vals.min())
            data_max = max(ref_vals.max(), cmp_vals.max())
            k = int(np.ceil((data_max - data_min) / h))
            if k < 2:
                k = int(np.ceil(np.log2(n_ref) + 1))
        k = max(min(k, 100), 2)
        logger.info(f"  → Number of bins for histogram: k={k}")

        data_min = min(ref_vals.min(), cmp_vals.min())
        data_max = max(ref_vals.max(), cmp_vals.max())
        bins = np.linspace(data_min, data_max, k + 1)

        p_ref, _ = np.histogram(ref_vals, bins=bins, density=True)
        p_cmp, _ = np.histogram(cmp_vals, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        p_ref_probs = p_ref * bin_width
        p_cmp_probs = p_cmp * bin_width

        js = Report._js_divergence(p_ref_probs, p_cmp_probs)
        return js


    def _compute_numeric_kde_metrics(self, col: str) -> (float, float):
        logger.info(f"Computing KDE‐based L2 & JS metrics for '{col}'.")
        ref_vals = self.ref_df[col].dropna().astype(float).values
        cmp_vals = self.cmp_df[col].dropna().astype(float).values
        if len(ref_vals) < 2 or len(cmp_vals) < 2:
            logger.info("  → Not enough data for KDE: returning (NaN, NaN).")
            return np.nan, np.nan

        pooled = np.concatenate([ref_vals, cmp_vals])
        bw = np.std(pooled, ddof=1) * (len(pooled) ** (-1/5))
        if bw <= 0:
            bw = 1.0
        logger.info(f"  → Bandwidth for KDE (Silverman estimate) = {bw:.6f}")

        try:
            kde_ref = gaussian_kde(ref_vals, bw_method=bw / np.std(ref_vals, ddof=1))
            kde_cmp = gaussian_kde(cmp_vals, bw_method=bw / np.std(cmp_vals, ddof=1))
        except np.linalg.LinAlgError:
            logger.info("  → KDE failed (singular covariance): returning (NaN, NaN).")
            return np.nan, np.nan

        data_min = pooled.min()
        data_max = pooled.max()
        grid = np.linspace(data_min, data_max, 512)
        pdf_ref = kde_ref(grid)
        pdf_cmp = kde_cmp(grid)

        dx = grid[1] - grid[0]
        pdf_ref_norm = pdf_ref / (np.trapz(pdf_ref, grid))
        pdf_cmp_norm = pdf_cmp / (np.trapz(pdf_cmp, grid))

        l2 = np.sqrt(np.trapz((pdf_ref_norm - pdf_cmp_norm) ** 2, grid))
        logger.info(f"  → KDE L2 distance = {l2:.6f}")

        p = pdf_ref_norm * dx
        q = pdf_cmp_norm * dx
        js = Report._js_divergence(p, q)
        return l2, js


    def categorical_tvd_similarity(self, col: str) -> float:
        logger.info(f"Computing TVD similarity for categorical column '{col}'.")
        real_counts = self.ref_df[col].dropna().value_counts(normalize=True)
        syn_counts  = self.cmp_df[col].dropna().value_counts(normalize=True)
        all_cats = set(real_counts.index).union(syn_counts.index)
        tvd = 0.0
        for cat in all_cats:
            p = real_counts.get(cat, 0.0)
            q = syn_counts.get(cat, 0.0)
            tvd += abs(p - q)
        tvd *= 0.5
        sim = 1.0 - tvd
        logger.info(f"  → TVD = {tvd:.6f}, similarity = {sim:.6f}")
        return sim


    def categorical_coverage(self, col: str) -> float:
        logger.info(f"Computing category coverage for '{col}'.")
        real_uniques = set(self.ref_df[col].dropna().unique())
        syn_uniques  = set(self.cmp_df[col].dropna().unique())
        if not real_uniques:
            logger.info("  → No real categories: returning NaN.")
            return np.nan
        covered = len(real_uniques.intersection(syn_uniques))
        coverage = covered / len(real_uniques)
        logger.info(f"  → Coverage = {coverage:.6f} ({covered}/{len(real_uniques)})")
        return coverage


    def _compute_categorical_js(self, col: str) -> float:
        logger.info(f"Computing JS divergence on categorical frequencies for '{col}'.")
        ref_counts = self.ref_df[col].dropna().value_counts(normalize=True)
        cmp_counts = self.cmp_df[col].dropna().value_counts(normalize=True)
        all_cats = sorted(set(ref_counts.index).union(cmp_counts.index))
        p = np.array([ref_counts.get(cat, 0.0) for cat in all_cats], dtype=float)
        q = np.array([cmp_counts.get(cat, 0.0) for cat in all_cats], dtype=float)
        js = Report._js_divergence(p, q)
        return js


    def _compute_text_token_js(self, col: str, top_k: int = 5000) -> float:
        logger.info(f"Computing token‐level JS divergence for text column '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side is empty: returning NaN.")
            return np.nan

        tokens_ref = [tok for t in texts_ref for tok in t.split()]
        tokens_cmp = [tok for t in texts_cmp for tok in t.split()]
        ctr_ref = Counter(tokens_ref)
        ctr_cmp = Counter(tokens_cmp)
        combined = ctr_ref + ctr_cmp
        most_common = [tok for tok, _ in combined.most_common(top_k)]

        p = np.array([ctr_ref.get(tok, 0) for tok in most_common], dtype=float)
        q = np.array([ctr_cmp.get(tok, 0) for tok in most_common], dtype=float)
        if p.sum() == 0 or q.sum() == 0:
            logger.info("  → Zero counts on one side: returning NaN.")
            return np.nan
        p = p / p.sum()
        q = q / q.sum()
        js = Report._js_divergence(p, q)
        return js


    def _compute_text_tfidf_cosine(self, col: str, max_features: int = 5000) -> float:
        logger.info(f"Computing TF–IDF cosine for text column '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side is empty: returning NaN.")
            return np.nan

        corpus = texts_ref + texts_cmp
        logger.info(f"  → Fitting TF–IDF on {len(corpus)} documents (max_features={max_features}).")
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', lowercase=True)
        tfidf_all = vectorizer.fit_transform(corpus)
        n_ref = len(texts_ref)
        A_ref = tfidf_all[:n_ref, :]
        A_cmp = tfidf_all[n_ref:, :]

        c_ref = np.asarray(A_ref.mean(axis=0)).ravel()
        c_cmp = np.asarray(A_cmp.mean(axis=0)).ravel()
        denom = np.linalg.norm(c_ref) * np.linalg.norm(c_cmp)
        cosine = float(np.dot(c_ref, c_cmp) / denom) if denom > 0 else np.nan
        logger.info(f"  → TF–IDF cosine = {cosine:.6f}")
        return cosine


    def _compute_text_vocab_jaccard(self, col: str) -> float:
        logger.info(f"Computing vocabulary Jaccard for text column '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side is empty: returning NaN.")
            return np.nan

        tokens_ref = set(tok for t in texts_ref for tok in t.split())
        tokens_cmp = set(tok for t in texts_cmp for tok in t.split())
        if not tokens_ref and not tokens_cmp:
            logger.info("  → Both sets empty: returning NaN.")
            return np.nan
        if not tokens_ref or not tokens_cmp:
            logger.info("  → One set empty: returning 0.0.")
            return 0.0
        inter = tokens_ref.intersection(tokens_cmp)
        union = tokens_ref.union(tokens_cmp)
        jaccard = len(inter) / len(union)
        logger.info(f"  → Vocabulary Jaccard = {jaccard:.6f}")
        return jaccard


    def _compute_document_length_diff(self, col: str) -> float:
        logger.info(f"Computing document‐length difference for text column '{col}'.")
        lens_ref = self.ref_df[col].dropna().astype(str).str.split().apply(len)
        lens_cmp = self.cmp_df[col].dropna().astype(str).str.split().apply(len)
        if lens_ref.empty or lens_cmp.empty:
            logger.info("  → One side is empty: returning NaN.")
            return np.nan
        diff = abs(lens_ref.mean() - lens_cmp.mean())
        logger.info(f"  → Avg length real = {lens_ref.mean():.2f}, syn = {lens_cmp.mean():.2f}, Δ = {diff:.2f}")
        return diff


    def _compute_text_embedding_metrics(
        self,
        col: str,
        max_samples: int = 200
    ) -> (float, float):
        """
        Subsample up to max_samples texts from each side,
        compute embedding‐cosine and MMD on those.
        """
        logger.info(f"Computing embedding metrics for text column '{col}' (max_samples={max_samples}).")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side empty: returning (NaN, NaN).")
            return np.nan, np.nan

        rng = np.random.default_rng(seed=42)
        if len(texts_ref) > max_samples:
            texts_ref = list(rng.choice(texts_ref, max_samples, replace=False))
        if len(texts_cmp) > max_samples:
            texts_cmp = list(rng.choice(texts_cmp, max_samples, replace=False))

        logger.info(f"  → Sampling: real={len(texts_ref)}, syn={len(texts_cmp)}.")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb_ref = model.encode(texts_ref, convert_to_numpy=True)
            emb_cmp = model.encode(texts_cmp, convert_to_numpy=True)
        except Exception as e:
            logger.info(f"  → Error loading/encoding embeddings: {e}. Returning (NaN, NaN).")
            return np.nan, np.nan

        cent_ref = emb_ref.mean(axis=0)
        cent_cmp = emb_cmp.mean(axis=0)
        denom = np.linalg.norm(cent_ref) * np.linalg.norm(cent_cmp)
        emb_cosine = float(np.dot(cent_ref, cent_cmp) / denom) if denom > 0 else np.nan
        logger.info(f"  → Embedding cosine = {emb_cosine:.6f}")

        combined = np.vstack([emb_ref, emb_cmp])
        pdists = np.sqrt(np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2))
        median_dist = np.median(pdists)
        gamma = 1.0 / (2 * (median_dist ** 2 + 1e-12))
        logger.info(f"  → MMD gamma parameter = {gamma:.6e}")

        def rbf_matrix(X, Y, gamma):
            dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * dists)

        K_rr = rbf_matrix(emb_ref, emb_ref, gamma)
        K_cc = rbf_matrix(emb_cmp, emb_cmp, gamma)
        K_rc = rbf_matrix(emb_ref, emb_cmp, gamma)

        m = emb_ref.shape[0]
        n = emb_cmp.shape[0]
        mmd_sq = (np.sum(K_rr) / (m * m)) - (2 * np.sum(K_rc) / (m * n)) + (np.sum(K_cc) / (n * n))
        emb_mmd = float(np.sqrt(max(mmd_sq, 0.0)))
        logger.info(f"  → Embedding MMD = {emb_mmd:.6f}")

        return emb_cosine, emb_mmd


    def compute_column(self, col: str) -> dict:
        """
        Compute per‐column metrics using the new univariate methods.
        """
        logger.info(f"compute_column called for '{col}'.")
        results = {}
        # Missing-rate difference
        mr_diff = abs(self.ref_df[col].isna().mean() - self.cmp_df[col].isna().mean())
        results['missing_rate_diff'] = mr_diff
        logger.info(f"  → Missing‐rate difference = {mr_diff:.6f}")

        ctype = self.column_types.get(col)
        logger.info(f"  → Column type = {ctype}")

        if ctype == "numeric":
            # 1) KS similarity
            results['ks_similarity'] = self.numeric_ks_similarity(col)

            # 2) Wasserstein (raw + normalized)
            Wraw, Wnorm = self.numeric_wasserstein(col)
            results['wasserstein_raw']  = Wraw
            results['wasserstein_norm'] = Wnorm

            # 3) Detailed summary‐stats
            stats = self.numeric_summary_stats(col)
            results.update(stats)

            # 4) Range coverage
            results['range_coverage'] = self.numeric_range_coverage(col)

        elif ctype == "categorical":
            # 1) TVD similarity
            results['tvd_similarity'] = self.categorical_tvd_similarity(col)
            # 2) Category coverage
            results['category_coverage'] = self.categorical_coverage(col)

        elif ctype == "text":
            # 1) Token‐JS Divergence
            tok_js   = self._compute_text_token_js(col)

            # 2) Bigram Jaccard & Bigram‐JS
            bigram_j = self._compute_text_bigram_jaccard(col)
            bigram_js = self._compute_text_bigram_js(col)

            # 3) TF–IDF Cosine
            tfidf_cos = self._compute_text_tfidf_cosine(col)

            # 4) Vocabulary Jaccard (unigrams)
            vocab_j = self._compute_text_vocab_jaccard(col)

            # 5) OOV Rate
            oov_real = self._compute_text_oov_rate(col)

            # 6) Document Length Stats
            length_stats = self._compute_text_length_stats(col)
            len_real = length_stats['len_real']
            len_syn  = length_stats['len_syn']
            len_diff = length_stats['len_diff']

            # 7) Topic‐Distribution Cosine
            topic_cos = self._compute_text_topic_cosine(col)

            # 8) Embedding Cosine & MMD
            emb_cos, emb_mmd = self._compute_text_embedding_metrics(col)

            # Populate results
            results['text_tok_js']         = tok_js
            results['text_bigram_jaccard'] = bigram_j
            results['text_bigram_js']      = bigram_js
            results['text_tfidf_cosine']   = tfidf_cos
            results['text_vocab_jaccard']  = vocab_j
            results['text_oov_rate']       = oov_real
            results['len_real']            = len_real
            results['len_syn']             = len_syn
            results['len_diff']            = len_diff
            results['text_topic_cosine']   = topic_cos
            results['text_emb_cosine']     = emb_cos
            results['text_emb_mmd']        = emb_mmd

        else:
            # datetime or other types: only missing‐rate
            logger.info(f"  → No specialized univariate for type '{ctype}', returning only missing‐rate.")

        logger.info(f"  → Completed compute_column for '{col}'.")
        return {'dtype': ctype, **results}


    def plot_column_histogram(self, col: str):
        logger.info(f"Generating histogram plot for '{col}'.")
        ref_vals = self.ref_df[col].dropna().astype(float)
        cmp_vals = self.cmp_df[col].dropna().astype(float)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ref_vals,
            histnorm='probability density',
            name='Reference',
            opacity=0.6,
            nbinsx=30
        ))
        fig.add_trace(go.Histogram(
            x=cmp_vals,
            histnorm='probability density',
            name='Comparison',
            opacity=0.6,
            nbinsx=30
        ))
        fig.update_layout(
            barmode='overlay',
            title=f"Histogram (Density) for '{col}'",
            xaxis_title=col,
            yaxis_title='Density'
        )
        logger.info(f"  → Histogram figure ready for '{col}'.")
        return fig


    def plot_column_kde(self, col: str):
        logger.info(f"Generating KDE plot for '{col}'.")
        ref_vals = self.ref_df[col].dropna().astype(float).values
        cmp_vals = self.cmp_df[col].dropna().astype(float).values

        if len(np.unique(ref_vals)) < 2 or len(np.unique(cmp_vals)) < 2:
            logger.info("  → Not enough distinct values for KDE: returning empty figure.")
            return go.Figure()

        try:
            kde_ref = gaussian_kde(ref_vals)
            kde_cmp = gaussian_kde(cmp_vals)
        except np.linalg.LinAlgError:
            logger.info("  → KDE failed (singular covariance): returning empty figure.")
            return go.Figure()

        pooled = np.concatenate([ref_vals, cmp_vals])
        xmin, xmax = pooled.min(), pooled.max()
        grid = np.linspace(xmin, xmax, 200)

        pdf_ref = kde_ref(grid)
        pdf_cmp = kde_cmp(grid)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=pdf_ref, mode='lines', name='Ref KDE'))
        fig.add_trace(go.Scatter(x=grid, y=pdf_cmp, mode='lines', name='Cmp KDE'))
        fig.update_layout(
            title=f"KDE Plot for '{col}'",
            xaxis_title=col,
            yaxis_title='Density'
        )
        logger.info(f"  → KDE figure ready for '{col}'.")
        return fig


    def plot_category_bar(self, col: str):
        logger.info(f"Generating category bar chart for '{col}'.")
        ref_counts = self.ref_df[col].dropna().value_counts(normalize=True).reset_index()
        ref_counts.columns = [col, 'proportion']
        ref_counts['dataset'] = 'Reference'

        cmp_counts = self.cmp_df[col].dropna().value_counts(normalize=True).reset_index()
        cmp_counts.columns = [col, 'proportion']
        cmp_counts['dataset'] = 'Comparison'

        df_union = pd.concat([ref_counts, cmp_counts], ignore_index=True)

        fig = px.bar(
            df_union,
            x=col,
            y='proportion',
            color='dataset',
            barmode='group',
            title=f"Category Proportions for '{col}'"
        )
        logger.info(f"  → Category bar chart ready for '{col}'.")
        return fig


    def plot_text_top_tokens(self, col: str, top_n: int = 20):
        logger.info(f"Generating top‐token bar chart for '{col}' (top_n={top_n}).")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()

        tokens_ref = [tok for t in texts_ref for tok in t.split()]
        tokens_cmp = [tok for t in texts_cmp for tok in t.split()]

        ctr_ref = Counter(tokens_ref)
        ctr_cmp = Counter(tokens_cmp)
        combined = ctr_ref + ctr_cmp
        most_common = [tok for tok, _ in combined.most_common(top_n)]

        ref_freqs = [ctr_ref.get(tok, 0) for tok in most_common]
        cmp_freqs = [ctr_cmp.get(tok, 0) for tok in most_common]

        n = len(most_common)
        df_plot = pd.DataFrame({
            'token': most_common * 2,
            'count': ref_freqs + cmp_freqs,
            'dataset': ['Reference'] * n + ['Comparison'] * n
        })

        fig = px.bar(
            df_plot,
            x='token',
            y='count',
            color='dataset',
            barmode='group',
            title=f"Top {n} Token Frequencies for '{col}'"
        )
        logger.info(f"  → Top‐token bar chart ready for '{col}'.")
        return fig


    def plot_text_length_hist(self, col: str):
        logger.info(f"Generating document‐length histogram for '{col}'.")
        lens_ref = self.ref_df[col].dropna().astype(str).str.split().apply(len)
        lens_cmp = self.cmp_df[col].dropna().astype(str).str.split().apply(len)

        df_plot = pd.DataFrame({
            'length': pd.concat([lens_ref, lens_cmp], ignore_index=True),
            'dataset': ['Reference'] * len(lens_ref) + ['Comparison'] * len(lens_cmp)
        })

        fig = px.histogram(
            df_plot,
            x='length',
            color='dataset',
            histnorm='probability density',
            barmode='overlay',
            nbins=30,
            title=f"Document Length Distribution for '{col}'"
        )
        logger.info(f"  → Document‐length histogram ready for '{col}'.")
        return fig


    def plot_text_embedding_scatter(self, col: str):
        logger.info(f"Generating (empty) embedding scatter for '{col}'.")
        # Placeholder (no implementation)
        return go.Figure()


    # ───── PCA‐Based Methods ────────────────────────────────────────────────────

    def _get_numeric_matrix(self) -> (np.ndarray, np.ndarray, list):
        logger.info("Building numeric matrices for PCA.")
        numeric_cols = [col for col in self.common_columns if self.column_types[col] == 'numeric']
        logger.info(f"  → Numeric columns: {numeric_cols}")
        if not numeric_cols:
            return np.array([]), np.array([]), []

        ref_num = self.ref_df[numeric_cols].dropna(axis=0, how='any').astype(float)
        cmp_num = self.cmp_df[numeric_cols].dropna(axis=0, how='any').astype(float)
        logger.info(f"  → After dropping NaNs: ref_num shape={ref_num.shape}, cmp_num shape={cmp_num.shape}")

        scaler_ref = StandardScaler().fit(ref_num.values)
        X_ref = scaler_ref.transform(ref_num.values)

        scaler_cmp = StandardScaler().fit(cmp_num.values)
        X_cmp = scaler_cmp.transform(cmp_num.values)

        return X_ref, X_cmp, numeric_cols


    def compute_pca_eigen_cosines(self, n_components: int = 5) -> dict:
        logger.info(f"Computing PCA eigenvector cosines (n_components={n_components}).")
        X_ref, X_cmp, numeric_cols = self._get_numeric_matrix()
        if X_ref.size == 0 or X_cmp.size == 0:
            logger.info("  → Not enough numeric data: returning empty PCA result.")
            return {
                'cosine_similarities': [],
                'explained_variance_ref': [],
                'explained_variance_cmp': [],
                'weighted_cosine': np.nan
            }

        pca_ref = PCA(n_components=n_components)
        pca_cmp = PCA(n_components=n_components)
        pca_ref.fit(X_ref)
        pca_cmp.fit(X_cmp)

        eigvecs_ref = pca_ref.components_
        eigvecs_cmp = pca_cmp.components_

        cosines = []
        for i in range(len(eigvecs_ref)):
            v1 = eigvecs_ref[i]
            v2 = eigvecs_cmp[i]
            dot = np.dot(v1, v2)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            cos = dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0
            cosines.append(cos)
            logger.info(f"  → PC{i+1} cosine = {cos:.6f}")

        weights = pca_ref.explained_variance_ratio_
        weighted_cos = float(np.dot(cosines, weights[:len(cosines)]))
        logger.info(f"  → Weighted cosine = {weighted_cos:.6f}")

        return {
            'cosine_similarities': cosines,
            'explained_variance_ref': pca_ref.explained_variance_ratio_.tolist(),
            'explained_variance_cmp': pca_cmp.explained_variance_ratio_.tolist(),
            'weighted_cosine': weighted_cos
        }


    def get_pca_projection_dataframe(self, n_components: int = 2) -> pd.DataFrame:
        logger.info(f"Building PCA projection dataframe (n_components={n_components}).")
        X_ref, X_cmp, numeric_cols = self._get_numeric_matrix()
        if X_ref.size == 0 or X_cmp.size == 0:
            logger.info("  → Not enough numeric data: returning empty DataFrame.")
            return pd.DataFrame()

        X_all = np.vstack([X_ref, X_cmp])
        pca = PCA(n_components=n_components)
        coords_all = pca.fit_transform(X_all)

        n_ref = X_ref.shape[0]
        df_coords = pd.DataFrame(coords_all, columns=[f"PC{i+1}" for i in range(n_components)])
        df_coords['dataset'] = ['Reference'] * n_ref + ['Comparison'] * (coords_all.shape[0] - n_ref)
        logger.info("  → PCA projection dataframe ready.")
        return df_coords


    def plot_pca_scatter(self, n_components: int = 2):
        logger.info(f"Generating PCA scatter plot (components={n_components}).")
        df_coords = self.get_pca_projection_dataframe(n_components=n_components)
        if df_coords.empty or 'PC1' not in df_coords or 'PC2' not in df_coords:
            logger.info("  → Not enough PCA components: returning empty figure.")
            return go.Figure()

        fig = px.scatter(
            df_coords,
            x="PC1",
            y="PC2",
            color="dataset",
            symbol="dataset",
            title=f"PCA Projection (PC1 vs PC2)"
        )
        fig.update_traces(marker=dict(size=5, opacity=0.75))
        logger.info("  → PCA scatter plot ready.")
        return fig


    def compute_correlation_similarity(self) -> float:
        """
        Computes the average pairwise Pearson‐correlation similarity between
        numeric columns in the reference vs. comparison data. If there are fewer
        than 2 numeric columns, returns np.nan.
        """
        logger.info("Computing global correlation similarity (numeric × numeric).")
        numeric_cols = [c for c in self.common_columns if self.column_types.get(c) == "numeric"]
        logger.info(f"  → Numeric columns: {numeric_cols}")
        if len(numeric_cols) < 2:
            logger.info("  → Fewer than 2 numeric columns: returning NaN.")
            return np.nan

        df_real = self.ref_df[numeric_cols].dropna()
        df_syn  = self.cmp_df[numeric_cols].dropna()
        if df_real.shape[0] < 2 or df_syn.shape[0] < 2:
            logger.info("  → Not enough rows after dropping NaNs: returning NaN.")
            return np.nan

        corr_real = df_real.corr(method="pearson")
        corr_syn  = df_syn.corr(method="pearson")
        sims = []
        for (i, j) in combinations(numeric_cols, 2):
            r_r = corr_real.at[i, j]
            r_s = corr_syn.at[i, j]
            if pd.isna(r_r) or pd.isna(r_s):
                continue
            sim_ij = 1.0 - abs(r_r - r_s)
            sims.append(sim_ij)
            logger.info(f"  → Pair ({i},{j}): corr_real={r_r:.4f}, corr_syn={r_s:.4f}, sim={sim_ij:.6f}")

        if not sims:
            logger.info("  → No valid correlation pairs: returning NaN.")
            return np.nan
        mean_sim = float(np.mean(sims))
        logger.info(f"  → Mean correlation similarity = {mean_sim:.6f}")
        return mean_sim


    def compute_contingency_similarity(self) -> float:
        """
        Computes the average Total Variation Distance (TVD) similarity over all
        pairs of categorical columns. If fewer than 2 categorical columns, returns np.nan.

        For each pair of categorical columns (A,B):
          • Build the joint frequency table in real and synthetic (normalized).
          • TVD(A,B) = 0.5 * sum |p_real( a,b ) - p_syn( a,b )| over all (a,b).
          • sim(A,B) = 1 - TVD(A,B).
        Finally, return the mean(sim(A,B)) for all A < B.
        """
        logger.info("Computing global contingency similarity (categorical × categorical).")
        categorical_cols = [c for c in self.common_columns if self.column_types.get(c) == "categorical"]
        logger.info(f"  → Categorical columns: {categorical_cols}")
        if len(categorical_cols) < 2:
            logger.info("  → Fewer than 2 categorical columns: returning NaN.")
            return np.nan

        sims = []
        for (c1, c2) in combinations(categorical_cols, 2):
            logger.info(f"  → Processing pair ({c1},{c2}).")
            real_pair = self.ref_df[[c1, c2]].dropna().astype(str)
            syn_pair  = self.cmp_df[[c1, c2]].dropna().astype(str)

            if real_pair.shape[0] < 1 or syn_pair.shape[0] < 1:
                logger.info(f"    → One side empty after dropna: skipping pair.")
                continue

            real_counts = real_pair.groupby([c1, c2]).size().rename("count").reset_index()
            syn_counts  = syn_pair.groupby([c1, c2]).size().rename("count").reset_index()

            real_pivot = real_counts.pivot(index=c1, columns=c2, values="count").fillna(0)
            syn_pivot  = syn_counts.pivot(index=c1, columns=c2, values="count").fillna(0)

            all_index = real_pivot.index.union(syn_pivot.index)
            all_columns = real_pivot.columns.union(syn_pivot.columns)
            real_norm = real_pivot.reindex(index=all_index, columns=all_columns, fill_value=0)
            syn_norm  = syn_pivot.reindex(index=all_index, columns=all_columns, fill_value=0)

            real_prob = real_norm.values / real_norm.values.sum()
            syn_prob  = syn_norm.values / syn_norm.values.sum()

            tvd = 0.5 * np.abs(real_prob - syn_prob).sum()
            sim = 1.0 - tvd
            sims.append(sim)
            logger.info(f"    → TVD for pair = {tvd:.6f}, similarity = {sim:.6f}")

        if not sims:
            logger.info("  → No valid categorical pairs: returning NaN.")
            return np.nan
        mean_sim = float(np.mean(sims))
        logger.info(f"  → Mean contingency similarity = {mean_sim:.6f}")
        return mean_sim


    def compute_distinguishability_auc(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        max_rows_per_side: int = 1000
    ) -> float:
        """
        Trains a LogisticRegression on a balanced subsample
        of up to max_rows_per_side real vs. synthetic rows.
        Returns ROC AUC. If not enough variance, returns NaN.
        """
        logger.info(f"Computing distinguishability AUC (max_rows_per_side={max_rows_per_side}).")
        df_real = self.ref_df.copy()
        df_real["_is_synthetic"] = 0
        df_syn  = self.cmp_df.copy()
        df_syn["_is_synthetic"] = 1

        if len(df_real) > max_rows_per_side:
            df_real = df_real.sample(n=max_rows_per_side, random_state=random_state)
            logger.info(f"  → Subsampled real to {max_rows_per_side} rows.")
        if len(df_syn) > max_rows_per_side:
            df_syn = df_syn.sample(n=max_rows_per_side, random_state=random_state)
            logger.info(f"  → Subsampled synthetic to {max_rows_per_side} rows.")

        df_all = pd.concat([df_real, df_syn], ignore_index=True)
        logger.info(f"  → Concatenated dataset shape = {df_all.shape}")

        numeric_cols = [c for c in self.common_columns if self.column_types.get(c) == "numeric"]
        categorical_cols = [c for c in self.common_columns if self.column_types.get(c) == "categorical"]
        logger.info(f"  → Numeric columns: {numeric_cols}")
        logger.info(f"  → Categorical columns: {categorical_cols}")

        if numeric_cols:
            num_mat = df_all[numeric_cols].astype(float).fillna(0.0)
            scaler = StandardScaler()
            X_num = scaler.fit_transform(num_mat.values)
            logger.info(f"  → Numeric matrix shape = {X_num.shape}")
        else:
            X_num = np.empty((len(df_all), 0))
            logger.info("  → No numeric columns: X_num is empty array.")

        if categorical_cols:
            df_cat = df_all[categorical_cols].astype(str).fillna("NA").copy()
            for col in categorical_cols:
                top_cats = df_cat[col].value_counts().nlargest(50).index
                df_cat[col] = df_cat[col].where(df_cat[col].isin(top_cats), other="OTHER")
                logger.info(f"  → For '{col}', collapsed rare categories to 'OTHER'.")
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat = encoder.fit_transform(df_cat.values)
            logger.info(f"  → One-hot encoded shape = {X_cat.shape}")
        else:
            X_cat = np.empty((len(df_all), 0))
            logger.info("  → No categorical columns: X_cat is empty array.")

        X = np.hstack([X_num, X_cat])
        y = df_all["_is_synthetic"].values
        logger.info(f"  → Combined feature matrix shape = {X.shape}, labels length = {len(y)}")

        if len(np.unique(y)) < 2:
            logger.info("  → Only one class present: returning NaN.")
            return np.nan

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            logger.info(f"  → Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")
        except ValueError as e:
            logger.info(f"  → Train/test split error: {e}. Returning NaN.")
            return np.nan

        clf = LogisticRegression(
            solver="liblinear", max_iter=100, random_state=random_state
        )
        clf.fit(X_train, y_train)
        logger.info("  → LogisticRegression fitted on training data.")

        y_prob = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
            logger.info(f"  → Computed ROC AUC = {auc:.6f}")
        except ValueError as e:
            logger.info(f"  → ROC AUC computation error: {e}. Returning NaN.")
            return np.nan

        return float(auc)


    # ─── Text‐Bigram & OOV Helpers ────────────────────────────────────────────

    def _compute_text_bigram_jaccard(self, col: str) -> float:
        logger.info(f"Computing bigram‐level Jaccard for '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side empty: returning NaN.")
            return np.nan

        def all_bigrams(texts):
            bigrams = set()
            for t in texts:
                tokens = t.split()
                for i in range(len(tokens) - 1):
                    bigrams.add((tokens[i], tokens[i + 1]))
            return bigrams

        big_ref = all_bigrams(texts_ref)
        big_cmp = all_bigrams(texts_cmp)
        if not big_ref and not big_cmp:
            logger.info("  → Both bigram sets empty: returning NaN.")
            return np.nan
        if not big_ref or not big_cmp:
            logger.info("  → One bigram set empty: returning 0.0.")
            return 0.0
        intersect = big_ref.intersection(big_cmp)
        union = big_ref.union(big_cmp)
        jaccard = len(intersect) / len(union)
        logger.info(f"  → Bigram Jaccard = {jaccard:.6f}")
        return jaccard


    def _compute_text_bigram_js(self, col: str, top_k: int = 5000) -> float:
        logger.info(f"Computing bigram‐level JS divergence for '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side empty: returning NaN.")
            return np.nan

        def bigram_counts(texts):
            ctr = Counter()
            for t in texts:
                tokens = t.split()
                for i in range(len(tokens) - 1):
                    ctr[(tokens[i], tokens[i + 1])] += 1
            return ctr

        ctr_ref = bigram_counts(texts_ref)
        ctr_cmp = bigram_counts(texts_cmp)
        combined = ctr_ref + ctr_cmp
        most_common = [bg for bg, _ in combined.most_common(top_k)]
        logger.info(f"  → Top {len(most_common)} bigrams selected.")

        p = np.array([ctr_ref.get(bg, 0) for bg in most_common], dtype=float)
        q = np.array([ctr_cmp.get(bg, 0) for bg in most_common], dtype=float)
        if p.sum() == 0 or q.sum() == 0:
            logger.info("  → Zero counts on one side: returning NaN.")
            return np.nan
        p = p / p.sum()
        q = q / q.sum()
        js = Report._js_divergence(p, q)
        return js


    def _compute_text_oov_rate(self, col: str) -> float:
        logger.info(f"Computing OOV rate for '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side empty: returning NaN.")
            return np.nan

        tokens_ref = set(tok for t in texts_ref for tok in t.split())
        tokens_cmp = set(tok for t in texts_cmp for tok in t.split())
        if not tokens_ref:
            logger.info("  → No tokens in real: returning NaN.")
            return np.nan
        oov_real = len(tokens_ref - tokens_cmp) / len(tokens_ref)
        logger.info(f"  → OOV rate (real→syn) = {oov_real:.6f}")
        return oov_real


    def _compute_text_length_stats(self, col: str) -> dict:
        logger.info(f"Computing text length stats for '{col}'.")
        lens_ref = self.ref_df[col].dropna().astype(str).str.split().apply(len)
        lens_cmp = self.cmp_df[col].dropna().astype(str).str.split().apply(len)
        if lens_ref.empty or lens_cmp.empty:
            logger.info("  → One side empty: returning NaN stats.")
            return {'len_real': np.nan, 'len_syn': np.nan, 'len_diff': np.nan}
        avg_r = lens_ref.mean()
        avg_s = lens_cmp.mean()
        diff = avg_s - avg_r
        logger.info(f"  → Avg length real={avg_r:.2f}, syn={avg_s:.2f}, diff={diff:.2f}")
        return {'len_real': avg_r, 'len_syn': avg_s, 'len_diff': diff}


    def _compute_text_topic_cosine(
        self,
        col: str,
        n_topics: int = 5,
        max_features: int = 1000,
        max_docs_per_side: int = 200
    ) -> float:
        logger.info(f"Computing topic cosine for '{col}'.")
        texts_ref = self.ref_df[col].dropna().astype(str).tolist()
        texts_cmp = self.cmp_df[col].dropna().astype(str).tolist()
        if not texts_ref or not texts_cmp:
            logger.info("  → One side empty: returning NaN.")
            return np.nan

        if len(texts_ref) > max_docs_per_side:
            rng = np.random.default_rng(seed=0)
            texts_ref = list(rng.choice(texts_ref, max_docs_per_side, replace=False))
            logger.info(f"  → Subsampled {len(texts_ref)} real docs.")
        if len(texts_cmp) > max_docs_per_side:
            rng = np.random.default_rng(seed=1)
            texts_cmp = list(rng.choice(texts_cmp, max_docs_per_side, replace=False))
            logger.info(f"  → Subsampled {len(texts_cmp)} synthetic docs.")

        corpus = texts_ref + texts_cmp
        logger.info(f"  → TF–IDF on {len(corpus)} docs (max_features={max_features}).")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        try:
            tfidf_all = vectorizer.fit_transform(corpus)
        except Exception as e:
            logger.info(f"  → TF–IDF fit error: {e}. Returning NaN.")
            return np.nan

        nmf = NMF(
            n_components=n_topics,
            random_state=0,
            init='nndsvda',
            max_iter=100
        )
        logger.info(f"  → Fitting NMF (n_topics={n_topics}) on TF–IDF matrix.")
        try:
            W_all = nmf.fit_transform(tfidf_all)
        except Exception as e:
            logger.info(f"  → NMF fit error: {e}. Returning NaN.")
            return np.nan

        n_ref = len(texts_ref)
        W_ref = W_all[:n_ref, :]
        W_cmp = W_all[n_ref:, :]

        mean_ref = W_ref.mean(axis=0)
        mean_cmp = W_cmp.mean(axis=0)
        denom = np.linalg.norm(mean_ref) * np.linalg.norm(mean_cmp)
        cosine = float(np.dot(mean_ref, mean_cmp) / denom) if denom > 0 else np.nan
        logger.info(f"  → Topic cosine = {cosine:.6f}")
        return cosine
