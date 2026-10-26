import re
import numpy as np
import pandas as pd
from config import DATA_DIR, EMOTIONS, K

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")
# Stopword ringkas. Kata negasi (tidak, gak, nggak, bukan, jangan) SENGAJA dipertahankan
# karena membawa makna emosi.
STOP = set("""yang dan di ke dari ini itu untuk dengan pada adalah atau juga saya aku kamu dia kita
mereka ada akan sudah udah telah oleh dalam sebagai agar bahwa para si sang nya lah kah pun yg dgn
dg utk krn karena sih deh dong kok nih tuh lho loh kan""".split())
ID_HINT = set("""yang dan di ke dari ini itu untuk dengan tidak gak ga nggak aku kamu saya nya sama udah aja
banget lagi kalau kalo karena bisa ada jadi mau apa kok sih deh dong yg dgn tapi""".split())


def clean_full(t):
    """Alur penuh untuk model TF-IDF."""
    t = str(t).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    t = re.sub(r"[^\w\s]", " ", t)
    t = REPEAT_RE.sub(r"\1\1", t)
    return " ".join(w for w in t.split() if w not in STOP)


def clean_min(t):
    """Alur minimal untuk IndoBERTweet (konvensi @USER / HTTPURL)."""
    t = URL_RE.sub("HTTPURL", str(t))
    t = MENTION_RE.sub("@USER", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_key(t):
    t = MENTION_RE.sub("", URL_RE.sub("", str(t).lower()))
    return re.sub(r"\s+", " ", t).strip()


def find_file(keyword):
    cands = [p for p in DATA_DIR.rglob("*.csv")
             if keyword in p.name.lower()
             and not any(x in p.name.lower() for x in ("english", "translat", "_en."))]
    if not cands:
        raise FileNotFoundError(f"File CSV berisi '{keyword}' tidak ditemukan di {DATA_DIR}")
    return sorted(cands)[0]


def _is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _text_col(df):
    for c in df.columns:
        if str(c).strip().lower() == "tweet":
            return c
    obj = [c for c in df.columns if _is_text(df[c])]
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def _label_col(df, exclude, thr):
    best, frac_best = None, 0
    for c in df.columns:
        if c == exclude or not _is_text(df[c]):
            continue
        frac = df[c].astype(str).str.lower().str.strip().isin(EMOTIONS).mean()
        if frac > frac_best:
            best, frac_best = c, frac
    if best is None or frac_best < thr:
        raise ValueError(f"Kolom label emosi tidak terdeteksi. Kolom tersedia: {list(df.columns)}")
    return best


def _wide_to_label(df):
    cols = [c for c in df.columns if str(c).strip().lower() in EMOTIONS]
    if len(cols) < 4:
        return None
    sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    lab = sub.idxmax(axis=1).astype(str).str.lower().str.strip()
    lab[sub.sum(axis=1) == 0] = "none"
    return lab


def load_human():
    df = pd.read_csv(find_file("human"))
    tc = _text_col(df)
    lc = _label_col(df, tc, 0.8)
    out = pd.DataFrame({"tweet": df[tc].astype(str),
                        "label": df[lc].astype(str).str.lower().str.strip()})
    bad = ~out.label.isin(EMOTIONS)
    if bad.any():
        print(f"[peringatan] {bad.sum()} baris label di luar enam emosi dibuang")
        out = out[~bad]
    out["key"] = out.tweet.map(norm_key)
    g = out.groupby("key").label.agg(["size", "nunique"])
    conflict = set(g.index[g["nunique"] > 1])
    load_human.report = {
        "rows_raw": int(len(out)),
        "duplicate_keys": int((g["size"] > 1).sum()),
        "rows_in_duplicate_keys": int(g.loc[g["size"] > 1, "size"].sum()),
        "keys_with_conflicting_labels": len(conflict),
        "rows_in_conflicting_keys": int(g.loc[list(conflict), "size"].sum()) if conflict else 0,
    }
    # Teks kembar berlabel sama dipertahankan satu kali; yang berlabel bertentangan dibuang (ambigu).
    out = out[~out.key.isin(conflict)].drop_duplicates("key").reset_index(drop=True)
    load_human.report["n_final"] = int(len(out))
    print("[audit Human]", load_human.report)
    return out


def load_lexicon():
    df = pd.read_csv(find_file("lexicon"))
    tc = _text_col(df)
    lab = _wide_to_label(df)
    if lab is None:
        lab = df[_label_col(df, tc, 0.5)].astype(str).str.lower().str.strip()
    out = pd.DataFrame({"tweet": df[tc].astype(str), "label": lab.values})
    out["key"] = out.tweet.map(norm_key)
    return out.drop_duplicates("key").reset_index(drop=True)


def load_corpus():
    df = pd.read_csv(find_file("corpus"))
    return pd.DataFrame({"tweet": df[_text_col(df)].astype(str)})


def iter_corpus(chunksize=300_000):
    """Baca korpus per potongan (hemat memori)."""
    p = find_file("corpus")
    tc = _text_col(pd.read_csv(p, nrows=200))
    for ch in pd.read_csv(p, usecols=[tc], chunksize=chunksize):
        yield ch[tc].astype(str).reset_index(drop=True)


def key_hashes(series):
    """Hash 64-bit dari teks ternormalisasi; menggantikan penyimpanan string kunci."""
    return pd.util.hash_pandas_object(series.map(norm_key), index=False).to_numpy()


def word_counts(series):
    return series.str.count(r"\S+").to_numpy()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def scores(pipe, X):
    """Probabilitas (atau softmax dari margin untuk SVM) -> array (n, K)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    return softmax(pipe.decision_function(X))


def association_stats(ct):
    """ct: DataFrame tabel kontingensi (emosi x topik). Chi-square, Cramer's V, residual terstandar."""
    from scipy.stats import chi2_contingency
    obs = ct.values.astype(float)
    chi2, p, dof, exp = chi2_contingency(obs, correction=False)
    n = obs.sum()
    r, c = obs.shape
    v = float(np.sqrt(chi2 / (n * (min(r, c) - 1))))
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rc, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    v_corr = float(np.sqrt(phi2 / (min(rc, cc) - 1))) if min(rc, cc) > 1 else float("nan")
    rp = obs.sum(1, keepdims=True) / n
    cp = obs.sum(0, keepdims=True) / n
    resid = (obs - exp) / np.sqrt(exp * (1 - rp) * (1 - cp))
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "n": int(n),
            "cramers_v": v, "cramers_v_bias_corrected": v_corr,
            "min_expected": float(exp.min())}, pd.DataFrame(resid, index=ct.index, columns=ct.columns)
