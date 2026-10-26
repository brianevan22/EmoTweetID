"""Tahap 3: latih ulang model terbaik pada seluruh data Human, label seluruh Corpus.
Dapat dilanjutkan jika sesi terputus (hasil per potongan disimpan di results/chunks).
Keluaran: results/corpus_labeled.parquet, corpus_sample.csv, corpus_summary.json, models/final*"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from config import E2I, EMOTIONS, K, MODELS, RES, SEED
from common import STOP, clean_full, clean_min, key_hashes, load_corpus, load_human, scores, word_counts

CHUNK = 250_000


def distinctive_words(texts, labels, topn=15, alpha=500.0):
    """Log-odds berbobot dengan prior informatif Dirichlet (Monroe et al., 2008), emosi vs lainnya."""
    cv = CountVectorizer(min_df=20, max_features=60000, stop_words=list(STOP))
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    tot = np.asarray(X.sum(0)).ravel().astype(float)
    prior = alpha * tot / tot.sum()
    a0 = prior.sum()
    out = {}
    for e in EMOTIONS:
        m = (labels == e)
        if m.sum() == 0:
            out[e] = []; continue
        y1 = np.asarray(X[m].sum(0)).ravel().astype(float)
        y2 = tot - y1
        n1, n2 = y1.sum(), y2.sum()
        d = np.log((y1 + prior) / (n1 + a0 - y1 - prior)) - np.log((y2 + prior) / (n2 + a0 - y2 - prior))
        z = d / np.sqrt(1 / (y1 + prior) + 1 / (y2 + prior))
        top = np.argsort(-z)[:topn]
        out[e] = [{"word": vocab[i], "z": float(z[i]), "count": int(y1[i])} for i in top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = seluruh korpus")
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()

    m = json.loads((RES / "metrics.json").read_text())
    best = m["best_model"]
    print("Model terbaik dari tahap 2:", best)
    h = load_human()
    yh = h.label.map(E2I).to_numpy()

    corpus = load_corpus()
    n0 = len(corpus)
    hs = key_hashes(corpus.tweet)
    first = ~pd.Series(hs).duplicated().to_numpy()
    n1 = int(first.sum())
    keep = first & ~np.isin(hs, key_hashes(h.tweet))
    corpus = corpus[keep].reset_index(drop=True)
    del hs, first, keep
    print(f"Korpus: {n0:,} -> {n1:,} setelah deduplikasi -> {len(corpus):,} setelah membuang tweet berlabel (anti kebocoran)")
    n_after = len(corpus)
    if a.sample:
        corpus = corpus.sample(min(a.sample, len(corpus)), random_state=SEED).reset_index(drop=True)
    texts = corpus.tweet.tolist()

    # ---- latih ulang pada seluruh data berlabel
    if best == "IndoBERTweet":
        from bert_utils import fine_tune, predict_proba
        Xm = [clean_min(t) for t in h.tweet]
        tr, va = train_test_split(np.arange(len(h)), test_size=0.1, stratify=yh, random_state=SEED)
        model, tok, _ = fine_tune([Xm[i] for i in tr], yh[tr], [Xm[i] for i in va], yh[va],
                                  lr=a.lr, epochs=a.epochs, max_len=a.max_len)
        model.save_pretrained(MODELS / "final"); tok.save_pretrained(MODELS / "final")
        (MODELS / "final" / "meta.json").write_text(json.dumps({"type": "bert", "max_len": a.max_len}))
        predict = lambda ts: predict_proba(model, tok, [clean_min(t) for t in ts], bs=a.bs, max_len=a.max_len)
    else:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        import importlib
        tr_mod = importlib.import_module("02_train_eval")
        pipe = tr_mod.make_pipe(tr_mod.CLASSIC[best]())
        pipe.fit([clean_full(t) for t in h.tweet], yh)
        joblib.dump(pipe, MODELS / "final.joblib")
        predict = lambda ts: scores(pipe, [clean_full(t) for t in ts])

    # ---- prediksi per potongan (dapat dilanjutkan)
    cdir = RES / "chunks"; cdir.mkdir(exist_ok=True)
    lab, conf = [], []
    for i, s in enumerate(range(0, len(texts), CHUNK)):
        f = cdir / f"pred_{a.sample}_{i}.npz"
        if f.exists():
            z = np.load(f)
        else:
            P = predict(texts[s:s + CHUNK])
            z = {"label": P.argmax(1).astype(np.uint8), "conf": P.max(1).astype(np.float32)}
            np.savez(f, **z)
        lab.append(z["label"]); conf.append(z["conf"])
        print(f"  potongan {i + 1}/{int(np.ceil(len(texts) / CHUNK))} selesai")
    corpus["label"] = np.array(EMOTIONS)[np.concatenate(lab)]
    corpus["conf"] = np.concatenate(conf)

    # ---- ambang keyakinan dari kurva presisi-cakupan (data OOF)
    ok = [p for p in m["pr_curve"] if p["precision"] >= a.target_precision]
    tau = min(p["t"] for p in ok) if ok else None
    print("Ambang keyakinan (presisi >= %.2f): %s" % (a.target_precision, tau))
    hi = corpus.conf >= tau if tau is not None else pd.Series(False, index=corpus.index)

    corpus.to_parquet(RES / "corpus_labeled.parquet", index=False)
    corpus.sample(min(50_000, len(corpus)), random_state=SEED).to_csv(RES / "corpus_sample.csv", index=False)

    wc = pd.Series(word_counts(corpus.tweet), index=corpus.index)
    ds = min(400_000, len(corpus))
    s = corpus.sample(ds, random_state=SEED)
    summ = {"model": best, "n_corpus_raw": n0, "n_after_dedup": n1, "n_after_remove_human": n_after,
            "n_labeled": len(corpus),
            "coverage_by_threshold": {f"{x:.1f}": float((corpus.conf >= x).mean()) for x in (0.3, 0.5, 0.7, 0.9)},
            "mean_conf_overall": float(corpus.conf.mean()), "is_sample": bool(a.sample), "threshold": tau,
            "target_precision": a.target_precision, "n_high_conf": int(hi.sum()),
            "counts_all": corpus.label.value_counts().reindex(EMOTIONS, fill_value=0).to_dict(),
            "counts_high_conf": corpus[hi].label.value_counts().reindex(EMOTIONS, fill_value=0).to_dict(),
            "mean_conf": corpus.groupby("label").conf.mean().reindex(EMOTIONS).round(4).to_dict(),
            "mean_words": wc.groupby(corpus.label).mean().reindex(EMOTIONS).round(3).to_dict(),
            "distinctive_words": distinctive_words([clean_full(t) for t in s.tweet], s.label.to_numpy())}
    (RES / "corpus_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps({k: summ[k] for k in ("n_labeled", "counts_all", "counts_high_conf", "mean_conf")}, indent=2))


if __name__ == "__main__":
    main()
