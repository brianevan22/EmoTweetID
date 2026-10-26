"""Tahap 2: model klasik vs IndoBERTweet, 5-fold CV, uji H1 dan H2.
Keluaran: results/metrics.json, results/oof_predictions.csv, results/oof_probs.npz"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from statsmodels.stats.contingency_tables import mcnemar
from config import E2I, EMOTIONS, K, RES, SEED
from common import clean_full, clean_min, load_human, load_lexicon, scores

CLASSIC = {"SVM": lambda: LinearSVC(class_weight="balanced"),
           "NaiveBayes": lambda: MultinomialNB(),
           "LogReg": lambda: LogisticRegression(class_weight="balanced", max_iter=3000)}


def make_pipe(clf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)), ("clf", clf)])


def metrics(y, p):
    rep = classification_report(y, p, labels=range(K), target_names=EMOTIONS, output_dict=True, zero_division=0)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "macro_precision": pr, "macro_recall": rc, "macro_f1": f1,
            "per_class": {e: {k: rep[e][k] for k in ("precision", "recall", "f1-score", "support")} for e in EMOTIONS},
            "confusion": confusion_matrix(y, p, labels=range(K)).tolist()}


def boot_f1_ci(y, p, B=1000):
    rng = np.random.default_rng(SEED)
    n = len(y)
    v = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(v, [2.5, 97.5]).tolist()


def boot_kappa(yh, pm, pl, B=1000):
    rng = np.random.default_rng(SEED)
    n = len(yh)
    km, kl, d = [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        a, b = cohen_kappa_score(yh[i], pm[i]), cohen_kappa_score(yh[i], pl[i])
        km.append(a); kl.append(b); d.append(a - b)
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    return ci(km), ci(kl), ci(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    h = load_human()
    y = h.label.map(E2I).to_numpy()
    n = len(h)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(h.tweet, y))
    Xf = [clean_full(t) for t in h.tweet]
    Xm = [clean_min(t) for t in h.tweet]
    probs, fold_f1 = {}, {}

    for name, mk in CLASSIC.items():
        print(f"== {name}")
        P = np.zeros((n, K)); fl = []
        for tr, te in folds:
            pipe = make_pipe(mk())
            pipe.fit([Xf[i] for i in tr], y[tr])
            P[te] = scores(pipe, [Xf[i] for i in te])
            fl.append(f1_score(y[te], P[te].argmax(1), average="macro"))
        probs[name], fold_f1[name] = P, fl
        print(f"   macro-F1 per lipatan: {np.round(fl, 4)}")

    if not a.skip_bert:
        import torch
        from bert_utils import fine_tune, predict_proba
        print("== IndoBERTweet")
        P = np.zeros((n, K)); fl = []
        for f, (tr, te) in enumerate(folds, 1):
            tr2, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
            print(f" lipatan {f}/5")
            model, tok, _ = fine_tune([Xm[i] for i in tr2], y[tr2], [Xm[i] for i in va], y[va],
                                      lr=a.lr, bs=a.bs, epochs=a.epochs, max_len=a.max_len)
            P[te] = predict_proba(model, tok, [Xm[i] for i in te], max_len=a.max_len)
            fl.append(f1_score(y[te], P[te].argmax(1), average="macro"))
            del model; torch.cuda.empty_cache()
        probs["IndoBERTweet"], fold_f1["IndoBERTweet"] = P, fl

    res = {"n": n, "classes": EMOTIONS, "models": {}}
    for name, P in probs.items():
        p = P.argmax(1)
        m = metrics(y, p)
        m["fold_macro_f1"] = fold_f1[name]
        m["fold_macro_f1_mean"] = float(np.mean(fold_f1[name]))
        m["fold_macro_f1_sd"] = float(np.std(fold_f1[name], ddof=1))
        m["macro_f1_ci95"] = boot_f1_ci(y, p, a.boot)
        res["models"][name] = m
    best = max(res["models"], key=lambda k: res["models"][k]["macro_f1"])
    best_cl = max(CLASSIC, key=lambda k: res["models"][k]["macro_f1"])
    res["best_model"], res["best_classic"] = best, best_cl
    print(f"Model terbaik: {best} | klasik terbaik: {best_cl}")

    # --- H1: McNemar
    if "IndoBERTweet" in probs:
        pb, pc = probs["IndoBERTweet"].argmax(1), probs[best_cl].argmax(1)
        cb, cc = pb == y, pc == y
        b, c = int((cb & ~cc).sum()), int((~cb & cc).sum())
        tab = [[int((cb & cc).sum()), b], [c, int((~cb & ~cc).sum())]]
        r = mcnemar(tab, exact=(b + c) < 25, correction=True)
        res["h1"] = {"bert_f1": res["models"]["IndoBERTweet"]["macro_f1"],
                     "classic_f1": res["models"][best_cl]["macro_f1"], "classic_name": best_cl,
                     "bert_right_classic_wrong": b, "classic_right_bert_wrong": c,
                     "mcnemar_stat": float(r.statistic), "mcnemar_p": float(r.pvalue), "alpha": 0.05}

    # --- H2: kappa model vs manusia, leksikon vs manusia (tweet yang sama)
    pred_best = np.array(EMOTIONS)[probs[best].argmax(1)]
    d = h.assign(pred=pred_best)
    try:
        lex = load_lexicon()
        m = d.merge(lex[["key", "label"]].rename(columns={"label": "lex"}), on="key")
        yh, pm, pl = m.label.to_numpy(), m.pred.to_numpy(), m.lex.to_numpy()
        ci_m, ci_l, ci_d = boot_kappa(yh, pm, pl, a.boot)
        km, kl = cohen_kappa_score(yh, pm), cohen_kappa_score(yh, pl)
        res["h2"] = {"n_aligned": len(m), "model_name": best, "kappa_model_human": km, "kappa_model_ci95": ci_m,
                     "kappa_lexicon_human": kl, "kappa_lexicon_ci95": ci_l,
                     "diff": km - kl, "diff_ci95": ci_d}
        d = d.merge(lex[["key", "label"]].rename(columns={"label": "lexicon"}), on="key", how="left")
    except Exception as e:  # noqa
        res["h2"] = {"error": str(e)}

    # --- kurva presisi-cakupan (data OOF) untuk ambang keyakinan
    P = probs[best]; conf = P.max(1); ok = P.argmax(1) == y
    res["pr_curve"] = [{"t": float(t), "precision": float(ok[conf >= t].mean()), "coverage": float((conf >= t).mean())}
                       for t in np.linspace(0.2, 0.99, 60) if (conf >= t).sum() >= 20]
    (RES / "metrics.json").write_text(json.dumps(res, indent=2))
    out = d[["tweet", "label"] + (["lexicon"] if "lexicon" in d else []) + ["pred"]].copy()
    out["conf"] = conf
    out.to_csv(RES / "oof_predictions.csv", index=False)
    np.savez_compressed(RES / "oof_probs.npz", y=y, **probs)
    for k, v in res["models"].items():
        print(f"{k:13s} macro-F1={v['macro_f1']:.4f} CI95={np.round(v['macro_f1_ci95'], 4)} acc={v['accuracy']:.4f}")


if __name__ == "__main__":
    main()
