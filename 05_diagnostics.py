"""Tahap 5 (diagnostik validitas): seberapa bergantung model pada kata emosi eksplisit, dan seberapa
berbeda korpus dari data berlabel. Keluaran: results/diagnostics.json"""
import importlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from config import E2I, EMOTIONS, K, RES, SEED
from common import clean_full, iter_corpus, load_human, scores

# Kata emosi eksplisit (daftar sengaja ditulis tangan agar dapat diperiksa dan dikritik)
SEEDS = set("""marah kesal kesel benci murka tersinggung jijik muak risih ngeri takut cemas gugup
senang bahagia bangga puas syukur sedih kecewa duka nangis sengsara heran terkejut terpesona dengki penat""".split())
tr = importlib.import_module("02_train_eval")


def has_seed(t):
    return any(w in SEEDS for w in clean_full(t).split())


def cv_f1(X, y):
    p = cross_val_predict(tr.make_pipe(tr.CLASSIC["LogReg"]()), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return float(f1_score(y, p, average="macro"))


def main():
    h = load_human()
    y = h.label.map(E2I).to_numpy()
    X = [clean_full(t) for t in h.tweet]
    pipe = tr.make_pipe(tr.CLASSIC["LogReg"]())
    pipe.fit(X, y)
    v = np.array(pipe[0].get_feature_names_out()); W = pipe[1].coef_
    top = {e: [w for w in v[np.argsort(-W[i])[:25]] if " " not in w][:8] for i, e in enumerate(EMOTIONS)}

    f_full = cv_f1(X, y)
    f_noseed = cv_f1([" ".join(w for w in x.split() if w not in SEEDS) for x in X], y)
    hum_has = np.array([has_seed(t) for t in h.tweet])
    d = {"top_features_logreg": top, "seed_words": sorted(SEEDS),
         "macro_f1_full": f_full, "macro_f1_without_seed_words": f_noseed,
         "chance_macro_f1": 1 / K, "share_human_with_seed": float(hum_has.mean())}

    S = pd.concat([ch.sample(min(len(ch), 10_000), random_state=i) for i, ch in enumerate(iter_corpus())]).reset_index(drop=True)
    cor_has = np.array([has_seed(t) for t in S])
    P = scores(pipe, [clean_full(t) for t in S])
    conf = P.max(1)
    oof = pd.read_csv(RES / "oof_predictions.csv").conf.to_numpy() if (RES / "oof_predictions.csv").exists() else None
    d.update({"n_corpus_sample": len(S), "share_corpus_with_seed": float(cor_has.mean()),
              "conf_mean_corpus": float(conf.mean()), "conf_mean_corpus_with_seed": float(conf[cor_has].mean()),
              "conf_mean_corpus_without_seed": float(conf[~cor_has].mean()),
              "conf_mean_human_oof": float(oof.mean()) if oof is not None else None,
              "share_conf_ge_0.8_corpus": float((conf >= 0.8).mean()),
              "pred_dist_corpus_sample": pd.Series(np.array(EMOTIONS)[P.argmax(1)]).value_counts(normalize=True).reindex(EMOTIONS).round(4).to_dict(),
              "note": "Model klasik LogReg dipakai sebagai diagnostik; bukan pengganti hasil IndoBERTweet."})
    (RES / "diagnostics.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in d.items() if k not in ("seed_words", "top_features_logreg")}, indent=2))


if __name__ == "__main__":
    main()
