"""Tahap 1: audit data (Bab III.G.1). Keluaran: results/audit.json"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from config import RES, SEED
from common import ID_HINT, iter_corpus, key_hashes, load_human, load_lexicon, word_counts


def hint_ratio(texts):
    return float(np.mean([any(w in ID_HINT for w in str(t).lower().split()) for t in texts]))


def hist(wc, hi=60):
    h, _ = np.histogram(np.clip(wc, 0, hi), bins=np.arange(0, hi + 2))
    return (h / h.sum()).round(5).tolist()


def main():
    human = load_human()
    hh = key_hashes(human.tweet)
    hashes, wcs, samples = [], [], []
    rng = np.random.default_rng(SEED)
    for i, ch in enumerate(iter_corpus()):
        hashes.append(key_hashes(ch))
        wcs.append(word_counts(ch).astype(np.int32))
        samples.append(ch.sample(min(len(ch), 12_000), random_state=SEED + i))
        print(f"  potongan {i + 1}: {sum(len(x) for x in hashes):,} tweet dibaca")
    hashes = np.concatenate(hashes)
    wc_c = np.concatenate(wcs)
    n_raw = len(hashes)
    uniq = np.unique(hashes)
    dup = int(n_raw - len(uniq))
    overlap = int(np.isin(hh, uniq).sum())
    wc_h = word_counts(human.tweet)
    sub = rng.choice(wc_c, size=min(100_000, n_raw), replace=False)
    ks = ks_2samp(sub, wc_h)
    samp_c = pd.concat(samples).sample(100_000, random_state=SEED)

    a = {"n_corpus_raw": n_raw, "n_corpus_duplicate": dup, "n_corpus_unique": n_raw - dup,
         "n_human": len(human), "human_cleaning": load_human.report, "n_overlap_human_in_corpus": overlap,
         "human_label_counts": human.label.value_counts().to_dict(),
         "words_corpus": {"mean": float(wc_c.mean()), "median": float(np.median(wc_c)),
                          "p95": float(np.percentile(wc_c, 95))},
         "words_human": {"mean": float(wc_h.mean()), "median": float(np.median(wc_h)),
                         "p95": float(np.percentile(wc_h, 95))},
         "len_hist_corpus": hist(sub), "len_hist_human": hist(wc_h),
         "ks_length_human_vs_corpus": {"statistic": float(ks.statistic), "p": float(ks.pvalue)},
         # heuristik kasar: tweet tanpa satu pun kata fungsi Indonesia umum
         "share_with_indonesian_function_word": {"corpus": hint_ratio(samp_c), "human": hint_ratio(human.tweet)}}
    try:
        lex = load_lexicon()
        m = human.merge(lex, on="key", suffixes=("_h", "_l"))
        a["lexicon"] = {"n": len(lex), "n_aligned_with_human": len(m),
                        "label_counts": lex.label.value_counts().to_dict(),
                        "raw_agreement_with_human": float((m.label_h == m.label_l).mean()) if len(m) else None}
    except Exception as e:  # noqa
        a["lexicon"] = {"error": str(e)}
    (RES / "audit.json").write_text(json.dumps(a, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in a.items() if not k.startswith("len_hist")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
