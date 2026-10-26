"""Tahap 4: BERTopic pada sampel berstrata + uji H3 (chi-square, Cramer's V, residual terstandar).
Keluaran: results/topics.json, topic_emotion_crosstab.csv, topic_emotion_residuals.csv, topic_map.csv, h3.json"""
import argparse
import json
import numpy as np
import pandas as pd
from config import EMOTIONS, RES, SEED
from common import STOP, association_stats, clean_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-emotion", type=int, default=20000)
    ap.add_argument("--high-conf-only", action="store_true")
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--min-topic-size", type=int, default=50)
    ap.add_argument("--nr-topics", type=int, default=30)
    ap.add_argument("--coh-docs", type=int, default=20000)
    a = ap.parse_args()

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    df = pd.read_parquet(RES / "corpus_labeled.parquet")
    if a.high_conf_only:
        tau = json.loads((RES / "corpus_summary.json").read_text())["threshold"]
        df = df[df.conf >= tau]
    df = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), a.per_emotion), random_state=SEED)).reset_index(drop=True)
    docs = [clean_full(t) for t in df.tweet]
    keep = [i for i, d in enumerate(docs) if len(d.split()) >= 2]
    df, docs = df.iloc[keep].reset_index(drop=True), [docs[i] for i in keep]
    print(f"{len(docs):,} tweet untuk pemodelan topik")

    emb = SentenceTransformer(a.embed_model).encode(docs, batch_size=128, show_progress_bar=True)
    tm = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
                  hdbscan_model=HDBSCAN(min_cluster_size=a.min_topic_size, metric="euclidean", cluster_selection_method="eom"),
                  vectorizer_model=CountVectorizer(stop_words=list(STOP), min_df=10),
                  nr_topics=a.nr_topics, calculate_probabilities=False, verbose=True)
    topics, _ = tm.fit_transform(docs, emb)
    df["topic"] = topics

    # koherensi c_v
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    toks = [d.split() for d in docs[: a.coh_docs]]
    dic = Dictionary(toks)
    tw = [[w for w, _ in tm.get_topic(t)[:10] if w in dic.token2id] for t in sorted(set(topics)) if t != -1]
    tw = [w for w in tw if len(w) >= 2]
    cv = float(CoherenceModel(topics=tw, texts=toks, dictionary=dic, coherence="c_v").get_coherence()) if tw else None

    # tabel kontingensi + H3 (tanpa outlier -1)
    ct = pd.crosstab(df.label, df.topic).reindex(EMOTIONS).fillna(0)
    ct_in = ct.drop(columns=[-1], errors="ignore")
    stats, resid = association_stats(ct_in)
    stats.update({"outlier_share": float((df.topic == -1).mean()), "n_topics": int(ct_in.shape[1]), "coherence_cv": cv})
    ct.to_csv(RES / "topic_emotion_crosstab.csv"); resid.to_csv(RES / "topic_emotion_residuals.csv")
    (RES / "h3.json").write_text(json.dumps(stats, indent=2))

    info = tm.get_topic_info()
    tj = [{"topic": int(r.Topic), "count": int(r.Count),
           "words": [w for w, _ in tm.get_topic(int(r.Topic))[:10]] if r.Topic != -1 else [],
           "by_emotion": {e: int(ct.loc[e, r.Topic]) if r.Topic in ct.columns else 0 for e in EMOTIONS}}
          for r in info.itertuples()]
    (RES / "topics.json").write_text(json.dumps(tj, indent=2, ensure_ascii=False))

    ix = np.random.default_rng(SEED).choice(len(df), size=min(10000, len(df)), replace=False)
    xy = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb[ix])
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "topic": df.topic.values[ix], "label": df.label.values[ix],
                  "tweet": df.tweet.values[ix]}).to_csv(RES / "topic_map.csv", index=False)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
