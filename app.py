"""Dashboard Streamlit EmoTweetID. Membaca hasil di results/ (jalankan skrip 01-04 dulu)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMOTIONS, MODELS, RES  # noqa: E402

st.set_page_config(page_title="EmoTweetID Dashboard", page_icon="📊", layout="wide")
COL = {"anger": "#D64545", "disgust": "#7A8F3A", "fear": "#7E57C2",
       "joy": "#F2B01E", "sadness": "#3F78C4", "surprise": "#E8833A"}
ID = {"anger": "Marah", "disgust": "Jijik", "fear": "Takut", "joy": "Senang", "sadness": "Sedih", "surprise": "Terkejut"}


@st.cache_data
def jload(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def cload(name, **kw):
    p = RES / name
    return pd.read_csv(p, **kw) if p.exists() else None


def need(x, script):
    if x is None:
        st.info(f"Hasil belum tersedia. Jalankan `python {script}` terlebih dahulu.")
        st.stop()
    return x


def fmt(n):
    return f"{int(n):,}".replace(",", ".")


def landis(k):
    if k < 0: return "buruk (< 0)"
    for t, s in [(0.20, "sangat rendah"), (0.40, "cukup"), (0.60, "sedang"), (0.80, "substansial")]:
        if k <= t: return s
    return "hampir sempurna"


def verdicts():
    m, c, h3 = jload("metrics.json"), jload("corpus_summary.json"), jload("h3.json")
    rows = []
    h1 = (m or {}).get("h1")
    if h1:
        ok = h1["bert_f1"] > h1["classic_f1"] and h1["mcnemar_p"] < h1["alpha"]
        rows.append(("H1", f"Macro-F1 IndoBERTweet {h1['bert_f1']:.3f} vs {h1['classic_name']} {h1['classic_f1']:.3f}; McNemar p = {h1['mcnemar_p']:.4g}",
                     "H₀ ditolak, H₁ didukung" if ok else "H₀ tidak dapat ditolak"))
    h2 = (m or {}).get("h2")
    if h2 and "kappa_model_human" in h2:
        lo, hi = h2["diff_ci95"]
        ok = h2["diff"] > 0 and lo > 0
        rows.append(("H2", f"κ model–manusia {h2['kappa_model_human']:.3f} vs κ leksikon–manusia {h2['kappa_lexicon_human']:.3f}; selisih {h2['diff']:.3f} (CI95% {lo:.3f} s.d. {hi:.3f})",
                     "H₀ ditolak, H₁ didukung" if ok else "H₀ tidak dapat ditolak"))
    if h3:
        ok = h3["p"] < 0.05
        rows.append(("H3", f"χ² = {h3['chi2']:.1f}, db = {h3['dof']}, p = {h3['p']:.3g}; Cramér's V = {h3['cramers_v']:.3f}",
                     "H₀ ditolak, H₁ didukung (cek besar efek)" if ok else "H₀ tidak dapat ditolak"))
    return rows


# ------------------------------------------------------------------ sidebar
st.sidebar.title("EmoTweetID")
st.sidebar.caption("Klasifikasi emosi dan pemodelan topik pada 3,1 juta tweet berbahasa Indonesia")
page = st.sidebar.radio("Halaman", ["Ringkasan", "Audit data", "H1 · Kinerja model", "H2 · Kesesuaian label",
                                    "Distribusi emosi korpus", "H3 · Topik per emosi", "Coba prediksi", "Unduh hasil"])
st.sidebar.divider()
st.sidebar.caption("Label hasil model adalah prediksi, bukan label manusia. Data tidak mewakili seluruh tweet Indonesia (diambil berbasis kata kunci emosi).")

# ------------------------------------------------------------------ Ringkasan
if page == "Ringkasan":
    st.title("Ringkasan hasil penelitian")
    m, c, a = jload("metrics.json"), jload("corpus_summary.json"), jload("audit.json")
    cols = st.columns(4)
    cols[0].metric("Tweet pada korpus", fmt(a["n_corpus_raw"]) if a else "–")
    cols[1].metric("Tweet berlabel manusia", fmt(a["n_human"]) if a else "–")
    cols[2].metric("Tweet berlabel model", fmt(c["n_labeled"]) if c else "–")
    cols[3].metric("Model terbaik", m["best_model"] if m else "–",
                   f"macro-F1 {m['models'][m['best_model']]['macro_f1']:.3f}" if m else None)
    if c and c.get("is_sample"):
        st.warning(f"Pelabelan dijalankan pada sampel acak {fmt(c['n_labeled'])} tweet, bukan seluruh korpus.")
    st.subheader("Status pengujian hipotesis")
    v = verdicts()
    if v:
        st.dataframe(pd.DataFrame(v, columns=["Hipotesis", "Statistik", "Keputusan"]), hide_index=True)
        st.caption("Keputusan dihitung otomatis dari hasil di folder results/. Pada data berjumlah jutaan, p-value hampir selalu kecil; nilai ukuran efek perlu dibaca bersama.")
    else:
        st.info("Belum ada hasil pengujian. Jalankan skrip 02 dan 04.")
    if c:
        d = pd.DataFrame({"Emosi": EMOTIONS, "Jumlah": [c["counts_all"][e] for e in EMOTIONS]})
        fig = px.bar(d, x="Emosi", y="Jumlah", color="Emosi", color_discrete_map=COL, title="Distribusi emosi hasil prediksi pada korpus")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

# ------------------------------------------------------------------ Audit
elif page == "Audit data":
    st.title("Audit data")
    a = need(jload("audit.json"), "01_audit.py")
    c = st.columns(4)
    c[0].metric("Tweet korpus (mentah)", fmt(a["n_corpus_raw"]))
    c[1].metric("Duplikat", fmt(a["n_corpus_duplicate"]), f"{a['n_corpus_duplicate'] / a['n_corpus_raw']:.2%}", delta_color="off")
    c[2].metric("Tweet berlabel muncul di korpus", fmt(a["n_overlap_human_in_corpus"]))
    c[3].metric("Berlabel manusia (unik, bersih)", fmt(a["n_human"]))
    l, r = st.columns(2)
    hc = pd.Series(a["human_label_counts"]).reindex(EMOTIONS).reset_index()
    hc.columns = ["Emosi", "Jumlah"]
    l.plotly_chart(px.bar(hc, x="Emosi", y="Jumlah", color="Emosi", color_discrete_map=COL, title="Distribusi label manusia").update_layout(showlegend=False))
    x = list(range(len(a["len_hist_corpus"])))
    fig = go.Figure([go.Scatter(x=x, y=a["len_hist_corpus"], name="Korpus (sampel)", mode="lines"),
                     go.Scatter(x=x, y=a["len_hist_human"], name="Berlabel manusia", mode="lines")])
    fig.update_layout(title="Panjang tweet (kata, ≥60 digabung)", xaxis_title="Jumlah kata", yaxis_title="Proporsi")
    r.plotly_chart(fig)
    ks = a["ks_length_human_vs_corpus"]
    st.markdown(f"**Uji asumsi keterwakilan.** Uji Kolmogorov–Smirnov panjang tweet: D = {ks['statistic']:.3f}, p = {ks['p']:.3g}. "
                f"Rata-rata kata: manusia {a['words_human']['mean']:.1f}, korpus {a['words_corpus']['mean']:.1f}.")
    s = a["share_with_indonesian_function_word"]
    st.markdown(f"**Heuristik bahasa.** Tweet yang memuat minimal satu kata fungsi bahasa Indonesia umum: korpus {s['corpus']:.1%}, berlabel manusia {s['human']:.1%}. Ini perkiraan kasar, bukan deteksi bahasa formal.")
    hc_ = a.get("human_cleaning")
    if hc_:
        st.markdown(f"**Pembersihan data berlabel.** {fmt(hc_['rows_raw'])} baris mentah; {hc_['duplicate_keys']} teks muncul lebih dari sekali "
                    f"({hc_['rows_in_duplicate_keys']} baris), di antaranya {hc_['keys_with_conflicting_labels']} teks diberi label berbeda "
                    f"({hc_['rows_in_conflicting_keys']} baris, dibuang karena ambigu). Tersisa **{fmt(hc_['n_final'])}** tweet unik untuk pemodelan.")
    dg = jload("diagnostics.json")
    if dg:
        st.subheader("Diagnostik validitas: ketergantungan pada kata emosi eksplisit")
        c = st.columns(4)
        c[0].metric("Berlabel memuat kata emosi", f"{dg['share_human_with_seed']:.1%}")
        c[1].metric("Korpus memuat kata emosi", f"{dg['share_corpus_with_seed']:.1%}")
        c[2].metric("Macro-F1 penuh → tanpa kata emosi", f"{dg['macro_f1_full']:.2f} → {dg['macro_f1_without_seed_words']:.2f}",
                    f"acak ≈ {dg['chance_macro_f1']:.2f}", delta_color="off")
        c[3].metric("Keyakinan rata-rata: berlabel vs korpus", f"{dg['conf_mean_human_oof']:.2f} vs {dg['conf_mean_corpus']:.2f}")
        st.caption(f"Sampel korpus {fmt(dg['n_corpus_sample'])} tweet; model diagnostik: regresi logistik TF-IDF. Daftar kata emosi: {', '.join(dg['seed_words'])}. "
                   "Model terutama mengenali kata emosi eksplisit, sedangkan sebagian besar korpus tidak memuatnya; hasil pelabelan korpus perlu dibaca bersama skor keyakinan.")
        tf = pd.DataFrame({e: pd.Series(w) for e, w in dg["top_features_logreg"].items()})
        st.markdown("**Kata berbobot tertinggi per emosi (regresi logistik)**")
        st.dataframe(tf, hide_index=True)
    if "n" in a.get("lexicon", {}):
        lx = a["lexicon"]
        st.markdown(f"**Leksikon.** {fmt(lx['n'])} tweet; {fmt(lx['n_aligned_with_human'])} cocok dengan data manusia; kecocokan mentah dengan label manusia {lx['raw_agreement_with_human']:.1%}.")

# ------------------------------------------------------------------ H1
elif page == "H1 · Kinerja model":
    st.title("H1 · Perbandingan kinerja model")
    m = need(jload("metrics.json"), "02_train_eval.py")
    names = list(m["models"])
    tab = pd.DataFrame([{"Model": k, "Macro-F1": v["macro_f1"], "CI95 bawah": v["macro_f1_ci95"][0], "CI95 atas": v["macro_f1_ci95"][1],
                         "Akurasi": v["accuracy"], "Presisi (macro)": v["macro_precision"], "Recall (macro)": v["macro_recall"],
                         "Rata-rata lipatan": v["fold_macro_f1_mean"], "SD lipatan": v["fold_macro_f1_sd"]} for k, v in m["models"].items()])
    st.dataframe(tab.round(4), hide_index=True)
    fig = go.Figure(go.Bar(x=tab.Model, y=tab["Macro-F1"], error_y=dict(type="data", symmetric=False,
                    array=tab["CI95 atas"] - tab["Macro-F1"], arrayminus=tab["Macro-F1"] - tab["CI95 bawah"])))
    fig.update_layout(title="Macro-F1 out-of-fold dengan interval kepercayaan 95% bootstrap", yaxis_title="Macro-F1")
    st.plotly_chart(fig)
    pc = pd.DataFrame({k: {e: v["per_class"][e]["f1-score"] for e in EMOTIONS} for k, v in m["models"].items()})
    st.plotly_chart(px.imshow(pc.round(3), text_auto=True, aspect="auto", color_continuous_scale="Blues", title="F1 per kelas"))
    sel = st.selectbox("Matriks konfusi untuk model", names, index=names.index(m["best_model"]))
    cm = np.array(m["models"][sel]["confusion"])
    norm = st.toggle("Normalisasi per baris (recall)", True)
    z = cm / cm.sum(1, keepdims=True) if norm else cm
    st.plotly_chart(px.imshow(z, x=EMOTIONS, y=EMOTIONS, text_auto=".2f" if norm else True, color_continuous_scale="Blues",
                              labels=dict(x="Prediksi", y="Label manusia")).update_layout(title=f"Matriks konfusi · {sel}"))
    h1 = m.get("h1")
    if h1:
        st.subheader("Uji McNemar")
        st.markdown(f"IndoBERTweet benar & {h1['classic_name']} salah: **{h1['bert_right_classic_wrong']}**; sebaliknya: **{h1['classic_right_bert_wrong']}**. "
                    f"Statistik = {h1['mcnemar_stat']:.3f}, p = {h1['mcnemar_p']:.4g}.")
    else:
        st.info("IndoBERTweet belum dievaluasi (skrip 02 dijalankan dengan --skip-bert).")

# ------------------------------------------------------------------ H2
elif page == "H2 · Kesesuaian label":
    st.title("H2 · Kesesuaian label (Cohen's κ)")
    m = need(jload("metrics.json"), "02_train_eval.py")
    h2 = m.get("h2", {})
    if "kappa_model_human" not in h2:
        st.error(f"H2 belum dapat dihitung: {h2.get('error', 'data leksikon tidak ada')}")
        st.stop()
    d = pd.DataFrame({"Pasangan": [f"{h2['model_name']} vs manusia", "Leksikon vs manusia"],
                      "κ": [h2["kappa_model_human"], h2["kappa_lexicon_human"]],
                      "lo": [h2["kappa_model_ci95"][0], h2["kappa_lexicon_ci95"][0]],
                      "hi": [h2["kappa_model_ci95"][1], h2["kappa_lexicon_ci95"][1]]})
    fig = go.Figure(go.Bar(x=d["Pasangan"], y=d["κ"], error_y=dict(type="data", symmetric=False, array=d.hi - d["κ"], arrayminus=d["κ"] - d.lo)))
    fig.update_layout(title=f"κ pada {fmt(h2['n_aligned'])} tweet yang sama (CI95% bootstrap)", yaxis_title="Cohen's κ")
    st.plotly_chart(fig)
    c = st.columns(3)
    c[0].metric("κ model–manusia", f"{h2['kappa_model_human']:.3f}", landis(h2["kappa_model_human"]), delta_color="off")
    c[1].metric("κ leksikon–manusia", f"{h2['kappa_lexicon_human']:.3f}", landis(h2["kappa_lexicon_human"]), delta_color="off")
    c[2].metric("Selisih κ", f"{h2['diff']:.3f}", f"CI95% [{h2['diff_ci95'][0]:.3f}, {h2['diff_ci95'][1]:.3f}]", delta_color="off")
    st.caption("Interpretasi menurut Landis & Koch (1977). Prediksi model berasal dari out-of-fold sehingga tidak melihat label tweet yang dinilai.")
    o = cload("oof_predictions.csv")
    if o is not None and "lexicon" in o:
        st.subheader("Contoh tweet saat model dan leksikon berbeda")
        dis = o[(o.pred != o.lexicon) & o.lexicon.notna()]
        st.dataframe(dis.sample(min(30, len(dis)), random_state=1)[["tweet", "label", "pred", "lexicon", "conf"]], hide_index=True)

# ------------------------------------------------------------------ Distribusi
elif page == "Distribusi emosi korpus":
    st.title("Distribusi emosi pada seluruh korpus")
    c = need(jload("corpus_summary.json"), "03_label_corpus.py")
    m = jload("metrics.json")
    if c["is_sample"]:
        st.warning(f"Hasil dari sampel acak {fmt(c['n_labeled'])} tweet.")
    st.caption(f"{fmt(c['n_corpus_raw'])} mentah → {fmt(c['n_after_dedup'])} setelah deduplikasi → {fmt(c['n_after_remove_human'])} setelah membuang tweet berlabel manusia.")
    d = pd.DataFrame({"Emosi": EMOTIONS * 2, "Jumlah": [c["counts_all"][e] for e in EMOTIONS] + [c["counts_high_conf"][e] for e in EMOTIONS],
                      "Subset": ["Semua"] * 6 + [f"Keyakinan tinggi (≥ {c['threshold']:.2f})" if c["threshold"] else "Keyakinan tinggi"] * 6})
    mode = st.radio("Tampilan", ["Jumlah", "Persentase"], horizontal=True)
    if mode == "Persentase":
        d["Jumlah"] = d["Jumlah"] / d.groupby("Subset")["Jumlah"].transform("sum").replace(0, np.nan) * 100
    st.plotly_chart(px.bar(d, x="Emosi", y="Jumlah", color="Subset", barmode="group", title="Sebaran emosi (analisis sensitivitas)"))
    cov = c.get("coverage_by_threshold")
    if cov:
        st.markdown("**Cakupan tweet korpus menurut ambang keyakinan** (rata-rata keyakinan keseluruhan "
                    f"{c['mean_conf_overall']:.2f}): " + "; ".join(f"≥ {k}: {v:.1%}" for k, v in cov.items()))
    if c["threshold"] is None:
        st.info("Tidak ada ambang yang mencapai presisi target pada data validasi, sehingga subset keyakinan tinggi kosong.")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(x=EMOTIONS, y=[c["mean_conf"][e] for e in EMOTIONS], color=EMOTIONS, color_discrete_map=COL,
                          labels=dict(x="Emosi", y="Rata-rata keyakinan"), title="Keyakinan prediksi per emosi").update_layout(showlegend=False))
    r.plotly_chart(px.bar(x=EMOTIONS, y=[c["mean_words"][e] for e in EMOTIONS], color=EMOTIONS, color_discrete_map=COL,
                          labels=dict(x="Emosi", y="Rata-rata kata"), title="Panjang tweet per emosi").update_layout(showlegend=False))
    if m and m.get("pr_curve"):
        pr = pd.DataFrame(m["pr_curve"])
        fig = go.Figure([go.Scatter(x=pr.t, y=pr.precision, name="Presisi"), go.Scatter(x=pr.t, y=pr.coverage, name="Cakupan")])
        fig.update_layout(title="Kurva presisi–cakupan (data out-of-fold)", xaxis_title="Ambang keyakinan")
        st.plotly_chart(fig)
    st.subheader("Kata paling khas per emosi")
    e = st.selectbox("Emosi", EMOTIONS, format_func=lambda x: f"{x} ({ID[x]})")
    w = pd.DataFrame(c["distinctive_words"][e])
    if len(w):
        st.plotly_chart(px.bar(w.iloc[::-1], x="z", y="word", orientation="h", color_discrete_sequence=[COL[e]],
                               labels=dict(z="Skor z log-odds", word="")))

# ------------------------------------------------------------------ H3
elif page == "H3 · Topik per emosi":
    st.title("H3 · Topik dominan per emosi")
    h3 = need(jload("h3.json"), "04_topics.py")
    tj = jload("topics.json")
    ct = cload("topic_emotion_crosstab.csv", index_col=0)
    rs = cload("topic_emotion_residuals.csv", index_col=0)
    c = st.columns(4)
    c[0].metric("χ²", f"{h3['chi2']:.1f}", f"db {h3['dof']}", delta_color="off")
    c[1].metric("p-value", f"{h3['p']:.3g}")
    c[2].metric("Cramér's V", f"{h3['cramers_v']:.3f}", "efek kecil" if h3["cramers_v"] < 0.1 else ("sedang" if h3["cramers_v"] < 0.3 else "besar"), delta_color="off")
    c[3].metric("Koherensi c_v", f"{h3['coherence_cv']:.3f}" if h3.get("coherence_cv") else "–")
    st.caption(f"{h3['n_topics']} topik; {h3['outlier_share']:.1%} tweet tidak masuk topik (outlier) dan dikeluarkan dari uji. Kriteria ukuran efek mengikuti konvensi Cohen untuk tabel besar dan hanya pedoman kasar.")
    topics = [t for t in tj if t["topic"] != -1]
    label = {t["topic"]: f"T{t['topic']}: " + ", ".join(t["words"][:3]) for t in topics}
    z = rs.rename(columns=lambda x: label.get(int(x), x))
    st.plotly_chart(px.imshow(z.round(1), aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=0, text_auto=".1f",
                              title="Residual terstandar emosi × topik (merah: lebih sering dari harapan; biru: lebih jarang)").update_layout(height=520))
    prop = (ct.drop(columns=["-1"], errors="ignore").T / ct.drop(columns=["-1"], errors="ignore").sum(axis=1)).T
    st.plotly_chart(px.imshow(prop.rename(columns=lambda x: label.get(int(x), x)).round(3), aspect="auto", color_continuous_scale="Viridis",
                              title="Proporsi topik di dalam tiap emosi").update_layout(height=450))
    sel = st.selectbox("Rincian topik", [t["topic"] for t in topics], format_func=lambda t: label[t])
    t = next(x for x in topics if x["topic"] == sel)
    st.write("Kata representatif:", ", ".join(t["words"]))
    st.plotly_chart(px.bar(x=EMOTIONS, y=[t["by_emotion"][e] for e in EMOTIONS], color=EMOTIONS, color_discrete_map=COL,
                           labels=dict(x="Emosi", y="Jumlah tweet")).update_layout(showlegend=False))
    mp = cload("topic_map.csv")
    if mp is not None:
        by = st.radio("Warna peta", ["topic", "label"], horizontal=True, format_func=lambda x: "Topik" if x == "topic" else "Emosi")
        mp = mp[mp.topic != -1] if st.checkbox("Sembunyikan outlier", True) else mp
        mp["topic"] = mp.topic.astype(str)
        st.plotly_chart(px.scatter(mp, x="x", y="y", color=by, hover_data=["tweet"], opacity=0.6,
                                   color_discrete_map=COL if by == "label" else None, title="Peta topik (UMAP 2D, sampel)").update_layout(height=560))

# ------------------------------------------------------------------ Prediksi
elif page == "Coba prediksi":
    st.title("Deteksi emosi dari teks")

    @st.cache_resource
    def load_predictor():
        from common import clean_full, clean_min, scores
        meta = MODELS / "final" / "meta.json"
        if meta.exists():
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            ml = json.loads(meta.read_text())["max_len"]
            tok = AutoTokenizer.from_pretrained(MODELS / "final")
            mod = AutoModelForSequenceClassification.from_pretrained(MODELS / "final").eval()

            def f(t):
                enc = tok([clean_min(t)], truncation=True, max_length=ml, return_tensors="pt")
                with torch.no_grad():
                    return torch.softmax(mod(**enc).logits, -1).numpy()[0]
            return f
        p = MODELS / "final.joblib"
        if p.exists():
            import joblib
            pipe = joblib.load(p)
            return lambda t: scores(pipe, [clean_full(t)])[0]
        return None

    f = load_predictor()
    if f is None:
        st.info("Model final belum ada. Jalankan `python 03_label_corpus.py` terlebih dahulu.")
        st.stop()
    txt = st.text_area("Tulis tweet atau kalimat", "Aku kesel banget sama pelayanan hari ini, nggak sesuai janji!")
    if st.button("Prediksi") and txt.strip():
        p = f(txt)
        d = pd.DataFrame({"Emosi": EMOTIONS, "Probabilitas": p})
        st.success(f"Prediksi: **{EMOTIONS[int(p.argmax())]}** ({ID[EMOTIONS[int(p.argmax())]]}) dengan keyakinan {p.max():.1%}")
        st.plotly_chart(px.bar(d, x="Emosi", y="Probabilitas", color="Emosi", color_discrete_map=COL).update_layout(showlegend=False, yaxis_range=[0, 1]))
        st.caption("Prediksi model, bukan penilaian manusia. Model dilatih pada tweet berbasis kata kunci emosi sehingga bisa keliru pada teks yang berbeda gaya.")

# ------------------------------------------------------------------ Unduh
else:
    st.title("Unduh hasil")
    for fname, desc in [("corpus_sample.csv", "Sampel 50.000 tweet berlabel prediksi (dataset turunan)"),
                        ("oof_predictions.csv", "Prediksi out-of-fold pada 2.243 tweet berlabel manusia"),
                        ("metrics.json", "Seluruh metrik evaluasi dan uji H1–H2"),
                        ("corpus_summary.json", "Ringkasan distribusi emosi korpus"),
                        ("h3.json", "Hasil uji asosiasi emosi–topik"),
                        ("topic_emotion_crosstab.csv", "Tabel kontingensi emosi × topik")]:
        p = RES / fname
        if p.exists():
            st.download_button(f"⬇ {fname}", p.read_bytes(), file_name=fname)
            st.caption(desc)
    st.info("Dataset turunan wajib mencantumkan atribusi EmoTweetID (Nugroho et al., 2025; CC BY 4.0).")
