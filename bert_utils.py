import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from config import BERT_MODEL, K, SEED

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = DEV.type == "cuda"


@torch.no_grad()
def predict_proba(model, tok, texts, bs=128, max_len=96, progress=False):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), K), dtype=np.float32)
    rng = range(0, len(texts), bs)
    for i in (tqdm(rng) if progress else rng):
        j = order[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=CUDA):
            lg = model(**enc).logits
        out[j] = torch.softmax(lg.float(), -1).cpu().numpy()
    return out


def fine_tune(tr_x, tr_y, va_x, va_y, lr=3e-5, bs=16, epochs=4, max_len=96, patience=2,
              seed=SEED, model_name=BERT_MODEL):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=K).to(DEV)
    tr_y = np.asarray(tr_y)
    cnt = np.bincount(tr_y, minlength=K)
    w = torch.tensor(len(tr_y) / (K * np.maximum(cnt, 1)), dtype=torch.float, device=DEV)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr_x) / bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=CUDA)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(tr_x))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            enc = tok([tr_x[k] for k in j], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=CUDA):
                lg = model(**enc).logits
            loss = loss_fn(lg.float(), torch.tensor(tr_y[j], device=DEV))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
        f = f1_score(va_y, predict_proba(model, tok, va_x, max_len=max_len).argmax(1), average="macro")
        print(f"   epoch {ep + 1}: val macro-F1 = {f:.4f}")
        if f > best:
            best, bad = f, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, tok, best
