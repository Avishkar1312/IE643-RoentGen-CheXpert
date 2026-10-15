"""Stage 4c: vision-language correctness with BiomedCLIP (open stand-in for the paper's Tables 3-4).

    python -m roentgen.eval_clip --tag C_step15000 [--n 2000 --size 256]
    python -m roentgen.eval_clip --tag real            # upper bound: real test images in place of synthetic
    (--device cpu runs it alongside a GPU job)

This substitutes for the paper's RadGraph / F1-CheXbert / ConVIRT / CXR-RePaiR stack, which needs
MIMIC-credentialed weights. It is not equivalent to that stack. BiomedCLIP was trained on PubMed
figure captions, not on CXR reports.

Uses the FID set from eval_fidelity: image i was generated from prompts_for(reference_rows)[i]. Each row is
put in one of six classes (exactly one of the five targets, or "normal" = No Finding with no target).
Rows with several targets, or with only unmodelled pathology (generic prompt), have no clean class and
are left out of the class-based metrics.
  * alignment: cosine(image, its own prompt) vs cosine(image, a prompt from another class); the
    gap is a CLIP-score margin over all classed rows. Zero-shot 6-way accuracy picks the best of
    one fixed text per class.
  * image-image retrieval: 200 class-balanced synthetic queries against 400 class-balanced real candidates
    (test rows outside the reference set). Precision@{5,10,50} counts label matches. The "real" tag gives
    the upper bound: the real images of the same query rows.
  * image-text retrieval: query = image, candidates = every templated prompt for the six classes.
    Precision@{1,5,10} counts label matches.

Writes results/clip_<tag>_<size>px.json.
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from roentgen.config import LABELS, NORMAL, SEED, TARGETS
from roentgen.data import read_manifest
from roentgen.eval_fidelity import EVAL_DIR, RESULTS, load_real, prompts_for, reference_rows, resize
from roentgen.prompts import FINDING_TEMPLATES, NORMAL_TEMPLATES, PHRASES

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASSES = TARGETS + ["normal"]
ZS_TEXT = {**{t: f"chest x-ray with {PHRASES[t]}" for t in TARGETS}, "normal": "normal frontal chest x-ray"}
N_QUERY, N_CAND = 200, 400


def classes_of(df):
    """Class index in CLASSES, or -1 for multi-target / generic rows."""
    y = df[LABELS].fillna(0).values == 1
    nt = y[:, :len(TARGETS)].sum(1)
    c = np.full(len(df), -1)
    c[nt == 1] = y[nt == 1, :len(TARGETS)].argmax(1)
    c[(nt == 0) & y[:, LABELS.index(NORMAL)]] = len(TARGETS)
    return c


def balanced(cls, n, rng):
    """Up to n // 6 indices per class (all of them if a class is short)."""
    per = n // len(CLASSES)
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(cls == k))[:per] for k in range(len(CLASSES))]))


def prompt_bank():
    texts, cls = [], []
    for k, t in enumerate(TARGETS):
        for tpl in FINDING_TEMPLATES:
            for v in ("PA", "AP"):
                texts.append(tpl.format(findings=PHRASES[t], view=v)); cls.append(k)
    for tpl in NORMAL_TEMPLATES:
        for v in ("PA", "AP"):
            texts.append(tpl.format(view=v)); cls.append(len(TARGETS))
    texts, idx = np.unique(texts, return_index=True)    # templates without {view} appear twice
    return list(texts), np.array(cls)[idx]


class BiomedCLIP:
    def __init__(self, device):
        import open_clip
        self.model, self.preprocess = open_clip.create_model_from_pretrained(MODEL)
        self.tok = open_clip.get_tokenizer(MODEL)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def images(self, imgs, bs=64):
        out = []
        for b in range(0, len(imgs), bs):
            x = torch.stack([self.preprocess(Image.fromarray(im).convert("RGB")) for im in imgs[b:b + bs]])
            out.append(self.model.encode_image(x.to(self.device), normalize=True).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def texts(self, texts, bs=256):
        out = []
        for b in range(0, len(texts), bs):
            t = self.tok(texts[b:b + bs], context_length=256).to(self.device)
            out.append(self.model.encode_text(t, normalize=True).float().cpu())
        return torch.cat(out).numpy()


def precision_at(sim, q_cls, c_cls, ks):
    order = np.argsort(-sim, axis=1)
    hit = c_cls[order] == q_cls[:, None]
    res = {f"P@{k}": float(hit[:, :k].mean()) for k in ks}
    res["per_class_P@10"] = {CLASSES[c]: float(hit[q_cls == c, :10].mean()) for c in np.unique(q_cls)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="eval dir name, e.g. C_step15000, sd, or 'real'")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = reference_rows("test", a.n)
    prompts = prompts_for(rows)
    cls = classes_of(rows)
    if a.tag == "real":
        imgs = load_real(rows, a.size)
    else:
        d = EVAL_DIR / a.tag / "fid"
        paths = [d / f"{i:05d}.png" for i in range(a.n)]
        assert all(p.exists() for p in paths), f"incomplete FID set in {d}; run eval_fidelity first"
        imgs = resize(np.stack([np.asarray(Image.open(p)) for p in paths]), a.size)

    clip = BiomedCLIP(a.device)
    emb = clip.images(imgs)
    res = {"args": vars(a), "model": MODEL, "class_counts": {c: int((cls == k).sum()) for k, c in enumerate(CLASSES)}}

    # --- alignment: own prompt vs a random prompt from a different class
    rng = np.random.default_rng(SEED)
    keep = np.flatnonzero(cls >= 0)
    p_emb = clip.texts(prompts)
    own = (emb * p_emb).sum(1)
    other = np.array([rng.choice(keep[cls[keep] != cls[i]]) for i in keep])
    mism = (emb[keep] * p_emb[other]).sum(1)
    zs = emb[keep] @ clip.texts([ZS_TEXT[c] for c in CLASSES]).T
    pred = zs.argmax(1)
    res["alignment"] = {
        "clip_own": float(own[keep].mean()), "clip_mismatched": float(mism.mean()),
        "margin": float((own[keep] - mism).mean()), "frac_own_gt_mismatched": float((own[keep] > mism).mean()),
        "clip_own_all_rows": float(own.mean()),
        "zeroshot_acc": float((pred == cls[keep]).mean()), "zeroshot_chance": 1 / len(CLASSES),
        "zeroshot_per_class_acc": {c: float((pred[cls[keep] == k] == k).mean()) for k, c in enumerate(CLASSES)},
        "zeroshot_text": ZS_TEXT,
    }

    # --- image-image retrieval against real candidates outside the reference set
    q = balanced(cls, N_QUERY, np.random.default_rng(SEED))
    cand_path = EVAL_DIR / f"biomedclip_cand_{a.size}px.npz"
    if cand_path.exists():
        z = np.load(cand_path)
        c_emb, c_cls = z["emb"], z["cls"]
    else:
        test = read_manifest("test")
        test = test[~test.rel_path.isin(set(rows.rel_path))].reset_index(drop=True)
        ci = balanced(classes_of(test), N_CAND, np.random.default_rng(SEED + 1))
        cdf = test.iloc[ci].reset_index(drop=True)
        c_emb, c_cls = clip.images(load_real(cdf, a.size)), classes_of(cdf)
        np.savez(cand_path, emb=c_emb, cls=c_cls, rel_path=cdf.rel_path.values)
    res["image_image"] = {"n_query": len(q), "n_cand": len(c_cls),
                          "chance": float(np.mean([(c_cls == cls[i]).mean() for i in q])),
                          **precision_at(emb[q] @ c_emb.T, cls[q], c_cls, (5, 10, 50))}

    # --- image-text retrieval against the prompt bank
    bank, b_cls = prompt_bank()
    res["image_text"] = {"bank_size": len(bank),
                         "chance": float(np.mean([(b_cls == cls[i]).mean() for i in keep])),
                         **precision_at(emb[keep] @ clip.texts(bank).T, cls[keep], b_cls, (1, 5, 10))}

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"clip_{a.tag}_{a.size}px.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("class_counts", "alignment", "image_image", "image_text")}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
