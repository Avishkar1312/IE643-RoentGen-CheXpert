"""Label vector -> templated radiology prompt (CheXpert has labels, not reports)."""
import random

from roentgen.config import LABELS, NORMAL, TARGETS

PHRASES = {
    "Pleural Other": "pleural abnormality",
    "Pneumonia": "pneumonia",
    "Lung Lesion": "lung lesion",
    "Fracture": "fracture",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
}

FINDING_TEMPLATES = [
    "chest x-ray with {findings}",
    "frontal chest radiograph showing {findings}",
    "{view} chest x-ray demonstrating {findings}",
    "{view} chest radiograph with {findings}",
]
NORMAL_TEMPLATES = [
    "chest radiograph, no acute cardiopulmonary findings",
    "normal frontal chest x-ray",
    "{view} chest x-ray, no acute cardiopulmonary abnormality",
]
# Images with none of the five targets and not "No Finding" carry other (unmodelled)
# pathology; give them a generic prompt rather than a false "normal" one.
GENERIC_TEMPLATES = ["chest x-ray", "frontal chest radiograph", "{view} chest x-ray"]

# Evaluation-only prompts (never seen in training) for the paper's Fig. 3 modifier demo.
MODIFIER_PROMPTS = [
    "small left-sided pleural abnormality",
    "large right-sided pleural abnormality",
    "right lower lobe pneumonia",
    "left upper lobe pneumonia",
    "small right upper lobe lung lesion",
    "large left lower lobe lung lesion",
    "left-sided rib fracture",
    "markedly enlarged cardiomediastinum",
]


def join_findings(findings: list[str]) -> str:
    if len(findings) == 1:
        return findings[0]
    if len(findings) == 2:
        return f"{findings[0]} and {findings[1]}"
    return ", ".join(findings[:-1]) + f", and {findings[-1]}"


def make_prompt(labels, view: str = "PA", rng: random.Random | None = None) -> str:
    """labels: sequence aligned with config.LABELS (0/1)."""
    rng = rng or random
    present = [t for t, v in zip(LABELS, labels) if v and t in TARGETS]
    view = "AP" if str(view).upper() == "AP" else "PA"
    if present:
        rng.shuffle(present)
        tpl = rng.choice(FINDING_TEMPLATES)
        return tpl.format(findings=join_findings([PHRASES[p] for p in present]), view=view)
    tpl = rng.choice(NORMAL_TEMPLATES if labels[LABELS.index(NORMAL)] else GENERIC_TEMPLATES)
    return tpl.format(view=view)
