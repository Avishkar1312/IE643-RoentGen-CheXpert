"""Stage 6 demo: prompt -> synthetic chest X-ray, with the judge classifier's reading of the result.

    source env.sh && python app/gradio_demo.py                 # http://127.0.0.1:7860
    python app/gradio_demo.py --share                          # public link (e.g. from Kaggle)
    python app/gradio_demo.py --step 22500                     # another run-C checkpoint

Loads run C (full U-Net @256) from artifacts/runs/C/ckpt-<step> and the 5-label DenseNet judge
(artifacts/cls/real_f1.0_s0/best.pt). Tick pathologies to build a training-style prompt, or type a
free-text prompt (it wins when non-empty). The side panel shows the judge's probability per label;
a finding is "called" when it exceeds the 5%-FPR threshold calibrated on real test negatives
(qualitative.Judge, the same rule as the qualitative figures; calibrating takes ~1 min at start-up).
"""
import argparse

import gradio as gr
import numpy as np
import torch
from PIL import Image

from roentgen.compare_runs import build_pipe
from roentgen.config import TARGETS
from roentgen.prompts import PHRASES, join_findings
from roentgen.qualitative import Judge
EXAMPLES = [
    ["frontal chest radiograph showing a large right-sided pleural abnormality", [], 4.0, 25, 0],
    ["", ["Pneumonia"], 4.0, 25, 1],
    ["", ["Enlarged Cardiomediastinum"], 5.5, 25, 2],
    ["", ["Lung Lesion", "Fracture"], 4.0, 25, 3],
    ["normal frontal chest x-ray", [], 4.0, 25, 4],
]


def build_prompt(text, findings):
    if text.strip():
        return text.strip()
    if not findings:
        return "normal frontal chest x-ray"
    return f"frontal chest radiograph showing {join_findings([PHRASES[t] for t in TARGETS if t in findings])}"


def make_demo(step):
    pipe, judge = build_pipe("C", step), Judge(256)

    @torch.no_grad()
    def generate(text, findings, guidance, steps, seed):
        prompt = build_prompt(text, findings)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            im = pipe(prompt, height=256, width=256, num_inference_steps=int(steps), guidance_scale=guidance,
                      generator=torch.Generator("cuda").manual_seed(int(seed)), output_type="np").images[0]
        u8 = (im.mean(-1) * 255).round().astype(np.uint8)
        p = judge(u8[None])[0]
        called = [t for t, v, thr in zip(TARGETS, p, judge.thr) if v > thr]
        summary = f"**Prompt:** {prompt}\n\n**Called at 5% FPR:** {', '.join(called) or 'none'}"
        return Image.fromarray(u8).resize((512, 512), Image.BICUBIC), dict(zip(TARGETS, map(float, p))), summary

    with gr.Blocks(title="RoentGen on CheXpert") as demo:
        gr.Markdown(f"# RoentGen on CheXpert: text → chest X-ray\nRun C (full U-Net fine-tune of SD 1.4 @256 px), "
                    f"checkpoint {step}. Research demo only, not for clinical use.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Free-text prompt (optional; overrides the checkboxes)")
                findings = gr.CheckboxGroup(TARGETS, label="Pathologies (none = normal)")
                guidance = gr.Slider(1.0, 9.0, value=4.0, step=0.5, label="Guidance scale")
                steps = gr.Slider(10, 75, value=25, step=1, label="DPM++ steps")
                seed = gr.Number(value=0, precision=0, label="Seed")
                go = gr.Button("Generate", variant="primary")
            with gr.Column():
                image = gr.Image(label="Synthetic CXR (256 px, shown at 512)", type="pil")
                info = gr.Markdown()
                probs = gr.Label(label="Judge classifier probability", num_top_classes=len(TARGETS))
        go.click(generate, [text, findings, guidance, steps, seed], [image, probs, info])
        gr.Examples(EXAMPLES, [text, findings, guidance, steps, seed])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=15000)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    make_demo(a.step).launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
