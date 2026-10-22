"""
Run 6: Blinds and anonymizes completed episodes for independent LLM Judge evaluation.
Slices inputs into batches for parallel evaluation.
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path


def prepare_inputs(slice_size: int = 60):
    base_dir = Path("results/real_world/run6_local")
    tasks_path = Path("results/real_world/new_benchmark_80/tasks.jsonl")

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = {t["task_id"]: t for t in (json.loads(line) for line in f if line.strip())}

    cb_path = base_dir / "runner_cognitive_browser.jsonl"
    pw_path = base_dir / "baseline_playwright_mcp.jsonl"
    bm_path = base_dir / "baseline_cognitive_bm25.jsonl"

    all_eps = []
    for p, arm in [(cb_path, "cognitive_browser"), (pw_path, "playwright_mcp"), (bm_path, "cognitive_bm25")]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ep = json.loads(line)
                        if ep.get("status") != "infra_error":
                            all_eps.append((arm, ep))

    print(f"Total valid episodes loaded across arms: {len(all_eps)}")

    random.seed(42)
    random.shuffle(all_eps)

    blind_dir = base_dir / "blind_screenshots"
    blind_dir.mkdir(parents=True, exist_ok=True)

    judge_inputs = []
    key_map = {}

    for idx, (arm, ep) in enumerate(all_eps):
        blind_id = f"eval_r6_{idx+1:04d}"
        t_id = ep["task_id"]
        task_def = tasks.get(t_id, {})
        orig_screen = ep.get("screenshot_path", "")
        blind_screen = ""
        if orig_screen and os.path.exists(orig_screen):
            blind_screen = str(blind_dir / f"{blind_id}.png")
            if os.path.exists(blind_screen) or os.path.islink(blind_screen):
                os.remove(blind_screen)
            try:
                os.symlink(os.path.abspath(orig_screen), blind_screen)
            except Exception:
                shutil.copy2(orig_screen, blind_screen)

        key_map[blind_id] = {
            "arm": arm,
            "task_id": t_id,
            "repetition": ep.get("repetition", 1),
            "original_sample_id": ep.get("sample_id", ""),
            "original_screenshot": orig_screen,
            "status": ep.get("status", ""),
            "error_message": ep.get("error_message", ""),
        }
        judge_inputs.append({
            "sample_id": blind_id,
            "task_id": t_id,
            "task_goal": task_def.get("goal", ""),
            "verification_criteria": task_def.get("verification", {}),
            "final_url": ep.get("final_url", ""),
            "page_snippet": ep.get("page_snippet", "")[:3500],
            "screenshot_path": blind_screen,
        })

    out_inputs = base_dir / "judge_inputs_anonymized.jsonl"
    with open(out_inputs, "w", encoding="utf-8") as f_in:
        for item in judge_inputs:
            f_in.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_key = base_dir / "anonymization_key.json"
    with open(out_key, "w", encoding="utf-8") as f_k:
        json.dump(key_map, f_k, indent=2, ensure_ascii=False)

    print(f"Prepared {len(judge_inputs)} blinded inputs in {out_inputs}")
    print(f"Saved anonymization key to {out_key}")

    slices_dir = base_dir / "judge_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    for old_f in slices_dir.glob("*.jsonl"):
        old_f.unlink()

    num_slices = (len(judge_inputs) + slice_size - 1) // slice_size
    for s_idx in range(num_slices):
        chunk = judge_inputs[s_idx * slice_size : (s_idx + 1) * slice_size]
        slice_f = slices_dir / f"slice_{s_idx+1:02d}.jsonl"
        with open(slice_f, "w", encoding="utf-8") as f_s:
            for item in chunk:
                f_s.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Created slice {slice_f.name} with {len(chunk)} samples")


if __name__ == "__main__":
    prepare_inputs()
