
from masking import mask
from video_processor import clock


# -----------------------------------------
# YOUR CODE: run phase 4 on the output of phases 1 to 3 (needs the GPU laptop on)
# usage: python run_masks.py
# -----------------------------------------

result = mask()

print(f"Prompts : {result['prompts']}\n")

for s in result["segments"]:

    if s.get("skipped"):
        print(f"[{s['id']:2d}] {s['mode']:<17} skipped: {s['skipped']}")
        continue

    found = {p: f"{o['keyframes_found']}/{len(s['keyframes'])} kf, score {o['best_score']:.2f}, area {o['mean_area_frac']:.3f}" for p, o in s["objects"].items()}

    print(f"[{s['id']:2d}] {s['mode']:<17} {found}")

print("\nSaved: processed/masks/masks.json")
