"""train_id.py — Mongolian ID fine-tune on top of checkpoint-5400.

Thin wrapper around train_v7.py tuned for the 524-sample id_data.jsonl task.
Splits the dataset 90/10 on first run, then delegates training to train_v7.py
with ID-specific hyperparameters and the 12 ID field tags injected as special
tokens before LoRA adapter load.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformers import AutoProcessor

ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "id_data.jsonl"
TRAIN = ROOT / "id_data_train_split.jsonl"
VAL   = ROOT / "id_data_val_split.jsonl"
CKPT  = ROOT / "doctag_model_origin" / "checkpoint-5400"
OUT   = ROOT / "doctag_model_id"

MODEL_ID = "ibm-granite/granite-docling-258M"

ID_TAGS = [
    "<register_number>", "</register_number>",
    "<gender>",          "</gender>",
    "<familyname>",      "</familyname>",
    "<surname>",         "</surname>",
    "<name>",            "</name>",
    "<birthday>",        "</birthday>",
]


def ensure_split() -> None:
    if TRAIN.exists() and VAL.exists():
        print(f"[Split] Using existing splits: {TRAIN.name} + {VAL.name}")
        return
    if not DATA.exists():
        raise FileNotFoundError(f"Source data not found: {DATA}")
    sys.path.insert(0, str(ROOT))
    from utils.json_splitter import shuffle_jsonl, split_json_file

    shuffle_jsonl(str(DATA), seed=42)
    split_json_file(str(DATA), str(TRAIN), str(VAL), split_ratio=0.90)


def report_tag_tokenization() -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    tok = proc.tokenizer
    vocab = tok.get_vocab()
    single = [t for t in ID_TAGS if t in vocab]
    missing = [t for t in ID_TAGS if t not in vocab]
    multi = [
        t for t in missing
        if len(tok.encode(t, add_special_tokens=False)) > 1
    ]
    print(f"[Tags] {len(single)}/{len(ID_TAGS)} already in vocab.")
    if missing:
        print(f"[Tags] Missing (will be added): {missing}")
    if multi:
        print(f"[Tags] Multi-token decomposition that will be replaced: {multi}")


def main() -> None:
    ensure_split()
    report_tag_tokenization()
    OUT.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(ROOT / "train_v7.py"),
        "--model_id",   MODEL_ID,
        "--train_data", str(TRAIN),
        "--val_data",   str(VAL),
        "--output_dir", str(OUT),
        "--adapter", str(CKPT),
        "--extra_special_tokens", ",".join(ID_TAGS),

        # --- dataset cadence (524 samples, 90/10 split → ~471 train) ---
        "--epochs",                "30",
        "--batch_size",            "2",
        "--grad_accum",            "8",          # effective batch = 16
        "--eval_batch_size",       "1",
        "--eval_steps",            "25",
        "--save_steps",            "25",
        "--save_total_limit",      "5",
        "--max_length",            "2048",       # image tokens (~1088) + doctag

        # --- learning rate: low, we resume an already-adapted LoRA ---
        "--lr",                    "5e-6",
        "--vision_lr_scale",       "0.2",        # vision_lr = 1e-6
        "--layer_decay",           "0.90",

        # --- LoRA (match checkpoint-5400: r=32, alpha=32) ---
        "--lora_r",                "32",
        "--lora_alpha",            "32",
        "--lora_dropout",          "0.15",       # higher dropout for tiny data

        # --- vision encoder: fully unfrozen per user request.
        #     setup_vision_partial_unfreeze() uses all_blocks[-N:] so 99
        #     effectively unfreezes every ViT block. ---
        "--unfreeze_vision_layers", "99",

        # --- Mongolian enhancement stack ---
        "--mongolian_aug_p",       "0.9",        # phone photos: heavier aug
        "--w_mongolian",           "15.0",
        "--lambda_homoglyph",      "0.3",
        "--homoglyph_margin",      "2.0",

        # --- DocTagLoss upweighting: the 6 ID tags are what matters ---
        "--w_doc",                 "25.0",
        "--w_struct",              "8.0",
        "--w_loc",                 "6.0",        # tight bbox supervision
        "--w_fcel",                "2.0",        # field contents (names, dates)
        "--lambda_loc",            "0.5",
        "--lambda_focal",          "1.0",
        "--focal_gamma",           "2.0",
    ]

    print("\n[train_id] Launching:\n  " + " \\\n  ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
