"""
Fine-tune ibm-granite/granite-docling-258M on Mongolian document pages
to output DocTag format.

The script loads (image, doctag) pairs from JSONL files produced by
prepare_dataset.py, fine-tunes the vision-language model using the
HuggingFace Trainer API, and saves the result to --output_dir.

LoRA (parameter-efficient fine-tuning) is strongly recommended to reduce
VRAM requirements. Pass --lora to enable it (requires peft).

Usage:
    # 1. Build the training dataset first
    python prepare_dataset.py --xml_dir dataset/xml_annotations \\
                               --images_dir dataset/uploads

    # 2. Train
    python train.py --train_data training_data_train.jsonl \\
                    --val_data   training_data_val.jsonl   \\
                    --lora                                 \\
                    --bf16                                 \\
                    --output_dir doctag_model
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# Default model to fine-tune
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"

# Prompt sent to the model together with each page image
SYSTEM_PROMPT = "Convert this document page to DocTags format."


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DocTagDataset(Dataset):
    """
    Loads (image_path, doctag_string) pairs from a JSONL file and returns
    tokenised inputs ready for the Trainer.

    Label masking strategy
    ----------------------
    The full conversation (prompt + assistant response) is tokenised once.
    Tokens that belong to the *prompt* part are masked to -100 so the loss
    is computed only on the assistant's DocTag output.
    """

    def __init__(self, jsonl_path: str, processor, max_length: int = 4096):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        doctag = record["doctag"]

        # --- build full conversation (prompt + response) ---
        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": doctag}],
            },
        ]
        text_full = self.processor.apply_chat_template(
            messages_full, add_generation_prompt=False
        )

        # --- build prompt-only conversation (to find where response starts) ---
        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, add_generation_prompt=True
        )

        # Tokenise both sequences (without images for the prompt-only pass)
        prompt_ids = self.processor.tokenizer(
            text_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        # Full pass: include the image so pixel_values are produced correctly
        inputs = self.processor(
            images=image,
            text=text_full,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # Remove the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Build labels: mask all prompt tokens with -100
        labels = inputs["input_ids"].clone()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100

        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

def doctag_data_collator(features: list[dict]) -> dict:
    """
    Pad a batch of samples produced by DocTagDataset to the same length.
    pixel_values may have an extra patches dimension; handle both cases.
    """
    from torch.nn.utils.rnn import pad_sequence

    batch: dict = {}
    for key in features[0].keys():
        tensors = [f[key] for f in features]
        if key == "labels":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
        elif key == "attention_mask":
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        elif key == "pixel_values":
            # SmolVLM pixel_values shape: [C, H, W] or [num_patches, C, patch_H, patch_W]
            # torch.stack works for both as long as all items share non-batch dims
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                # Variable patch counts: fall back to padding along dim 0
                max_patches = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_patches - t.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        elif key == "pixel_attention_mask":
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                max_patches = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_patches - t.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)
                        t = torch.cat([t, padding], dim=0)
                    padded.append(t)
                batch[key] = torch.stack(padded)
        else:
            batch[key] = pad_sequence(tensors, batch_first=True, padding_value=0)

    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ibm-granite/granite-docling-258M for DocTag generation"
    )
    parser.add_argument(
        "--train_data",
        default="training_data_train.jsonl",
        help="Path to training JSONL (default: training_data_train.jsonl)",
    )
    parser.add_argument(
        "--val_data",
        default="training_data_val.jsonl",
        help="Path to validation JSONL; skipped if the file does not exist",
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID to fine-tune (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output_dir",
        default="doctag_model",
        help="Directory where the fine-tuned model is saved (default: doctag_model)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device training batch size (default: 1)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum token length (default: 2048)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (recommended)")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning (requires peft)",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05)",
    )
    args = parser.parse_args()

    if not Path(args.train_data).exists():
        raise FileNotFoundError(
            f"Training data not found: {args.train_data}\n"
            "Run prepare_dataset.py first."
        )

    # ------------------------------------------------------------------
    # Load processor and model
    # ------------------------------------------------------------------
    print(f"Loading processor and model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    torch_dtype = (
        torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=torch_dtype)
    # Gradient checkpointing (enabled below) is incompatible with use_cache.
    # Set it explicitly here so the Trainer does not emit a warning and the
    # model behaves consistently during both training and evaluation.
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # Optionally apply LoRA
    # ------------------------------------------------------------------
    if args.lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning. Install it with:\n"
                "  pip install peft"
            ) from exc

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    print("Building datasets …")
    train_dataset = DocTagDataset(args.train_data, processor, args.max_length)

    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(args.val_data, processor, args.max_length)
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples | No validation file found at {args.val_data}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
    )

    print("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
