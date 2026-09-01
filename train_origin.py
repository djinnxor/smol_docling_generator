"""
Fine-tune ibm-granite/granite-docling-258M on Mongolian document pages
to output DocTag format.

LoRA (parameter-efficient fine-tuning) is strongly recommended to reduce
VRAM requirements. Pass --lora to enable it (requires peft).

Usage:
    # 1. Build the training dataset first
    python prepare_dataset.py --xml_dir dataset/xml_annotations \\
                               --images_dir dataset/uploads

    # 2. Train
    python train_origin.py --train_data training_data_train.jsonl \\
                           --val_data   training_data_val.jsonl   \\
                           --lora                                 \\
                           --bf16                                 \\
                           --output_dir doctag_model

DocTag-level scoring (tag accuracy, loc MAE, tree edit distance) lives in
doctag_metrics.py and is meant to be run against *generated* output by an
evaluation script, not against teacher-forced logits during training.
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

from mlops import load_project

# argparse fields copied into the MLOps run record
TRACKED = {"model_id", "train_data", "val_data", "output_dir", "epochs", "batch_size",
           "grad_accum", "lr", "max_length", "fp16", "bf16", "lora", "lora_r", "lora_alpha"}
# throughput numbers that are not worth recording as run metrics
SKIP = {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "epoch", "step"}

# Default model to fine-tune
DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"

SYSTEM_PROMPT = "Convert this document page to DocTags format."


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

    def __init__(self, jsonl_path: str, processor, max_length: int = 1024):
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

    def _process(self, image: Image.Image, text: str) -> dict:
        """Run the processor, repeating the page image if the chat template
        emitted more than one <image> placeholder."""
        try:
            return self.processor(images=image, text=text, return_tensors="pt")
        except ValueError as exc:
            count = text.count("<image>")
            if "number of images" not in str(exc) or count <= 1:
                raise
            return self.processor(images=[image] * count, text=text, return_tensors="pt")

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        # Normalise <image>...</image> structural tags to <picture>...</picture>
        # so the processor does not mistake them for image placeholders.
        doctag = record["doctag"].replace("<image>", "<picture>").replace("</image>", "</picture>")

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True
        )
        text_full = self.processor.apply_chat_template(
            prompt_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": doctag}]}
            ],
            add_generation_prompt=False,
        )

        with Image.open(record["image"]) as handle:
            image = handle.convert("RGB")

        # Do NOT truncate inside the processor: it can drop image placeholder
        # tokens from the text while leaving pixel_values unchanged, which the
        # model rejects as a token-count mismatch.  Truncate the 1-D tensors
        # below instead, once the prompt length is known.
        inputs = {k: v.squeeze(0) for k, v in self._process(image, text_full).items()}

        # The processor expands the single <image> placeholder into hundreds of
        # image tokens, so the prompt length cannot be measured by tokenising
        # the prompt *text*.  text_full is text_prompt plus a suffix holding no
        # placeholders, so the prompt region is whatever remains once the
        # suffix's tokens are subtracted from the processed sequence.
        suffix_ids = self.processor.tokenizer(
            text_full[len(text_prompt):], add_special_tokens=False
        ).input_ids
        prompt_len = len(inputs["input_ids"]) - len(suffix_ids)

        if len(inputs["input_ids"]) > self.max_length:
            if prompt_len >= self.max_length:
                raise ValueError(
                    f"{record['image']}: the prompt alone is {prompt_len} tokens but "
                    f"--max_length is {self.max_length}; raise --max_length."
                )
            inputs = {k: (v[: self.max_length] if v.dim() == 1 else v)
                      for k, v in inputs.items()}

        # Build labels: mask all prompt tokens with -100
        labels = inputs["input_ids"].clone()
        labels[:prompt_len] = -100
        inputs["labels"] = labels
        return inputs


def doctag_data_collator(features: list[dict]) -> dict:
    """
    Pad a batch of samples produced by DocTagDataset to the same length.
    pixel_values may have an extra patches dimension; handle both cases.
    """
    from torch.nn.utils.rnn import pad_sequence

    batch: dict = {}
    for key in features[0]:
        tensors = [f[key] for f in features]
        if key in ("pixel_values", "pixel_attention_mask"):
            # SmolVLM shape: [C, H, W] or [num_patches, C, patch_H, patch_W].
            # torch.stack works for both when all items share non-batch dims.
            try:
                batch[key] = torch.stack(tensors)
            except RuntimeError:
                # Variable patch counts: pad along dim 0 before stacking.
                max_patches = max(t.shape[0] for t in tensors)
                batch[key] = torch.stack([
                    torch.cat([t, torch.zeros(max_patches - t.shape[0], *t.shape[1:],
                                              dtype=t.dtype)])
                    if t.shape[0] < max_patches else t
                    for t in tensors
                ])
        else:
            batch[key] = pad_sequence(
                tensors, batch_first=True, padding_value=-100 if key == "labels" else 0
            )

    return batch


def _parse_args() -> argparse.Namespace:
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
        default=2,
        help="Gradient accumulation steps (default: 2)",
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
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05)",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to a previously trained LoRA adapter directory (must contain adapter_config.json). Loads adapter weights with a fresh optimizer.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory to resume training from, or 'last' "
            "to automatically resume from the latest checkpoint in --output_dir. "
            "Restores model weights, optimizer state, scheduler, and global step."
        ),
    )
    return parser.parse_args()


def _build_model(args: argparse.Namespace):
    print(f"Loading processor and model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    torch_dtype = (
        torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else torch.float32)
    )
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=torch_dtype)
    # Gradient checkpointing (enabled below) is incompatible with use_cache.
    model.config.use_cache = False

    if args.lora or args.lora_adapter_path:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning. Install it with:\n"
                "  pip install peft"
            ) from exc

        if args.lora_adapter_path:
            adapter_path = Path(args.lora_adapter_path)
            if not (adapter_path / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"adapter_config.json not found in {args.lora_adapter_path}"
                )
            print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
            model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
            ))
        model.print_trainable_parameters()

    return processor, model


def _resolve_checkpoint(args: argparse.Namespace) -> str | None:
    """Turn --resume_from_checkpoint into a concrete directory, or None."""
    checkpoint = args.resume_from_checkpoint
    if checkpoint is None:
        return None

    if checkpoint.lower() == "last":
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint = get_last_checkpoint(args.output_dir)
        if checkpoint is None:
            print(
                f"WARNING: --resume_from_checkpoint=last specified but no checkpoint "
                f"found in {args.output_dir!r}.  Starting from scratch."
            )
            return None
        print(f"Resuming from latest checkpoint: {checkpoint}")
        return checkpoint

    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    print(f"Resuming from checkpoint: {checkpoint}")
    return checkpoint


def _train(args: argparse.Namespace, run) -> None:
    processor, model = _build_model(args)

    print("Building datasets …")
    train_dataset = DocTagDataset(args.train_data, processor, args.max_length)

    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = DocTagDataset(args.val_data, processor, args.max_length)
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    else:
        print(f"Train: {len(train_dataset)} samples | No validation file found at {args.val_data}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doctag_data_collator,
    )

    print("Starting training …")
    try:
        result = trainer.train(resume_from_checkpoint=_resolve_checkpoint(args))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.  Saving emergency checkpoint …")
        emergency_dir = os.path.join(args.output_dir, "emergency_checkpoint")
        trainer.save_model(emergency_dir)
        processor.save_pretrained(emergency_dir)
        print(f"Emergency checkpoint saved to {emergency_dir}")
        raise

    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    run.log_metrics(train_loss=round(result.training_loss, 4))
    if val_dataset is not None:
        evals = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
        final = evals[-1] if evals else trainer.evaluate()
        run.log_metrics({"val_" + k.removeprefix("eval_"): round(v, 4)
                         for k, v in final.items()
                         if isinstance(v, (int, float)) and k not in SKIP})
    run.log_artifact(args.output_dir, os.path.abspath(args.output_dir))


def main() -> None:
    args = _parse_args()

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")

    writer = load_project("granite-docling")
    with writer.run(
        args.model_id,
        datasets={"ds-001": "train"},
        mode="finetune",
        notes=f"train_origin.py -> {args.output_dir}",
        architecture=(f"LoRA r={args.lora_r} alpha={args.lora_alpha}"
                      if args.lora else "full finetune"),
    ) as run:
        run.log_params({k: v for k, v in vars(args).items() if k in TRACKED})
        _train(args, run)

    print("Done.")


if __name__ == "__main__":
    main()
