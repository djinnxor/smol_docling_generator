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

DEFAULT_MODEL_ID = "ibm-granite/granite-docling-258M"
SYSTEM_PROMPT = "Convert this document page to DocTags format."


class DocTagDataset(Dataset):

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
            truncation=False,
            max_length=self.max_length,
        ).input_ids

        # Full pass: include the image so pixel_values are produced correctly
        inputs = self.processor(
            images=image,
            text=text_full,
            return_tensors="pt",
            truncation=False,
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