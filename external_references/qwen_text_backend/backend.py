"""Frozen Qwen-Drive VLM using the original VQA processor and batched generation.

Only padding/batching, strict VLM loading, and cache diagnostics are implemented here.
The upstream planning wrapper and task heads are never instantiated.
"""
import hashlib
import importlib
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import BatchFeature

from .prompts import CAMERAS, SYSTEM_MESSAGE
from .cache import description_errors

DEFAULT_SOURCE = Path(__file__).resolve().parents[4] / "Qwen-drive/Qwen-Drive-1.0-main/src"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[4] / "Qwen-drive/checkpoint"


def source_identity(source_path=DEFAULT_SOURCE):
    source_path = Path(source_path).resolve()
    package = source_path / "qwen_drive"
    if not (package / "scene.py").is_file():
        raise FileNotFoundError(f"Qwen-Drive source package not found in {source_path}")
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}


def import_qwen_drive(source_path=DEFAULT_SOURCE):
    source_path = Path(source_path).resolve()
    source_identity(source_path)
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    package = importlib.import_module("qwen_drive")
    if Path(package.__file__).resolve().parent != source_path / "qwen_drive":
        raise RuntimeError(f"Another qwen_drive package was imported: {package.__file__}")
    return package


class QwenDriveTextProcessor:
    def __init__(self, checkpoint_path, max_pixels=None, source_path=DEFAULT_SOURCE):
        package = import_qwen_drive(source_path)
        from transformers import AutoTokenizer

        self.config = package.QwenDriveConfig.from_pretrained(checkpoint_path, local_files_only=True)
        if max_pixels is not None:
            if max_pixels < 4 * (self.config.image_patch_size * self.config.image_spatial_merge_size) ** 2:
                raise ValueError("max_pixels is below Qwen-Drive's native minimum pixel budget")
            self.config.current_image_pixels = max_pixels
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Qwen-Drive tokenizer must define a padding token")
        self.original = package.QwenDriveProcessor(self.tokenizer, self.config)

    def encode_batch(self, image_batches, questions, system=SYSTEM_MESSAGE):
        if not image_batches or len(image_batches) != len(questions):
            raise ValueError("A nonempty image list is required for every question")
        # These calls are upstream CPU preprocessing, not per-scene GPU generation.
        samples = [self.original.encode_vqa(images, question, system=system, device="cpu")
                   for images, question in zip(image_batches, questions)]
        width = max(item["input_ids"].shape[1] for item in samples)
        padded_ids, masks = [], []
        for images, sample in zip(image_batches, samples):
            if sample["image_grid_thw"].shape[0] != len(images):
                raise RuntimeError("Qwen-Drive processor lost an image")
            ids = sample["input_ids"]
            pad = width - ids.shape[1]
            padded_ids.append(torch.nn.functional.pad(ids, (pad, 0), value=self.tokenizer.pad_token_id))
            masks.append(torch.nn.functional.pad(torch.ones_like(ids), (pad, 0), value=0))
        input_ids = torch.cat(padded_ids)
        return BatchFeature({
            "input_ids": input_ids,
            "attention_mask": torch.cat(masks),
            "mm_token_type_ids": (input_ids == self.config.vlm_config.image_token_id).long(),
            "pixel_values": torch.cat([item["pixel_values"] for item in samples]),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in samples]),
        })

    def rendered_prompt(self, inputs, index):
        ids = inputs["input_ids"][index][inputs["attention_mask"][index].bool()]
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def clean_answer(self, text):
        from qwen_drive.modeling_qwen_drive import _strip_thinking
        return _strip_thinking(text)


def load_processor(checkpoint_path, max_pixels=None, source_path=DEFAULT_SOURCE):
    return QwenDriveTextProcessor(checkpoint_path, max_pixels, source_path)


class QwenTextDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class QwenTextCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, records):
        images = [[record["camera_files"][name] for name, _ in CAMERAS] for record in records]
        inputs = self.processor.encode_batch(images, [record["question"] for record in records])
        return records, inputs


def decode_generated_batch(processor, generated_ids, input_width, eos_token_id):
    """Trim padded prompt width; retain raw output and upstream final-answer extraction."""
    suffixes = generated_ids[:, input_width:]
    texts = processor.batch_decode(suffixes, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    results = []
    for raw_text, ids in zip(texts, suffixes.tolist()):
        stop = next((i for i, token in enumerate(ids) if token in eos), None)
        text = getattr(processor, "clean_answer", str.strip)(raw_text)
        errors = description_errors(text)
        if stop is None:
            errors.append("truncated")
        results.append({
            "text": text.strip(), "raw_text": raw_text.strip(),
            "generated_tokens": len(ids) if stop is None else stop + 1,
            "finish_reason": "length" if stop is None else "eos",
            "complete": stop is not None and bool(text.strip()),
            "usable": not errors, "validation_errors": errors,
        })
    return results


class FrozenQwenTextGenerator:
    def __init__(self, checkpoint_path, processor, device="cuda:0", attn_implementation="eager"):
        from safetensors import safe_open
        from transformers import Qwen3_5ForConditionalGeneration

        self.processor = processor
        # Inspect only the root VLM; neither task-head directory is read.
        with safe_open(str(Path(checkpoint_path) / "model.safetensors"), framework="pt", device="cpu") as weights:
            keys = list(weights.keys())
        if not keys or not all(key.startswith("vlm.") for key in keys):
            raise ValueError("Expected Qwen-Drive root weights containing only vlm.* tensors")
        self.model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint_path, config=processor.config.vlm_config,
            key_mapping={r"^vlm\.": ""}, dtype=torch.bfloat16, device_map=device,
            attn_implementation=attn_implementation, local_files_only=True, output_loading_info=True,
        )
        errors = {key: [str(v) for v in value] for key, value in info.items() if value}
        if errors:
            raise RuntimeError(f"VLM weights did not load completely: {errors}")
        self.model.eval().requires_grad_(False)
        if processor.config.vlm_config.tie_word_embeddings:
            if self.model.get_input_embeddings().weight is not self.model.get_output_embeddings().weight:
                raise RuntimeError("VLM input/output embeddings were not tied")
        self.loading_report = {"checkpoint_tensors": len(keys), "loading_errors": errors,
                               "model_class": type(self.model).__name__,
                               "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                               "planner_instantiated": False, "perception_instantiated": False}

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens):
        from qwen_drive.modeling_qwen_drive import VQA_DECODE_DEFAULTS

        inputs = inputs.to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        params = dict(VQA_DECODE_DEFAULTS)
        torch.manual_seed(params.pop("seed"))
        params.pop("presence_penalty", None)  # Upstream removes this unsupported kwarg too.
        params.update(max_new_tokens=max_new_tokens, use_cache=True,
                      pad_token_id=self.processor.tokenizer.pad_token_id)
        generated_ids = self.model.generate(**inputs, **params)
        return decode_generated_batch(
            self.processor, generated_ids.cpu(), inputs["input_ids"].shape[1],
            self.model.generation_config.eos_token_id,
        )
