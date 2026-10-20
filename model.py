"""
VisionCaption AI — Model Module

Encapsulates all interaction with the Salesforce/blip-image-captioning-base
model.  Loading is performed once and cached via Streamlit's ``@st.cache_resource``
so that the ~990 MB download only happens on the first run (or after a cache
eviction).

The public API consists of:
    load_model()        → (processor, model, device_str)
    generate_caption()  → CaptionResult
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from config import (
    MODEL_ID,
    MODE_STANDARD,
    MODE_DETAILED,
    GENERATION_DEFAULTS,
)

logger = logging.getLogger(__name__)

# Set a download timeout so the app fails fast instead of hanging silently
# when the network is slow or unreachable (default in HF hub is very high).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


# ── Data class for inference results ─────────────────────────────────────

@dataclass
class CaptionResult:
    """Container for a single captioning result."""
    caption: str
    inference_time_seconds: float
    model_id: str
    device: str
    mode: str
    generation_params: dict
    success: bool = True
    error_message: Optional[str] = None


# ── Model loading (cached) ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration, str]:
    """Download / load the BLIP captioning model and return (processor, model, device).

    The function is decorated with ``@st.cache_resource`` so the heavy
    model object is shared across all Streamlit reruns within the same
    process.

    Returns
    -------
    processor : BlipProcessor
    model     : BlipForConditionalGeneration
    device    : str   – ``"cuda"`` or ``"cpu"``

    Raises
    ------
    RuntimeError
        When the model cannot be loaded (network, disk, or OOM).
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading BLIP model '%s' on device '%s' …", MODEL_ID, device)

        status = st.status("🔄 Loading BLIP model…", expanded=True)
        status.write("⬇️ Downloading / loading processor (tokenizer)…")
        t0 = time.perf_counter()
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        status.write(f"✅ Processor ready ({time.perf_counter() - t0:.1f}s)")

        status.write("⬇️ Downloading / loading model weights (~990 MB on first run)…")
        t1 = time.perf_counter()
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        status.write(f"✅ Weights loaded ({time.perf_counter() - t1:.1f}s)")

        status.write(f"🚀 Moving model to {device.upper()}…")
        model = model.to(device)
        model.eval()

        status.update(label="✅ Model ready!", state="complete", expanded=False)
        logger.info("Model loaded successfully on %s.", device)
        return processor, model, device

    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "Not enough GPU memory to load the model. "
            "Try restarting with CPU-only mode or a machine with more VRAM."
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to download or locate model weights. "
            f"Check your internet connection and disk space. Detail: {exc}"
        )
    except Exception as exc:
        logger.error("Model loading failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Unexpected error while loading the model: {exc}"
        )


# ── Caption generation ───────────────────────────────────────────────────

def generate_caption(
    image: Image.Image,
    mode: str = MODE_STANDARD,
    max_new_tokens: Optional[int] = None,
    num_beams: Optional[int] = None,
) -> CaptionResult:
    """Run inference on *image* and return a ``CaptionResult``.

    Parameters
    ----------
    image : PIL.Image.Image
        An RGB image ready for the processor.
    mode : str
        ``"Standard Caption"`` or ``"Detailed Description"``.
    max_new_tokens : int, optional
        Override the default maximum new tokens.
    num_beams : int, optional
        Override the default beam count.

    Returns
    -------
    CaptionResult
        Always returned — check ``.success`` and ``.error_message`` on failure.
    """
    # Resolve generation parameters
    if mode == MODE_DETAILED:
        tokens = max_new_tokens or GENERATION_DEFAULTS.max_new_tokens_detailed
        text_prompt: Optional[str] = GENERATION_DEFAULTS.detailed_prompt
    else:
        tokens = max_new_tokens or GENERATION_DEFAULTS.max_new_tokens
        text_prompt = None

    beams = num_beams or GENERATION_DEFAULTS.num_beams

    gen_params = {
        "max_new_tokens": tokens,
        "num_beams": beams,
        "min_new_tokens": GENERATION_DEFAULTS.min_new_tokens,
        "repetition_penalty": GENERATION_DEFAULTS.repetition_penalty,
        "length_penalty": GENERATION_DEFAULTS.length_penalty,
        "mode": mode,
    }

    try:
        processor, model, device = load_model()
    except RuntimeError as exc:
        return CaptionResult(
            caption="",
            inference_time_seconds=0.0,
            model_id=MODEL_ID,
            device="unknown",
            mode=mode,
            generation_params=gen_params,
            success=False,
            error_message=str(exc),
        )

    try:
        # Prepare inputs
        if text_prompt:
            inputs = processor(image, text=text_prompt, return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=tokens,
                num_beams=beams,
                min_length=GENERATION_DEFAULTS.min_new_tokens,
                repetition_penalty=GENERATION_DEFAULTS.repetition_penalty,
                length_penalty=GENERATION_DEFAULTS.length_penalty,
            )
        elapsed = time.perf_counter() - start

        # Decode
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        caption = _clean_caption(caption, text_prompt)

        if not caption:
            return CaptionResult(
                caption="",
                inference_time_seconds=elapsed,
                model_id=MODEL_ID,
                device=device,
                mode=mode,
                generation_params=gen_params,
                success=False,
                error_message="The model returned an empty caption.",
            )

        return CaptionResult(
            caption=caption,
            inference_time_seconds=elapsed,
            model_id=MODEL_ID,
            device=device,
            mode=mode,
            generation_params=gen_params,
        )

    except torch.cuda.OutOfMemoryError:
        return CaptionResult(
            caption="",
            inference_time_seconds=0.0,
            model_id=MODEL_ID,
            device=device,
            mode=mode,
            generation_params=gen_params,
            success=False,
            error_message=(
                "GPU ran out of memory during inference. "
                "Try a smaller image or switch to CPU."
            ),
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        return CaptionResult(
            caption="",
            inference_time_seconds=0.0,
            model_id=MODEL_ID,
            device=device,
            mode=mode,
            generation_params=gen_params,
            success=False,
            error_message=f"Inference error: {exc}",
        )


# ── Internal helpers ─────────────────────────────────────────────────────

def _clean_caption(raw: str, prompt: Optional[str] = None) -> str:
    """Normalise whitespace, capitalise, strip the conditioning prompt."""
    text = " ".join(raw.split())  # collapse whitespace

    # When a text prompt was used the decoded string starts with it
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
        # Also strip a leading comma / colon left over
        text = text.lstrip(",: ").strip()

    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."

    return text


def get_model_info() -> dict:
    """Return static model metadata for display in the UI."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "model_id": MODEL_ID,
        "architecture": "BLIP (Vision Transformer + Language Decoder)",
        "framework": "PyTorch + Hugging Face Transformers",
        "device": device,
        "license": "BSD-3-Clause",
        "generation_method": "Beam-search decoding on encoder-decoder architecture",
    }
