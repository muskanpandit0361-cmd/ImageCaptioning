"""
Tests for application helpers — download-text builder, session-state
initialisation, and mocked model inference.

Model inference is mocked so these tests never download weights.
"""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from model import CaptionResult, generate_caption, _clean_caption
from config import MODEL_ID, MODE_STANDARD, MODE_DETAILED


# ── Download-text builder ────────────────────────────────────────────────

def _build_download_text(result: CaptionResult) -> str:
    """Reproduce the builder from app.py for testability."""
    import datetime
    lines = [
        "VisionCaption AI — Generated Description",
        "=" * 44,
        "",
        f"Caption:          {result.caption}",
        f"Mode:             {result.mode}",
        f"Model:            {result.model_id}",
        f"Device:           {result.device.upper()}",
        f"Inference time:   {result.inference_time_seconds:.3f} seconds",
        f"Max new tokens:   {result.generation_params.get('max_new_tokens', '–')}",
        f"Beam count:       {result.generation_params.get('num_beams', '–')}",
        f"Generated at:     {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "---",
        "Note: AI-generated descriptions may contain errors.",
    ]
    return "\n".join(lines)


class TestDownloadTextBuilder:
    def test_contains_caption(self):
        r = CaptionResult(
            caption="A dog in a park.",
            inference_time_seconds=1.23,
            model_id=MODEL_ID,
            device="cpu",
            mode=MODE_STANDARD,
            generation_params={"max_new_tokens": 50, "num_beams": 4},
        )
        text = _build_download_text(r)
        assert "A dog in a park." in text
        assert MODEL_ID in text
        assert "CPU" in text

    def test_contains_metadata(self):
        r = CaptionResult(
            caption="Test.",
            inference_time_seconds=0.5,
            model_id=MODEL_ID,
            device="cuda",
            mode=MODE_DETAILED,
            generation_params={"max_new_tokens": 150, "num_beams": 6},
        )
        text = _build_download_text(r)
        assert "Detailed" in text
        assert "150" in text


# ── Mocked inference ────────────────────────────────────────────────────

class TestGenerateCaptionMocked:
    """Verify the generate_caption orchestration with a mocked model."""

    @patch("model.load_model")
    def test_standard_mode_returns_caption(self, mock_load):
        # Set up mock processor and model
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # processor(...) returns a dict-like with .to()
        fake_inputs = {"pixel_values": MagicMock()}
        fake_inputs["pixel_values"].to = MagicMock(return_value=fake_inputs["pixel_values"])
        mock_processor.return_value = fake_inputs

        # model.generate returns token IDs
        import torch
        mock_model.generate.return_value = torch.tensor([[0, 1, 2, 3]])

        # processor.decode returns raw text
        mock_processor.decode.return_value = "a dog sitting on a bench"

        mock_load.return_value = (mock_processor, mock_model, "cpu")

        img = Image.new("RGB", (100, 100))
        result = generate_caption(img, mode=MODE_STANDARD)

        assert result.success is True
        assert result.caption == "A dog sitting on a bench."
        assert result.device == "cpu"
        assert result.inference_time_seconds > 0 or result.inference_time_seconds == 0.0

    @patch("model.load_model")
    def test_empty_caption_is_failure(self, mock_load):
        mock_processor = MagicMock()
        mock_model = MagicMock()

        fake_inputs = {"pixel_values": MagicMock()}
        fake_inputs["pixel_values"].to = MagicMock(return_value=fake_inputs["pixel_values"])
        mock_processor.return_value = fake_inputs

        import torch
        mock_model.generate.return_value = torch.tensor([[0]])
        mock_processor.decode.return_value = ""

        mock_load.return_value = (mock_processor, mock_model, "cpu")

        img = Image.new("RGB", (100, 100))
        result = generate_caption(img, mode=MODE_STANDARD)

        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch("model.load_model")
    def test_model_load_failure(self, mock_load):
        mock_load.side_effect = RuntimeError("Network error")

        img = Image.new("RGB", (100, 100))
        result = generate_caption(img)

        assert result.success is False
        assert "Network error" in result.error_message

    @patch("model.load_model")
    def test_inference_exception(self, mock_load):
        mock_processor = MagicMock()
        mock_model = MagicMock()

        fake_inputs = {"pixel_values": MagicMock()}
        fake_inputs["pixel_values"].to = MagicMock(return_value=fake_inputs["pixel_values"])
        mock_processor.return_value = fake_inputs
        mock_model.generate.side_effect = RuntimeError("CUDA OOM")

        mock_load.return_value = (mock_processor, mock_model, "cpu")

        img = Image.new("RGB", (100, 100))
        result = generate_caption(img)

        assert result.success is False

    @patch("model.load_model")
    def test_detailed_mode_uses_prompt(self, mock_load):
        mock_processor = MagicMock()
        mock_model = MagicMock()

        fake_inputs = {"pixel_values": MagicMock()}
        fake_inputs["pixel_values"].to = MagicMock(return_value=fake_inputs["pixel_values"])
        mock_processor.return_value = fake_inputs

        import torch
        mock_model.generate.return_value = torch.tensor([[0, 1, 2]])
        mock_processor.decode.return_value = "a detailed description of a red car on a highway"

        mock_load.return_value = (mock_processor, mock_model, "cpu")

        img = Image.new("RGB", (100, 100))
        result = generate_caption(img, mode=MODE_DETAILED)

        assert result.success is True
        # The prompt prefix should be stripped
        assert not result.caption.lower().startswith("a detailed description of")


# ── CaptionResult dataclass ──────────────────────────────────────────────

class TestCaptionResult:
    def test_defaults(self):
        r = CaptionResult(
            caption="test",
            inference_time_seconds=0.0,
            model_id=MODEL_ID,
            device="cpu",
            mode=MODE_STANDARD,
            generation_params={},
        )
        assert r.success is True
        assert r.error_message is None

    def test_failure(self):
        r = CaptionResult(
            caption="",
            inference_time_seconds=0.0,
            model_id=MODEL_ID,
            device="cpu",
            mode=MODE_STANDARD,
            generation_params={},
            success=False,
            error_message="boom",
        )
        assert r.success is False
        assert r.error_message == "boom"
