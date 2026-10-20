# System Architecture

## Overview

VisionCaption AI is a modular image-captioning web application built on four
layers that communicate through well-defined Python interfaces.

```
┌───────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)             │
│   Upload · Preview · Mode Select · Results · DL   │
└───────────────┬───────────────────┬───────────────┘
                │                   │
     ┌──────────▼──────┐  ┌────────▼────────┐
     │  image_utils.py │  │    model.py      │
     │  Validate · RGB │  │  Load · Infer    │
     └──────────┬──────┘  └────────┬────────┘
                │                   │
         ┌──────▼──────┐   ┌───────▼────────┐
         │   Pillow     │   │  Transformers  │
         │   (PIL)      │   │  + PyTorch     │
         └─────────────┘   └───────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  BLIP Pretrained │
                          │  Weights (HF Hub)│
                          └─────────────────┘
```

## Module Responsibilities

### `config.py`
- Centralises all tunable constants: model ID, upload limits, generation
  defaults, UI strings.
- No side effects on import.

### `image_utils.py`
- File-extension validation (allowlist-based).
- Upload-size enforcement.
- Safe image opening with `verify()` + `load()`.
- Dimension guard-rails.
- RGBA → RGB compositing with white background.
- Returns `(Image | None, error_message | None)` for clean caller logic.

### `model.py`
- Downloads and caches the BLIP processor + model via `@st.cache_resource`.
- Detects CUDA availability at load time.
- Exposes `generate_caption(image, mode, …) → CaptionResult`.
- Measures inference time with `time.perf_counter()`.
- Cleans decoded output (whitespace, prompt stripping, capitalisation).
- Returns a structured `CaptionResult` dataclass — never raises to the UI.

### `app.py`
- Page configuration and custom CSS injection.
- Session-state management for caption persistence across reruns.
- Two-column layout: upload (left) ↔ results (right).
- Sidebar: generation settings, model info, limitations.
- Header with "How it works" steps.
- Footer with model reference and disclaimer.
- All user-facing errors rendered as `st.error()` — no raw tracebacks.

## Inference Pipeline

```
1. User selects an image file via st.file_uploader
2. image_utils.validate_uploaded_file() runs:
   a. Extension check
   b. Size check
   c. Image.open → verify → load
   d. Dimension check
   e. RGB conversion
3. User clicks "Generate Description"
4. model.generate_caption() runs:
   a. Loads processor + model (cached)
   b. Processor converts PIL Image → tensor dict
   c. Inputs moved to device (CPU or CUDA)
   d. model.generate() produces token IDs (beam search)
   e. Processor decodes tokens → raw text
   f. _clean_caption() normalises output
5. CaptionResult returned → rendered in the right column
6. User can copy, download, or regenerate
```

## Caching Strategy

| What            | Mechanism              | Lifetime              |
|-----------------|------------------------|-----------------------|
| Model weights   | `@st.cache_resource`   | Process lifetime      |
| Caption result  | `st.session_state`     | Browser session       |
| Uploaded image  | In-memory bytes only   | Single request cycle  |

## Design Decisions

1. **BLIP Base over Large** — ~990 MB vs ~1.8 GB.  Fits within
   Streamlit Community Cloud's memory constraints.
2. **No fine-tuning at runtime** — inference only; reproducible across
   machines with the same model revision.
3. **Deterministic beam search** — no sampling; outputs are reproducible
   for the same image and settings.
4. **Modular model interface** — `generate_caption()` can be pointed at
   a different model by editing `model.py` without touching `app.py`.
5. **No database** — keeps deployment simple; session state is ephemeral.
