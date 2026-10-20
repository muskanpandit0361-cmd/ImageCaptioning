# VisionCaption AI — MCA Project Report

> **Academic Documentation Outline**
>
> This document provides a structured outline for an MCA final-year project
> report.  Fill in each section with your institutional formatting, page
> numbers, and any additional requirements from your university.

---

## 1. Title Page

- **Project Title:** VisionCaption AI — Intelligent Image Description Generator
- **Degree:** Master of Computer Applications (MCA)
- **Student Name:** [Your Name]
- **Roll Number:** [Your Roll Number]
- **Guide/Supervisor:** [Supervisor Name]
- **Institution:** [University / College Name]
- **Academic Year:** 2026–2027

---

## 2. Abstract

VisionCaption AI is an AI-powered web application that generates natural-language
descriptions of uploaded images.  The system uses the BLIP (Bootstrapped
Language-Image Pre-training) model, a pretrained vision-language transformer,
to analyse visual content and produce contextually relevant English captions.
Built with Python, Streamlit, PyTorch, and Hugging Face Transformers, the
application demonstrates a complete machine learning inference pipeline —
from image preprocessing through neural network inference to post-processed
textual output.  Evaluation is performed using BLEU and ROUGE-L metrics
against human reference captions.

---

## 3. Introduction

### 3.1 Background

Advances in deep learning, particularly transformer architectures and
vision-language models, have made it possible to generate natural-language
descriptions of images automatically.  This capability has applications in
accessibility (alt-text generation for visually impaired users), content
indexing, medical image reporting, and social media.

### 3.2 Scope

This project implements an end-to-end image captioning system using a
pretrained model.  It covers:

- A functional web interface for image upload and caption display.
- Real-time inference using the BLIP vision-language model.
- Quantitative evaluation against reference captions.
- Deployment guidance for cloud hosting.

### 3.3 Organisation of the Report

Brief description of each chapter.

---

## 4. Problem Statement

Conventional image classification systems assign discrete labels (e.g.,
"cat", "car") but do not describe the overall scene.  There is a need for
systems that produce human-readable descriptions capturing objects, actions,
relationships, and scene context.

---

## 5. Motivation

- Growing need for automated image descriptions for accessibility.
- Advances in vision-language models making accurate captioning feasible.
- Educational value of implementing a complete AI inference pipeline.
- Portfolio demonstration of applied Generative AI skills.

---

## 6. Objectives

1. Build a functional web application that accepts image uploads.
2. Integrate a pretrained vision-language model for caption generation.
3. Implement robust image validation and error handling.
4. Provide configurable generation parameters (beam count, token limit).
5. Evaluate caption quality using established NLP metrics.
6. Document the architecture, limitations, and deployment process.

---

## 7. Existing System

Discuss existing approaches to image captioning:

- Rule-based template systems with object detection.
- Encoder-decoder architectures (e.g., Show-and-Tell, Show-Attend-and-Tell).
- Commercial APIs (Google Cloud Vision, Azure Computer Vision, AWS Rekognition).

---

## 8. Limitations of Existing Approaches

- Rule-based systems cannot generalise to unseen scenes.
- Training custom models requires large annotated datasets and GPU resources.
- Commercial APIs require payment and send images to external servers.
- Most academic implementations lack a usable front-end interface.

---

## 9. Proposed System

VisionCaption AI addresses these limitations by:

- Using a pretrained model (no custom training needed).
- Running inference locally (no API keys, no data sent externally).
- Providing a polished web UI with validation, settings, and downloads.
- Supporting quantitative evaluation against reference captions.

---

## 10. Literature Review

Summarise relevant papers (use actual published papers; do not fabricate):

1. **Vinyals et al. (2015)** — Show and Tell: A Neural Image Caption Generator. CVPR.
2. **Xu et al. (2015)** — Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML.
3. **Anderson et al. (2018)** — Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering. CVPR.
4. **Li et al. (2022)** — BLIP: Bootstrapped Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML.
5. **Radford et al. (2021)** — Learning Transferable Visual Models from Natural Language Supervision. ICML. (CLIP)

---

## 11. Hardware and Software Requirements

### Hardware

| Component | Minimum            | Recommended          |
|-----------|--------------------|----------------------|
| CPU       | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7   |
| RAM       | 4 GB               | 8 GB+                |
| GPU       | Not required        | NVIDIA GPU (CUDA)    |
| Storage   | 2 GB free           | 5 GB free            |
| Network   | Required (initial)  | Required (initial)   |

### Software

| Component        | Version / Technology               |
|------------------|------------------------------------|
| OS               | Windows 10+, Linux, macOS          |
| Python           | 3.10 or 3.11                       |
| Web framework    | Streamlit ≥ 1.28                   |
| AI framework     | PyTorch ≥ 2.0                      |
| Model library    | Hugging Face Transformers ≥ 4.30   |
| Image library    | Pillow ≥ 9.5                       |
| IDE              | VS Code or Google Colab            |
| Version control  | Git + GitHub                       |

---

## 12. Functional Requirements

| ID    | Requirement               | Description                                         |
|-------|---------------------------|-----------------------------------------------------|
| FR-01 | Image Upload              | Upload JPG/JPEG/PNG/WEBP up to 10 MB                |
| FR-02 | Image Preprocessing       | Validate, open, convert to RGB                       |
| FR-03 | AI Caption Generation     | BLIP model inference, decode tokens, display caption |
| FR-04 | Description Modes         | Standard (concise) and Detailed (longer)             |
| FR-05 | Results Display           | Caption, metadata, copy, download                    |
| FR-06 | Error Handling            | Graceful messages for all failure modes               |
| FR-07 | Session Management        | Persist caption across reruns, avoid model reload     |

---

## 13. Non-Functional Requirements

| Requirement      | Description                                             |
|------------------|---------------------------------------------------------|
| Performance      | Caption generated within 2–10 s on CPU                  |
| Usability        | Clean, modern UI navigable without training             |
| Reliability      | All error paths handled; no raw tracebacks              |
| Security         | No permanent storage of uploads; no credential exposure |
| Maintainability  | Modular code with clear module boundaries               |
| Portability      | Runs on Windows, Linux, macOS with Python 3.10+         |

---

## 14. System Architecture

Include the architecture diagram from `docs/architecture.md`.

---

## 15. Data-Flow Diagram

```
Level 0 (Context Diagram):

   [User] ──── Upload Image ────► [VisionCaption AI System] ──── Caption ────► [User]


Level 1:

   [User]
     │
     ▼
   ┌────────────────┐
   │  Image Upload   │
   │  & Validation   │
   └───────┬────────┘
           │ Valid RGB Image
           ▼
   ┌────────────────┐
   │  BLIP Model     │
   │  Inference      │
   └───────┬────────┘
           │ Token IDs
           ▼
   ┌────────────────┐
   │  Post-process   │
   │  & Display      │
   └───────┬────────┘
           │ Caption + Metadata
           ▼
       [User]
```

---

## 16. Use-Case Diagram

Primary actors: **User**, **BLIP Model**

Use cases:
- Upload Image
- View Image Preview
- Select Description Mode
- Configure Generation Settings
- Generate Caption
- Copy Caption
- Download Caption
- View Model Information
- View Error Message

---

## 17. Module Descriptions

### 17.1 Configuration Module (`config.py`)
Centralises all constants — model ID, upload limits, generation parameters.

### 17.2 Image Utilities Module (`image_utils.py`)
Validates file extension, size, and content; opens images safely; converts to RGB.

### 17.3 Model Module (`model.py`)
Loads BLIP model (cached); runs inference; cleans decoded text; returns structured results.

### 17.4 Application Module (`app.py`)
Streamlit page configuration, custom CSS, session state, UI layout, user interaction.

### 17.5 Evaluation Module (`evaluation/evaluate.py`)
Batch evaluation against reference captions using BLEU and ROUGE-L metrics.

---

## 18. Algorithm and Model Workflow

### BLIP Inference Pipeline

1. User uploads image → raw bytes.
2. Validate extension, size, and image integrity.
3. Convert to RGB using Pillow.
4. BLIP processor converts PIL Image to normalised tensor.
5. Vision Transformer encoder extracts feature embeddings.
6. Text decoder generates token IDs via beam search.
7. Processor decodes tokens to text.
8. Post-processing: strip prompt, normalise whitespace, capitalise.
9. Return structured `CaptionResult` to the UI.

### Key Parameters

- `num_beams`: Number of beams for beam search (default: 4).
- `max_new_tokens`: Maximum tokens to generate (default: 50 standard, 150 detailed).
- `repetition_penalty`: Penalise repeated n-grams (default: 1.5).

---

## 19. Implementation Details

Describe key implementation decisions:

- Why Streamlit was chosen over Flask/FastAPI for this use case.
- How `@st.cache_resource` prevents repeated model downloads.
- How session state preserves captions across Streamlit reruns.
- How RGBA images are composited onto a white background.
- How the download file includes metadata alongside the caption.

Include code snippets from each module.

---

## 20. Test Cases and Results

| Test ID | Description                      | Input                  | Expected Output     | Status |
|---------|----------------------------------|------------------------|---------------------|--------|
| TC-01   | Valid JPEG upload                | 100×100 JPEG           | Image accepted      | Pass   |
| TC-02   | Valid PNG upload                 | 100×100 PNG            | Image accepted      | Pass   |
| TC-03   | Unsupported format               | .gif file              | Error message       | Pass   |
| TC-04   | Corrupted image bytes            | Random bytes           | Error message       | Pass   |
| TC-05   | Oversized file                   | >10 MB file            | Error message       | Pass   |
| TC-06   | RGBA to RGB conversion           | RGBA PNG               | RGB image           | Pass   |
| TC-07   | Caption whitespace normalisation | "  a  dog  "           | "A dog."            | Pass   |
| TC-08   | Empty caption rejection          | Model returns ""       | Error result        | Pass   |
| TC-09   | Model load failure               | Network error mock     | Error result        | Pass   |
| TC-10   | Detailed mode prompt stripping   | Prompt-prefixed text   | Clean caption       | Pass   |

---

## 21. Evaluation Metrics

| Metric   | Description                                        | Typical Range |
|----------|----------------------------------------------------|---------------|
| BLEU-1   | Unigram precision between generated and reference   | 0.3–0.7       |
| BLEU-4   | 4-gram precision (penalises short outputs)          | 0.1–0.4       |
| ROUGE-L  | Longest common subsequence F1                       | 0.3–0.6       |

**Important:** These metrics measure lexical overlap, not factual accuracy.
Actual scores depend on the evaluation dataset and should only be reported
after running the pipeline on real images with official reference captions.

---

## 22. Limitations

1. BLIP Base may produce factually incorrect or hallucinated descriptions.
2. Detailed mode does not guarantee longer or more detailed output — BLIP
   Base is not an instruction-following model.
3. CPU inference is slow (2–10 seconds per image).
4. English-only; may not describe culturally specific scenes accurately.
5. No persistent history or user accounts.
6. Free deployment platforms have memory constraints (~1 GB).

---

## 23. Future Enhancements

1. Integrate a larger model (BLIP-2, LLaVA) for instruction-following.
2. Add multilingual caption support.
3. Implement batch image processing.
4. Add a caption editing and feedback interface.
5. Fine-tune on domain-specific datasets.
6. Expose a REST API for programmatic access.
7. Add accessibility features (screen-reader-friendly alt text).

---

## 24. Conclusion

VisionCaption AI demonstrates a complete, functional application of
Generative AI for image understanding.  By leveraging the pretrained BLIP
model, the system generates natural-language image descriptions without
requiring custom training, paid APIs, or specialised hardware.  The modular
architecture allows future model upgrades, and the evaluation pipeline
provides a framework for measuring caption quality.

---

## 25. References

1. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapped
   Language-Image Pre-training for Unified Vision-Language Understanding
   and Generation. *ICML 2022*. arXiv:2201.12086.

2. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and
   Tell: A Neural Image Caption Generator. *CVPR 2015*.

3. Xu, K., et al. (2015). Show, Attend and Tell: Neural Image Caption
   Generation with Visual Attention. *ICML 2015*.

4. Anderson, P., et al. (2018). Bottom-Up and Top-Down Attention for
   Image Captioning and Visual Question Answering. *CVPR 2018*.

5. Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation
   of Machine Translation. *ACL 2002*.

6. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of
   Summaries. *Text Summarization Branches Out, ACL Workshop*.

7. Streamlit Documentation. https://docs.streamlit.io

8. Hugging Face Transformers Documentation. https://huggingface.co/docs/transformers

---

*Note: Do not fabricate additional references.  Only cite papers and
resources you have actually read or used.*
