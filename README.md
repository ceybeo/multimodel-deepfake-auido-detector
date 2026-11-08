# Multimodal Deepfake and Cloned Voice Detection Framework

An open-source deep learning framework designed to detect synthetic media, facial deepfakes, and cloned voices using audiovisual feature fusion.

## Overview
As generative AI models become increasingly accessible, digital deception and voice cloning pose major threats to public trust, journalists, and civil society. This project provides a transparent, audit-ready verification tool that evaluates both visual artifacts (facial manipulation) and acoustic inconsistencies (synthetic voice footprints).

## Key Features
- **Visual Artifact Analysis:** Frame-level feature extraction using lightweight CNN backbones (EfficientNet).
- **Acoustic Verification:** Spectrogram and frequency analysis for synthetic audio and cloned voice detection.
- **Multimodal Fusion:** Cross-attention mechanism combining audio and visual cues for robust classification.
- **Open & Reproducible:** Designed to be easily audited and deployed by small teams and research organizations.

## Tech Stack
- Python, PyTorch
- OpenCV, Librosa
- Hugging Face Transformers

## License
MIT License
