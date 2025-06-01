# TTS-GAN for Structural Health Monitoring Data Augmentation (ttsgan-shm-augmentation)

This repository contains the code and dataset for our research:

📝 Structural Dynamic Response Synthesis: A Transformer-Based Time-Series GAN Approach  
🔗 Available online in Results in Engineering  
DOI: https://doi.org/10.1016/j.rineng.2025.105549

---

## 📌 Project Overview

This project presents a novel Transformer-based Time-Series Generative Adversarial Network (TTS-GAN) designed to synthesize realistic structural dynamic responses. Unlike conventional GANs, the model integrates:

- A transformer architecture with an attention mechanism
- Composite loss function combining time-domain and frequency-domain alignment
- Capability to generate multi-channel acceleration signals directly from noise vectors

The approach is validated on a benchmark three-story aluminum structure, showing that TTS-GAN:

✅ Accurately replicates amplitude distribution and natural frequencies  
✅ Preserves structural behavior under various excitation types  
✅ Outperforms baseline models like Time-Series GAN (TSGAN) in the frequency domain

This research addresses data scarcity in structural health monitoring (SHM) and enables high-fidelity data augmentation for training damage detection models.

---

## 🗂️ Project Structure

.
├── code/
│ ├── adamw.py
│ ├── cfg.py
│ ├── dataLoader.py
│ ├── functions.py
│ ├── GANModels.py
│ ├── train_GAN.py
│ ├── Three_STORY_GAN_Train.py # Main training script
│ └── LoadSyntheticSignals.py # Generates synthetic responses
│
├── dataset/
│ ├── original_data/
│ ├── augmented_data/
│ └── final_dataset/
│
└── generated_data/
  └── for_final_dataset/

yaml
Copy
Edit

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch >= 1.11
- numpy
- matplotlib
- scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
🚀 How to Run
Train the GAN model:

bash
Copy
Edit
python code/Three_STORY_GAN_Train.py
Generate synthetic acceleration signals:

bash
Copy
Edit
python code/LoadSyntheticSignals.py
Make sure dataset paths are correctly configured in code/cfg.py.

📊 Dataset Description
original_data/: Raw structural acceleration data

augmented_data/: Data augmented with domain-specific techniques

final_dataset/: Preprocessed dataset used for training TTS-GAN

generated_data/: Synthetic signals generated using trained GAN models

If the datasets are too large for GitHub, external links (e.g., Google Drive) can be provided here.

📄 License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

🙏 Acknowledgment
Part of this project is adapted from the original TTS-GAN repository:
https://github.com/imics-lab/tts-gan
We thank the original authors for making their code publicly available.

📚 Citation
If you use this code or dataset in your research, please cite:

@article{your2025tts,
title = {Structural Dynamic Response Synthesis: A Transformer-Based Time-Series GAN Approach},
author = {Your Name and Co-authors},
journal = {Results in Engineering},
year = {2025},
doi = {10.1016/j.rineng.2025.105549}
}



