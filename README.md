
---

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

The model is trained on a benchmark structural system and validated for its ability to reproduce natural frequencies, amplitude distributions, and time-frequency characteristics. This method addresses data scarcity in structural health monitoring (SHM) and enables high-fidelity data augmentation for training damage detection models.

---

## 🧪 Experimental Model

The evaluation of the proposed TTS-GAN was conducted using an experimental three-story building model provided by Los Alamos National Laboratory. This scaled aluminum structure features:

- Three floors (17.7 cm height each), 30.5 cm × 30.5 cm footprint  
- Aluminum columns (2.8 cm × 0.6 cm) and plates (30.5 cm × 2.5 cm)  
- Central suspended column (15 cm × 2.5 cm × 2.5 cm)  
- Movement constrained to the x-direction  
- Excitation via electromagnetic shaker  
- Instrumentation: 4 accelerometers (one per floor center) and 1 force transducer at the base  
- Dataset Description:

   This system provides high-resolution dynamic response data used for model training and validation.
   Raw structural acceleration signals (original_data) obtained from the benchmark dataset by Sandia National Laboratories:
  
   📝"Experimental Data for Structural Health Monitoring of a Three-Story Frame Structure" by James P. Lynch and Kerri L. Sundaresan (2004).
   DOI: https://doi.org/10.2172/961604

  *Please cite this dataset if you use the original_data folder in your own research.*

---

## 🎯 Data Augmentation Techniques

To address limited data availability, we implemented the following time-series augmentation methods:

- Jittering: Additive noise (2%, 5%, 10% of peak amplitude)
- Rotation: 180° mirroring along the time axis
- Scaling: Interpolation to increase data points by 20%
- Permutation: Reordering of final three segments in a 4-part split
- Integration: Sequential application of two transformations from above (6 unique combinations)

These augmentations were validated to ensure no significant distortion in time or frequency domains, allowing their use in GAN training.

---

## 📊 Dataset Scenarios

After preprocessing and augmentation, multiple datasets were created to evaluate training performance at varying data volumes. For this implementation, we used:

- Scenario C: Dataset size = [1000 × 1024 × 3]  
Each sample contains 1024 time steps and 3 acceleration components (story1, story2, story3).

---

## 🗂️ Project Structure

```text
.
├── code/
│   ├── adamw.py
│   ├── cfg.py
│   ├── dataLoader.py
│   ├── functions.py
│   ├── GANModels.py
│   ├── train_GAN.py
│   ├── Three_STORY_GAN_Train.py       # Main training script
│   └── LoadSyntheticSignals.py        # Generates synthetic responses
│
├── dataset/
│   ├── original_data/                 # Raw experimental data
│   └── final_dataset/                 # Preprocessed + Data Augmentation Techniques + resized data (scenario C)
│
├── generated_data/
│   └── scenario_C/                   # Training Dataset
│
├── README.md                         # Project overview and usage instructions
├── requirements.txt                  # Python dependencies
├── LICENSE.txt                       # Apache 2.0 license
└── NOTICE                            # Info about reused code
```

---

## ⚙️ Requirements

- Python 3.8+
- All other dependencies are listed in requirements.txt

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Train the GAN model:

```bash
python code/Three_STORY_GAN_Train.py
```

This script launches training with preconfigured parameters and internally calls train_GAN.py with arguments defined in code/cfg.py.

2. Generate synthetic acceleration signals:

```bash
python code/LoadSyntheticSignals.py
```

Make sure to update the following variables in LoadSyntheticSignals.py before running:

- Running_model_path: path to the trained generator checkpoint  
- adl_data_path: path to real acceleration input file (.mat)

---

## 📦 Synthetic Signal Output

LoadSyntheticSignals.py will generate:

- Denormalized synthetic signals as CSV: denormalized_data.csv  
- Optional visualizations of synthetic time-series: output.png  

---

## 📘 Configurations

You can modify training parameters by:

- Editing code/cfg.py  
- Or passing command-line arguments, e.g.:

```bash
python code/train_GAN.py --class_name 3_STORY --augment_times 5 --exp_name ttsgan_exp
```

Important parameters include:

- --g_lr: Generator learning rate  
- --latent_dim: Latent space dimension  
- --augment_times: Number of synthetic signals per real sample  
- --exp_name: Experiment folder name (for logs/checkpoints)  
- --loss: GAN loss type (hinge, wgan-gp)

---

## 🧩 Key Scripts Description

- Three_STORY_GAN_Train.py – Wrapper script that calls train_GAN.py with full hyperparameters  
- train_GAN.py – Core training script: loads data, builds model, optimizer, scheduler  
- LoadSyntheticSignals.py – Uses trained Generator to produce synthetic signals from real input  
- cfg.py – Argument parser for hyperparameters and paths  
- dataLoader.py – Loads and preprocesses structural time-series dataset  
- GANModels.py – Defines Transformer-based Generator and Discriminator  
- functions.py – Core training utilities, loss functions, weight loading/saving  

---

## 📄 License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

---

## 🙏 Acknowledgment

Part of this project is adapted from the original TTS-GAN repository:  
https://github.com/imics-lab/tts-gan  
We thank the original authors for making their code publicly available.

---

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{KHODAEI2025105549,
title = {Structural Dynamic Response Synthesis: A Transformer-Based Time-Series GAN Approach},
journal = {Results in Engineering},
pages = {105549},
year = {2025},
issn = {2590-1230},
doi = {https://doi.org/10.1016/j.rineng.2025.105549},
url = {https://www.sciencedirect.com/science/article/pii/S2590123025016196},
author = {Sayed Ali Khodaei and Maryam Bitaraf},
keywords = {Deep learning, Generative adversarial network (GAN), Data augmentation, Structural health monitoring (SHM), Synthetic dynamic Responses, TTS-GAN, TSGAN},
abstract = {This study introduces a novel Transformer-based Time-Series Generative Adversarial Network (TTS-GAN) for synthesizing structural dynamic responses, addressing the critical challenge of data scarcity in structural health monitoring (SHM). Unlike existing methods, TTS-GAN generates realistic multi-channel acceleration signals directly from random noise vectors, without requiring auxiliary time- or frequency-domain inputs. The model integrates a transformer architecture with an attention mechanism to capture complex temporal dependencies and employs a composite loss function that aligns generated outputs with real signals in both time and frequency domains. Validation on a three-story aluminum structure demonstrates that TTS-GAN accurately replicates key structural features, including amplitude distribution and natural frequencies. Comparative results confirm that TTS-GAN outperforms a baseline Time-Series GAN (TSGAN), particularly in frequency-domain fidelity. The proposed approach represents a novel and efficient data augmentation framework for SHM, enabling high-quality signal generation from limited measured data.}
}
```
