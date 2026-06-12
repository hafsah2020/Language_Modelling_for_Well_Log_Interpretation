# 🧠 Language Modelling for Well Log Interpretation

A hands-on companion repository for learning language modelling concepts, applied to **petrophysical well log interpretation**. Each module pairs a core NLP/ML concept with a practical well log use case — so you build intuition for both domains simultaneously.

---

## 🎯 What This Repo Is

This repo follows a language modelling curriculum and maps every concept to well log interpretation tasks. You'll work with real-world petrophysical data (GR, RHOB, NPHI, DT, RT, etc.) to ground abstract ML ideas in something tangible.

**Assumed background:** Basic Python, some familiarity with well logs or petrophysics.

---

## 📂 Repository Structure

```
.
├── data/
│   ├── raw/                  # Raw LAS files and formation tops
│   ├── processed/            # Tokenized, normalized, windowed datasets
│   └── README.md             # Data sources and description
│
├── notebooks/
│   ├── 01_probability_distributions/
│   ├── 02_ngrams/
│   ├── 03_ngram_vs_transformer/
│   ├── 04_dataset_preparation/
│   ├── 05_train_slm/          ← Final Challenge Lab
│   ├── 06_preprocessing/
│   ├── 07_character_word_tokenization/
│   ├── 08_subword_tokenization/
│   ├── 09_bpe_tokenizer/
│   ├── 10_embeddings/
│   ├── 11_slm_with_bpe/
│   ├── 12_signal_vs_noise/
│   ├── 13_single_layer_nn/
│   ├── 14_complex_data_separation/
│   ├── 15_mlp_design/
│   ├── 16_hyperparameter_tuning/
│   ├── 17_overfitting/
│   ├── 18_gradients/
│   ├── 19_keras_training/
│   ├── 20_attention_weights/
│   ├── 21_attention_mechanism/
│   ├── 22_masked_multihead_attention/
│   ├── 23_positional_embeddings/
│   └── 24_transformer_parameters/
│
├── src/
│   ├── tokenizers/
│   ├── models/
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## 📚 Curriculum & Well Log Parallels

### Part 1 — Foundations of Language Modelling

| # | Module | Well Log Application |
|---|--------|----------------------|
| 01 | **Create Your Own Probability Distribution** | **Well Log Uncertainty Quantification.** Assign probability distributions over lithofacies (sandstone, shale, limestone, etc.) given log readings (GR, RHOB, NPHI, RT). Covers: hand-coding a facies posterior, Monte Carlo sampling with `random.choices` to generate geological realisations, propagating classification uncertainty into porosity P10/P50/P90, context-shifting (low-GR sand vs. high-GR shale), variance decomposition across measurement/model/geological uncertainty sources, sequential depth-column interpretation, and temperature scaling as an analogue for data-quality confidence. |
| 02 | **Experiment with N-grams** | Treat discretised log sequences as "text". Build N-gram models to predict the next depth interval's facies from the previous N intervals. |
| 03 | **Compare N-gram and Transformer Language Models** | Compare N-gram facies prediction against a small transformer on the same well log sequence. Evaluate accuracy and context sensitivity. |
| 04 | **Prepare the Dataset for Training an SLM** | Curate a multi-well LAS dataset: align depths, handle missing curves, split into train/val/test by well. |
| 05 | **🏆 Train Your Own Small Language Model (Final Challenge Lab)** | Train an SLM end-to-end to predict continuous log curves (e.g., predict DT from GR + RHOB + NPHI). Evaluate on a blind well. |

---

### Part 2 — Tokenization

| # | Module | Well Log Application |
|---|--------|----------------------|
| 06 | **Preprocess Data** | Normalise log curves (depth alignment, despiking, null removal). Convert LAS files to structured arrays. |
| 07 | **Tokenize Texts into Characters and Words** | Discretise log values into depth tokens (characters = single samples; words = depth windows). Explore vocabulary size vs. resolution trade-offs. |
| 08 | **Tokenize Texts into Subword Tokens** | Apply BPE-style grouping to log sequences. Discover recurring log patterns (e.g., coarsening-upward cycles) as "subword" tokens. |
| 09 | **Implement a BPE Tokenizer** | Implement Byte-Pair Encoding from scratch on discretised log sequences. Visualise merged patterns and their geological meaning. |
| 10 | **Experiment with Embeddings** | Learn dense embeddings for log facies tokens. Visualise with UMAP/t-SNE — do geologically similar facies cluster together? |
| 11 | **Train an SLM with Your BPE Tokenizer** | Re-train the SLM from Module 05 using your custom BPE tokenizer. Compare performance against character-level tokenization. |

---

### Part 3 — Neural Network Fundamentals

| # | Module | Well Log Application |
|---|--------|----------------------|
| 12 | **Distinguish Between Signal and Noise** | Apply signal processing (moving average, Savitzky-Golay) to noisy resistivity logs. Distinguish real bed boundaries from tool noise. |
| 13 | **Make Predictions with a Single-Layer Neural Network** | Build a single-layer network to classify clean sand vs. shale from GR alone. Examine decision boundary. |
| 14 | **Separate More Complex Data** | Add RHOB + NPHI. Show why a linear model fails for carbonate vs. clastic separation. Motivate deeper networks. |
| 15 | **Design Your Own MLP** | Design a multi-layer perceptron for multi-class lithofacies prediction (shale, sand, carbonate, coal). Tune depth and width. |
| 16 | **Tune Hyperparameters** | Grid/random search over learning rate, batch size, and hidden units for the lithofacies MLP. Use a validation well. |
| 17 | **Mitigate Overfitting** | Apply dropout, L2 regularisation, and early stopping. Show overfitting curves on small well datasets and how to fix them. |
| 18 | **Gradients** | Visualise gradient flow through the lithofacies network. Explore vanishing gradients when stacking many layers on log data. |
| 19 | **Train Your Model with Keras** | Reimplement the MLP in Keras with callbacks (ModelCheckpoint, EarlyStopping, TensorBoard). Train on a multi-well dataset. |

---

### Part 4 — Transformers

| # | Module | Well Log Application |
|---|--------|----------------------|
| 20 | **Visualising Attention Weights** | Train a small transformer on log sequences. Visualise which depth intervals the model attends to when predicting a bed boundary. |
| 21 | **Implement the Attention Mechanism** | Implement scaled dot-product attention from scratch. Apply to a sequence of log depth windows and inspect query-key-value dynamics. |
| 22 | **Implement Masked Multi-Head Attention** | Apply causal masking for autoregressive log curve prediction (predict next sample without seeing the future). |
| 23 | **Positional Embeddings** | Implement sinusoidal and learned positional embeddings for depth sequences. Explore whether depth order matters for your prediction task. |
| 24 | **Trainable Parameters in the Transformer Model** | Count and analyse every trainable parameter in your log transformer. Relate model size to dataset size and risk of overfitting. |

---

## 🛠️ Getting Started

```bash
# Clone the repo
git clone https://github.com/hafsah2020/Language_Modelling_for_Well_Log_Interpretation.git
cd Language_Modelling_for_Well_Log_Interpretation

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

### Key Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow / keras
torch                  # optional, for transformer modules
lasio                  # reading LAS well log files
welly                  # well log data wrangling
umap-learn             # embedding visualisation
jupyter
```

---

## 📡 Data

Well log data used in this repo comes from publicly available sources:

- [Kansas Geological Survey](http://www.kgs.ku.edu/Petrova/Oil/index.html)
- [Equinor Volve Dataset](https://www.equinor.com/energy/volve-data-sharing)
- [FORCE 2020 ML Competition](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
- [SEG 2016 ML Challenge (Facies Classification)](https://github.com/seg/2016-ml-contest)

See `data/README.md` for download instructions.

---

## 🗺️ Learning Path

```
Probability → N-grams → Dataset prep → ★ Train SLM ★
                                              ↓
Tokenization (char → word → subword → BPE) → Embeddings
                                              ↓
Single neuron → MLP → Overfitting → Gradients → Keras
                                              ↓
              Attention → Multi-head → Transformer
```

---

## 🤝 Contributing

Corrections, better well log datasets, or alternative geological interpretations are welcome — open an issue or PR.

---

## 📄 Attribution & License

Course curriculum sourced from [Google DeepMind via Google Skillboost](https://www.skills.google/collections/deepmind). All original course materials, lab structures, and exercises remain the intellectual property of Google DeepMind.

Well log interpretation extensions, petrophysical adaptations, and original notebooks in this repository are original work by [Hafsah Anibaba].
