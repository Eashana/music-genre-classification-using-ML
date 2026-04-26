# Music Genre Classification using Machine Learning

Classify songs into musical genres using traditional machine learning models trained on audio features extracted from the **GTZAN** dataset.

This project is implemented as a **Jupyter Notebook**.

## Dataset

**GTZAN Genre Collection** is a widely used benchmark dataset for music genre classification.

- 10 genres (commonly: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- 100 tracks per genre
- 30-second audio clips

> Note: The dataset is often distributed via academic mirrors / Kaggle. Please download it separately and update the dataset path in the notebook as needed.

## Models used

The notebook trains and compares multiple classical ML classifiers:

- Naive Bayes
- Stochastic Gradient Descent (SGD)
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)

## Typical approach

A standard pipeline for this notebook is:

1. Load GTZAN audio files
2. Extract audio features (commonly using `librosa`), such as:
   - MFCCs
   - Chroma
   - Spectral centroid / bandwidth / rolloff
   - Zero-crossing rate
3. Prepare a feature matrix `X` and labels `y`
4. Split into train/test sets
5. Train and evaluate each model
6. Compare model performance (accuracy and/or classification report, confusion matrix)

## Repository contents

- `music-genre-classification-using-ml.ipynb` — end-to-end notebook (data prep → training → evaluation)

## Getting started

### Prerequisites

- Python 3.8+
- Jupyter Notebook (or Google Colab)

### Install dependencies

Exact packages depend on your environment and notebook imports, but the following are typically required:

```bash
pip install numpy pandas matplotlib scikit-learn librosa jupyter
```

### Run locally

```bash
jupyter notebook
```

Open `music-genre-classification-using-ml.ipynb` and run cells from top to bottom.

## Results

Add your final metrics here (recommended):

- Best model: **TBD**
- Test accuracy: **TBD**

If you share the best accuracy (and which model achieved it), I can update this section with your real results.

## License

This notebook is released under the **Apache License 2.0**.

If you want GitHub to detect it automatically, add a `LICENSE` file containing the full Apache-2.0 license text.
