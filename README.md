# Language Detection Model

A machine learning model that classifies short audio clips as **Urdu**,
**English**, or **Mixed** (code-switched), using MFCC audio features and
multinomial logistic regression — implemented both from scratch (NumPy)
and with Scikit-learn for comparison.

## How It Works

1. **Feature Extraction** — each `.wav` file is loaded at a 16kHz sampling
   rate, and 13-dimensional MFCCs (Mel-Frequency Cepstral Coefficients) are
   extracted and averaged over time into a single feature vector per clip,
   with a bias term prepended.
2. **Labeling** — labels are inferred directly from filenames using a
   naming convention (e.g. `ur-1-123.wav` → Urdu, `en-...` → English,
   `mix-...` → Mixed).
3. **Model (from scratch)** — a `MultinomialLogisticRegression` class
   implements softmax activation, cross-entropy loss, and batch gradient
   descent entirely in NumPy, with a plotted training loss curve.
4. **Model (Scikit-learn)** — a second `LogisticRegression` model
   (multinomial, `lbfgs` solver) is trained on the same data as a baseline
   comparison.
5. **Evaluation** — both models are evaluated on a held-out test split
   using accuracy and a confusion matrix.
6. **Inference** — a `predict_single_file()` helper runs either trained
   model on a brand-new `.wav` file and returns the predicted language.

## Dataset

Audio files are expected in `.wav` format, following the naming convention:

```
<language_code>-<speaker>-<clip_id>.wav
```

Where `language_code` is one of:

| Code | Language |
|---|---|
| `ur` | Urdu |
| `en` | English |
| `mix` | Mixed (code-switched) |

The notebook was originally built to run in Google Colab, loading data
from a Google Drive folder — update the `DATASET_PATH` variable to point
to your own local or cloud dataset location.

## Results

On the held-out test split, the Scikit-learn multinomial logistic
regression model achieved:

- **Accuracy:** ~70.8%
- **Confusion Matrix:**

  |  | Predicted Urdu | Predicted English | Predicted Mixed |
  |---|---|---|---|
  | **Actual Urdu** | 36 | 2 | 6 |
  | **Actual English** | 2 | 16 | 2 |
  | **Actual Mixed** | 9 | 5 | 11 |

The from-scratch model trains via batch gradient descent over 1500 epochs
and its loss curve is plotted directly in the notebook.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `language_detection_model.ipynb` in Jupyter, JupyterLab, or Google Colab.
3. Update `DATASET_PATH` to point to your folder of labeled `.wav` files.
4. Run all cells — the notebook will train both models, plot the loss curve, and report accuracy/confusion matrices for each.

## Tech

- Python
- NumPy (custom multinomial logistic regression implementation)
- Librosa (MFCC audio feature extraction)
- Scikit-learn (baseline logistic regression, evaluation metrics)
- Matplotlib (loss curve visualization)


## Publication copy

Published 5 October 2026 at the owner's request. This is a sanitized source snapshot. Original local Git history and original files remain unchanged. Pictures, videos, binary archives, private/runtime data, dependency folders and credentials are excluded. Notebook outputs, attachments and incidental metadata are removed. Documents are text-only extracts. Media references and redacted configuration may need replacements before running. No claim of successful rerun, production readiness, sole authorship or independent validation is implied.

Existing GitHub work checked and sanitized. Any supplied attribution is retained. Runtime operation not verified here.
