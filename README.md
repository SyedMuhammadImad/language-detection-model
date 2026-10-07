# MFCC language classification: scratch and scikit-learn

Completed experimental experiment with fresh local-recording evaluation. This is a small-data model comparison, not a dependable general-purpose language detector. Code was repaired in a publication copy; original project and local recordings are retained. Team recordings and shared project are not claimed as solely authored work.

## Run

Install `requirements.txt` in Python 3.12. Run `python -m pytest -q`, then:

```text
python experiment.py --data-dir /path/to/recordings --cache-dir /path/to/local-cache --output /path/to/results.json --epochs 30
```

For the classical comparison `--epochs` does not change the fixed 1,500-step scratch optimization; it controls neural training only. WAV files can be arranged in folders such as `speaker-a english`, `speaker-a urdu`, and `speaker-a mixed`. Recognized suffixes also include eng/en, ur, mix/ue, and ar/arabic. For other layouts, pass `--manifest manifest.csv` with `path,label,speaker` columns; paths must be relative to `--data-dir`. Labels and speaker IDs must be nonempty. Every speaker must have recordings in each class. At least three speakers are required.

No recordings, audio features, model binaries, or plots are committed. The JSON metrics and feature/dataset fingerprint are included; exact reproduction requires the same locally retained recordings. No new recordings are downloaded. Audio is resampled to mono 16 kHz, capped at the first 30 seconds, and converted to 13 MFCCs (512-sample FFT, 256-sample hop). WAV input must be 0.15–60 seconds, mono/stereo and under 32 MB. Silent/invalid files and conflicting duplicate recordings fail explicitly; exact duplicates are removed. Scaling is fitted on training data only.

## Fresh results: held-out speakers

The measured dataset has 268 recordings, three speakers, and English, Urdu and mixed speech. Each of the three speakers is used once as the held-out test speaker. Results pooled across those test predictions:

- scratch: 38.81% accuracy, 0.379 macro F1
- sklearn: 37.69% accuracy, 0.367 macro F1

The scratch model retains explicit batch gradient descent, stable softmax, cross-entropy and bias, with class decoding, train-only standardization and small L2 regularization. A finite-difference gradient test checks the implementation. Both non-test speakers train each fold; no held-out-speaker labels tune hyperparameters. The separate random 80/20 clip split in `metrics.json` allows the same speakers on both sides and is not evidence of performance on new speakers. Its higher accuracy demonstrates why the split matters.

Programmatic single-file inference keeps the training scaler and label decoding together: fit `audio.LanguageDetector` on 13 mean-MFCC features and labels, then call `predict_file(wav_path)`. The file undergoes the same feature extraction and scaling as training. Neither a recording nor fitted model is embedded in this repository.

These results show weak transfer across speakers. Three voices, recording conditions and possible shared utterances are too limited to support broad deployment claims; speaker separation does not by itself control channel or phrase confounding. Mixed speech is one whole-clip class, not word-level code-switch detection. Models use no large pretrained speech representation.

## Notebook and verification

The notebook follows the same setup → experiment → results pattern as the repaired project repositories. It reads the included numeric metrics and runs an implementation smoke check; retraining uses the command above with your local recordings. Outputs are stripped. `VERIFICATION.json` records executed tests and the full fresh training run. Tests cover gradient correctness, noncontiguous labels, finite probabilities, invalid inputs, speaker separation, real feature extraction and single-file preprocessing; neural checks also verify invariance to padded frames.
