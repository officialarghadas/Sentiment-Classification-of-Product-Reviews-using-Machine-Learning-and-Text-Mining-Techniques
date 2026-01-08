# Sentiment-Classification-of-Product-Reviews-using-Machine-Learning-and-Text-Mining-Techniques

dataset link: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews?fbclid=IwY2xjawPM9FFicmlkETFPMUZoQjA1NVRyRkpBbm1vc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHu1qhixcC7Js4KS79d8A8Lq1vYcxh3U_0UTqj7LYzI2PLLEZhQJMiGEfroB5&brid=sx5T0aMqDdI3mF4Jopxvlw

Please download the dataset first, and keep it in the same folder. Then follow the instructions below:
1. Run the Pre-Processing file first.
2. Run the feature engineering file.
3. Run the model Evaluation file.

If you get any error, kindly check whether the datasets are being created in the designated folder or not.

For any further info, kindly contact: https://www.linkedin.com/in/argha-das-08899223b/


# project_code_full.ipynb — End-to-end Notebook (README section)

This section documents the purpose, inputs, outputs, usage and reproducibility notes for `project_code_full.ipynb`.

Link to notebook: [project_code_full.ipynb](https://github.com/officialarghadas/Sentiment-Classification-of-Product-Reviews-using-Machine-Learning-and-Text-Mining-Techniques/blob/main/project_code_full.ipynb)

---

## Summary

`project_code_full.ipynb` is a single-file, end-to-end pipeline that demonstrates binary sentiment classification for product reviews. The notebook:

- Loads raw labeled review data (fastText-style lines),
- Cleans text and creates train/validation splits,
- Extracts TF-IDF + simple lexical features,
- Trains multiple classical ML classifiers,
- Evaluates models on a hold-out validation set,
- Runs a short inference demo on example sentences.

It is designed as a consolidated demonstration of preprocessing, feature engineering, training, evaluation, and inference.

---

## Expected input

Place a file named `test.ft.txt` in the same directory as the notebook. The notebook expects each non-empty line to be in the format:

```
__label__<label> <review text>
```

Example lines:

```
__label__2 This product is amazing! I love it.
__label__1 Terrible quality, I regret buying this.
```

The notebook extracts the label using the regex `__label__([0-9]+)`.

---

## Outputs produced by the notebook

When executed, the notebook creates the following files/folders:

- `splits/train.csv` — training split (80%).
- `splits/validation.csv` — validation split (20%).
- `features/tfidf_lex_combined_train.npz` — sparse combined training features (TF-IDF + lexical).
- `features/tfidf_lex_combined_val.npz` — sparse combined validation features.
- `features/train_labels.npy` — training labels (NumPy array).
- `features/val_labels.npy` — validation labels (NumPy array).
- `features/tfidf_vectorizer.pkl` — pickled TF-IDF vectorizer object.

Note: The notebook trains models in memory and currently does not persist trained model artifacts. If you wish to save models, add `pickle.dump()` calls after training.

---

## Dependencies

The notebook imports and uses the following packages (direct imports visible in notebook):

- pandas
- numpy
- scipy (sparse matrices)
- scikit-learn (feature extraction, model selection, classifiers, metrics)
- pickle (stdlib)
- pathlib, re, json (stdlib)
- jupyter / ipykernel

Suggested minimal requirements to add to `requirements.txt` or `environment.yml`:

```text
python>=3.8
pandas>=1.3
numpy>=1.21
scipy>=1.7
scikit-learn>=1.0
jupyterlab or notebook
```

Adjust and pin versions after testing in your target environment.

---

## Main processing steps (high level)

1. Import libraries and define helper functions.
2. Preprocessing:
   - Read `test.ft.txt`.
   - Parse label and review text per line.
   - `clean_text()` normalizes text: lowercases, removes URLs/HTML, strips non-alphanumeric characters (keeps apostrophes), collapses whitespace.
   - Adds `text_clean`, `length`, `tokens`.
   - Performs an 80/20 stratified train/validation split (random_state=42).
   - Saves splits under `splits/`.
3. Feature engineering:
   - Lexical features: `char_count`, `word_count`, `avg_word_length`, `exclamation_count`, `question_count`.
   - TF-IDF: `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, stop_words='english')`.
   - Concatenates TF-IDF (sparse) with lexical features and saves `.npz` files and vectorizer `.pkl`.
4. Model training & evaluation:
   - Models trained: Decision Tree, Logistic Regression, Linear SVM (LinearSVC), AdaBoost.
   - Each model has an optional GridSearchCV block (tunable via `RUN_*_TUNING` flags). Default is training the default estimator (no grid search).
   - Evaluation uses accuracy and `classification_report` (precision/recall/f1) on the validation set.
5. Quick prediction demo:
   - Demonstrates inference on three example texts and prints predicted label and a simple Positive/Negative mapping (`label == 2 -> Positive`, otherwise Negative).

---

## Example metrics (from a particular notebook run)

The notebook prints evaluation metrics for each model. In the captured run the following approximate accuracies were reported:

- Decision Tree Accuracy ≈ 0.7706
- Logistic Regression Accuracy ≈ 0.8960 (note: convergence warning suggests increasing `max_iter`)
- Linear SVM Accuracy ≈ 0.8640
- AdaBoost Accuracy ≈ 0.7279

These values depend on the dataset, preprocessing, TF-IDF vocabulary, and the train/validation split.

---

## How to run

Interactively in Jupyter / JupyterLab:

1. Ensure dependencies are installed.
2. Place `test.ft.txt` in the notebook directory.
3. Open `project_code_full.ipynb` and run cells top-to-bottom.

Headless execution (non-interactive):

```bash
# Execute the notebook (adjust timeout as needed)
jupyter nbconvert --to notebook --execute project_code_full.ipynb --ExecutePreprocessor.timeout=1200
```

After running, check `splits/` and `features/` for artifacts and view the printed evaluation output in notebook logs.

---

## Configurable parameters inside the notebook

- `RUN_DT_TUNING`, `RUN_LR_TUNING`, `RUN_SVM_TUNING`, `RUN_ADA_TUNING` — set to `True` to enable GridSearchCV hyperparameter tuning for the respective model (slower).
- TF-IDF params (ngram range, min_df, max_df, stop_words) — edit inline to experiment.
- Classifier hyperparameter grids — defined inline when tuning is enabled.
- Random seeds: `random_state=42` is used for splitting and for models that support it.

---

## Recommended README additions & reproducibility improvements

To help other users reproduce and use the notebook, add or implement the following:

- requirements.txt or environment.yml with pinned versions.
- A small sample `test.ft.txt` (e.g., 10–50 lines) to enable quick testing without downloading the full Kaggle dataset.
- Precise expected filename(s) and folder layout in README (the notebook expects `test.ft.txt`).
- Save trained model artifacts (e.g., `models/logistic_regression.pkl`) and add a short `infer.py` script showing how to load the vectorizer and model to make predictions.
- Consider splitting notebook logic into scripts (`src/preprocess.py`, `src/features.py`, `src/train.py`) for CI and reproducibility.
- Add unit tests or a minimal CI workflow that runs the pipeline on the small sample dataset.

---

## Notes & potential gotchas

- The notebook assumes label tokens like `__label__1` and `__label__2`. If using a different dataset format, update the parsing logic.
- Large TF-IDF vocabularies can create large sparse matrices; consider `max_features` or dimensionality reduction if running into memory issues.
- Convergence warnings for LogisticRegression and LinearSVC may appear — increase `max_iter` or switch solver/regularization as needed.
- The notebook demonstrates a binary mapping (`2 -> Positive`, `1 -> Negative`) — if your data contains more classes, update evaluation & mapping logic.
- Confirm the license/usage terms of the Kaggle dataset before redistributing any derived data or models.

---

## Quick sample `test.ft.txt` (mini example for testing)

You can create a tiny `test.ft.txt` locally to test the pipeline:

```
__label__2 I absolutely love this product. Works perfectly!
__label__1 Very disappointed — stopped working after a week.
__label__2 Fantastic value for money, highly recommend.
__label__1 Poor build quality and bad customer service.
```

Place the file in the same directory as the notebook and run the notebook to verify it executes and produces `splits/` and `features/`.

---

