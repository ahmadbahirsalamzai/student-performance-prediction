# Student Performance Prediction

An end-to-end machine learning application that predicts a student's math exam score from their
demographic background, reading score, and writing score. The project is built as a modular
training pipeline rather than a single notebook: ingestion, transformation, and model training are
separate components with their own configuration, and the fitted preprocessor and model are
persisted so the Flask web app can serve predictions without retraining.

## Dataset

`data/stud.csv` — 1,000 student records, 8 columns, no missing values.

| Type | Columns |
| --- | --- |
| Target | `math_score` |
| Numerical | `reading_score`, `writing_score` |
| Categorical | `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course` |

Reading and writing scores correlate strongly with the math score (0.82 and 0.80) and almost
perfectly with each other (0.96), which explains why the linear models perform as well as the
ensembles here.

## Architecture

```
src/
  components/
    data_ingestion.py       reads the raw csv, writes train/test splits to artifacts/
    data_transformation.py  builds and fits the ColumnTransformer, saves preprocessor.pkl
    model_trainer.py        grid searches 7 regressors, saves the best as model.pkl
  pipeline/
    train_pipeline.py       runs ingestion -> transformation -> training
    predict_pipeline.py     loads the saved artifacts and scores new input
  exception.py              custom exception carrying file name and line number
  logger.py                 timestamped run logs written to logs/
  utils.py                  serialization helpers and the grid search loop
app.py                      Flask app with two routes: landing page and prediction form
templates/                  index.html and home.html
static/style.css            stylesheet for both pages
```

Each component owns a `@dataclass` config holding its output paths, so artifact locations are
declared in one place instead of being scattered as string literals through the code.

**Preprocessing.** Numerical columns are median imputed and standardized; categorical columns are
mode imputed, one hot encoded, and scaled. Both branches run inside a single `ColumnTransformer`,
which is fitted on the training split only and then saved, so the web app applies exactly the same
transformations it was trained with. `handle_unknown="ignore"` means a category the model never saw
does not crash a prediction.

**Model selection.** Seven regressors are grid searched with 3-fold cross validation, refit on the
training set with the selected parameters, and ranked by R² on the held-out test set. The best one
is serialized.

## Results

Test set, 200 rows:

| Model | Test R² |
| --- | --- |
| **Linear Regression** | **0.8804** |
| Gradient Boosting | 0.8756 |
| CatBoost | 0.8593 |
| Random Forest | 0.8518 |
| XGBoost | 0.8492 |
| AdaBoost | 0.8475 |
| Decision Tree | 0.7277 |

Linear Regression wins. That is not the result you expect from a model bake-off, but it is the
right one here: with two numerical features that are near-linearly related to the target, the
gradient boosted models have nothing extra to find and mostly add variance. The single decision
tree, with no ensembling to average it out, overfits and lands eight points lower. The pipeline
still runs the full comparison rather than assuming a winner.

## Running it

```bash
pip install -r requirements.txt

python -m src.pipeline.train_pipeline   # trains and writes artifacts/
python app.py                           # serves the form on http://localhost:8080
```

`artifacts/` and `logs/` are not committed; run the training pipeline once before starting the app.

## Built with

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, Flask, joblib
