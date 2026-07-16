# EcoCover: Forest Cover Type Prediction Dashboard

EcoCover is a portfolio-ready Streamlit application for a COMP5310 forest cover classification project. It presents an interactive predictor, class distribution analysis, data-cleaning impact, model comparison, and resume-ready project notes.

## Live Demo

https://ecocover-forest-prediction.streamlit.app

## Tech Stack

- Python
- Pandas / NumPy
- scikit-learn
- Streamlit
- Plotly

## Project Results

The reported model results are from `Stage2_Modelling_And_Evaluation.ipynb`:

| Model | Accuracy | Precision macro | Recall macro | F1 macro |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.617241 | 0.458425 | 0.662787 | 0.493775 |
| Decision Tree | 0.532069 | 0.412443 | 0.624582 | 0.434131 |
| Logistic Regression | 0.506897 | 0.408942 | 0.588029 | 0.420206 |

Best estimator:

- `RandomForestClassifier`
- `n_estimators=200`
- `max_depth=20`
- `min_samples_leaf=10`
- `max_features="sqrt"`
- `class_weight="balanced"`

## Data and Model Provenance

This repository bundles the cleaned forest cover dataset used by the COMP5310 project:

`data/DatasetC_Forest_Cover_Cleaned.csv`

Required columns:

- `Elevation`
- `avg_hillshade`
- `Soil_Type`
- `Area`
- `Forest_Cover`

On startup, the app trains a scikit-learn preprocessing and Random Forest pipeline from the bundled cleaned CSV. The model comparison table reports the submitted notebook experiment results.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository, for example `ecocover-forest-prediction`.
2. Put the contents of this folder at the repository root.
3. Go to Streamlit Community Cloud and create a new app.
4. Select the GitHub repository and set the main file path to `app.py`.
5. Deploy and add the generated `streamlit.app` URL to your resume.

## Resume Summary

**EcoCover: Forest Cover Type Prediction Dashboard**  
Built an interactive machine learning dashboard for a seven-class forest cover prediction task using environmental features including elevation, hillshade, soil type and wilderness area. Processed multi-source COMP5310 data, engineered interpretable features, compared Logistic Regression, Decision Tree and Random Forest models, and achieved 61.72% test accuracy and 66.28% macro recall with a tuned Random Forest model.
