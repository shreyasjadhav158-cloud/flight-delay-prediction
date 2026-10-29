# ✈️ Flight Delay Prediction Using Machine Learning

Mini project for **Machine Learning** (TY Div A, Semester I, 2026-27)
**JSPM's Rajarshi Shahu College of Engineering, Tathawade, Pune**

**Author:** Shreyas Jadhav · Roll No: CS3154

A comparative study of **Logistic Regression, K-Nearest Neighbors, Support Vector Machine and Random Forest** that predicts, *before take-off*, whether a flight will arrive more than 15 minutes late. Data: the [nycflights13](https://pypi.org/project/nycflights13/) dataset (336,776 flights from EWR, JFK and LGA in 2013, plus hourly weather).

## Highlights

- Only information known **before departure** is used (schedule, airline, route, distance, weather). `dep_delay`, `dep_time`, `arr_time` and `air_time` are excluded to avoid data leakage.
- Pre-processing (median imputation, scaling, one-hot encoding) lives inside a scikit-learn `Pipeline`, so it is learned only from training data.
- Imbalanced target (~23.7% delayed), so models are judged with F1, recall and ROC-AUC, not accuracy alone.
- 5-fold stratified cross-validation and `GridSearchCV` tuning (including `class_weight="balanced"`).

## Results (from the project report)

Tuned models on 2,400 unseen test flights:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.670 | 0.384 | 0.645 | 0.481 | 0.726 |
| KNN | 0.762 | 0.495 | 0.271 | 0.350 | 0.672 |
| SVM | 0.728 | 0.445 | 0.596 | 0.510 | 0.739 |
| **Random Forest** | 0.750 | 0.477 | 0.555 | **0.513** | **0.747** |

Random Forest is best, SVM is practically tied, and the scheduled hour plus weather (pressure, humidity, dew point, temperature) are the most important features. See the full write-up in [`docs/`](docs/Flight_Delay_ML_Mini_Project_Report_Shreyas_Jadhav.docx).

> Exact numbers can differ slightly from the report depending on your scikit-learn version.

## Project structure

```
.
├── flight_delay_ml.py   # complete pipeline: data → EDA → models → tuning → plots
├── requirements.txt
├── data/                # sampled dataset is cached here on first run
├── outputs/             # figures (fig2–fig9) and result tables (CSV)
├── docs/                # project report (.docx)
├── LICENSE
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/flight-delay-prediction.git
cd flight-delay-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python flight_delay_ml.py
```

The first run downloads nothing: `nycflights13` ships the data inside the pip package. It builds a stratified 12,000-flight sample (`random_state=42`) and caches it in `data/flights_sample.csv`. Tuning the SVM and Random Forest can take a few minutes.

### Options

| Flag | Description |
| --- | --- |
| `--sample-size N` | Number of flights to sample (default 12000) |
| `--out DIR` | Output folder for figures and tables (default `outputs`) |
| `--data PATH` | Cache path of the sampled CSV |
| `--synthetic` | Use fake data with the same schema, only to check the code runs quickly |

## Outputs

Saved to `outputs/`: class distribution, delay rate by hour/month/airline/airport, correlation heatmap, default-vs-tuned comparison, confusion matrices, ROC curves, Random Forest feature importance, and CSV versions of Tables 5–7 from the report.

## Limitations & future work

- Uses a 12,000-flight sample from a single year and one test split, so small differences between top models are not statistically strong.
- Key delay causes (late incoming aircraft, air traffic control, crew) are not in the data.
- Next steps: full dataset with Gradient Boosting/XGBoost, previous-flight delay features, threshold tuning, time-based split, and a small web app for delay risk.

## License

MIT, see [LICENSE](LICENSE).
