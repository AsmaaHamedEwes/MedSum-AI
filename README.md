# MedSum-AI — Intelligent Clinical Report Summarization & Predictive Outcome Analytics

End-to-end pipeline for radiology-report summarization and outcome prediction across **two complementary datasets**.

## Datasets

| Role        | Dataset                                           | Language | Records | Task                                  |
|-------------|---------------------------------------------------|----------|---------|---------------------------------------|
| Primary     | Indiana University Chest X-Ray (IU-CXR)           | English  | 3,955   | Multi-label MeSH classification + binary Normal/Abnormal |
| Secondary   | CASIA-CXR (Metmer & Yang, *Neurocomputing* 2024)  | French   | 13,672  | 5-class condition classification      |

CASIA-CXR adds:

* **Language diversity** — first public French radiology-report dataset.
* **Balanced single-label** classes (Cardiomegaly, Mass, Pleural Effusion, Pneumonia, Pneumothorax).
* **Rich metadata** — patient age, gender, projection, image dimensions.
* **Cross-lingual benchmarking** for multilingual summarization models (mBART, CamemBERT).

## Project structure

```
MedSUMAI/
├── CASIA_CXR_data/           # raw CASIA-CXR (5 condition folders + Reports / Labels CSVs)
├── NLMCXR_reports/           # raw IU-CXR XML reports
├── NLMCXR_png/               # raw IU-CXR images
├── data/                     # cleaned / engineered CSVs (produced by pipeline)
│   ├── iu_cxr_reports_parsed.csv
│   ├── iu_cxr_cleaned.csv
│   ├── iu_cxr_features.csv
│   ├── casia_cxr_combined.csv
│   ├── casia_cxr_cleaned.csv
│   └── casia_cxr_features.csv
├── notebooks/                # 6 step-by-step Jupyter notebooks (dual-dataset)
│   ├── 01_EDA_and_Statistics.ipynb
│   ├── 02_Data_Cleansing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_NLP_Summarization.ipynb
│   ├── 05_Outcome_Prediction.ipynb
│   └── 06_Cross_Dataset_Comparison.ipynb    # NEW — IU-CXR vs CASIA-CXR
├── src/
│   ├── run_full_eda.py                # master pipeline (runs both datasets)
│   ├── casia_cxr_pipeline.py          # dedicated CASIA-CXR module
│   └── generate_comparison_report.py  # NEW — cross-dataset visual report
└── outputs/
    ├── eda_figures/                # IU-CXR figures
    ├── eda_figures_casia/          # CASIA-CXR figures
    ├── comparison_report/          # NEW — IU vs CASIA charts + observations
    │   ├── 01_dataset_size_coverage.png ... 09_summary_dashboard.png
    │   ├── REPORT.md               # narrative report with all charts
    │   ├── observations.txt        # plain-text take-aways
    │   └── comparison_summary.json # machine-readable summary
    ├── eda_statistics_report.txt
    ├── casia_eda_statistics_report.txt
    ├── prediction_results.json
    └── casia_prediction_results.json
```

## Cross-dataset comparison

Run the comparison generator after the per-dataset pipelines complete:

```bash
python src/generate_comparison_report.py
```

or work through `notebooks/06_Cross_Dataset_Comparison.ipynb` to re-train all
models, plot the side-by-side charts, and emit the narrative report
interactively.

## Running the pipeline

```bash
# Both datasets (default)
python src/run_full_eda.py

# IU-CXR only
python src/run_full_eda.py --dataset iu

# CASIA-CXR only
python src/run_full_eda.py --dataset casia
```

The CASIA-CXR module can also be run standalone:

```bash
python src/casia_cxr_pipeline.py
```

## Pipeline stages

1. **Data ingestion** — XML parse (IU-CXR) and CSV merge (CASIA-CXR).
2. **EDA & statistics** — missing-data analysis, demographics, report-length, word frequencies, term × condition co-occurrence.
3. **Data cleansing** — Rahm & Do (2000) taxonomy: drop empty/duplicate, parse age/gender, filter truncated, language-aware text normalisation (English `XXXX` placeholders / French accent preservation).
4. **Feature engineering** — 25 structured features per language (clinical sentiment, severity, entity counts, readability, comorbidity scores, composites).
5. **Modeling** — Logistic Regression, Random Forest, XGBoost, Cox PHM (IU-CXR), plus TF-IDF text classifier (CASIA-CXR).
6. **Summarization** — BERT / BioBERT / ClinicalBERT / BART on IU-CXR; CamemBERT / mBART-50 / Lead-N baseline on CASIA-CXR.

## Reported results (5-fold / 10-fold CV)

| Dataset    | Task                      | Best Model       | Score (Macro-F1 / AUC)         |
|------------|---------------------------|------------------|--------------------------------|
| IU-CXR     | Normal vs Abnormal        | XGBoost          | AUC ≈ 0.976, F1 ≈ 0.961        |
| IU-CXR     | Cox survival (8 features) | Cox PHM          | C-index ≈ 0.737                |
| CASIA-CXR  | 5-class condition         | Random Forest    | Macro-F1 ≈ 0.999               |
| CASIA-CXR  | 5-class condition (text)  | LR + TF-IDF      | Macro-F1 ≈ 0.999               |

## Citation

```bibtex
@article{CASIA-CXR,
  author  = {Hichem Metmer and Xiaoshan Yang},
  title   = {An open chest X-ray dataset with benchmarks for automatic radiology report generation in French},
  journal = {Neurocomputing}, volume = {609}, pages = {128478}, year = {2024},
  doi     = {10.1016/j.neucom.2024.128478}
}
```

Author: **Asmaa Hamed** — May 2026
