"""
MedSum-AI: CASIA-CXR Secondary Dataset Pipeline
================================================
CASIA-CXR (French) Chest X-Ray Dataset — Loading, EDA, Cleansing,
Feature Engineering, and Modeling.

Author: Asmaa Hamed
Date: May 2026

The CASIA-CXR dataset (Metmer & Yang, 2024 — Neurocomputing) contains
high-resolution chest radiographs paired with French-language narrative
radiology reports, across 5 pathology classes:
    * Cardiomegaly       (5,503)
    * Pneumonia          (2,139)
    * Mass               (2,030)
    * Pneumothorax       (2,000)
    * PleuralEffusion    (2,000)
    -----------------------------
    Total                13,672 reports

This pipeline mirrors the IU-CXR (English) primary-dataset pipeline so
that downstream summarisation and outcome-prediction stages can be
trained / evaluated on both datasets.
"""

import os
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, 'CASIA_CXR_data')
DATA_DIR    = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
FIG_DIR     = os.path.join(OUTPUT_DIR, 'eda_figures_casia')
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

CONDITIONS = ['Cardiomegaly', 'Mass', 'PleuralEffusion', 'Pneumonia', 'Pneumothorax']


# =============================================================================
# STEP 1: LOAD & MERGE CASIA-CXR REPORTS
# =============================================================================
def load_casia_reports(save: bool = True) -> pd.DataFrame:
    """Load all 5 condition-specific Reports.csv files into a single DataFrame.

    Adds a `condition` column and a `dataset_source` tag.
    """
    print("\n" + "=" * 60)
    print("STEP 1: LOADING CASIA-CXR REPORTS")
    print("=" * 60)

    dfs = []
    for cond in CONDITIONS:
        rpt_path = os.path.join(RAW_DIR, f'CASIA-CXR_{cond}',
                                f'CASIA-CXR_{cond}_Reports.csv')
        if not os.path.exists(rpt_path):
            print(f"  WARNING: missing {rpt_path}")
            continue
        d = pd.read_csv(rpt_path, encoding='utf-8-sig')
        # Drop trailing placeholder cols ('', 'EndHere', '#') if present
        d = d.loc[:, ~d.columns.str.match(r'^(Unnamed.*|EndHere|)$')]
        d['condition'] = cond
        dfs.append(d)
        print(f"  {cond:<18} {len(d):>5,} records, {len(d.columns):>2} cols")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['dataset_source'] = 'casia_cxr'
    print(f"\n  Combined : {len(df):,} reports across {len(CONDITIONS)} conditions")

    if save:
        out = os.path.join(DATA_DIR, 'casia_cxr_combined.csv')
        df.to_csv(out, index=False)
        print(f"  -> {out}")
    return df


# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# =============================================================================
def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")


def run_eda(df: pd.DataFrame) -> dict:
    """Heavy EDA on the merged CASIA-CXR dataset. Saves figures + stats."""
    print("\n" + "=" * 60)
    print("STEP 2: CASIA-CXR EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('Set2')

    stats: dict = {'n_records': int(len(df)), 'n_columns': int(df.shape[1])}

    # --- Missing-data ----------------------------------------------------------
    miss_pct = (df.isnull().sum() / len(df) * 100).round(2)
    stats['missing_pct'] = miss_pct.to_dict()
    print("  Missing values (top-10):")
    for c, p in miss_pct.sort_values(ascending=False).head(10).items():
        print(f"    {c:<22} {p:>6.2f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    top = miss_pct.sort_values(ascending=True).tail(15)
    colors = ['#e74c3c' if p > 10 else '#f39c12' if p > 5 else '#27ae60' for p in top.values]
    ax.barh(top.index, top.values, color=colors, edgecolor='white')
    ax.set_xlabel('Percentage Missing (%)')
    ax.set_title('CASIA-CXR — Missing Data per Column', fontweight='bold')
    for i, p in enumerate(top.values):
        ax.text(p + 0.3, i, f'{p:.1f}%', va='center', fontsize=9)
    _save(fig, 'casia_fig01_missing_data.png')

    # --- Condition distribution ------------------------------------------------
    cond_counts = df['condition'].value_counts()
    stats['condition_counts'] = cond_counts.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cond_counts.plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2'),
                     edgecolor='white')
    axes[0].set_title('CASIA-CXR — Records per Condition', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=30)
    for i, v in enumerate(cond_counts.values):
        axes[0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    axes[1].pie(cond_counts.values, labels=cond_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('Set2'), startangle=90,
                wedgeprops={'edgecolor': 'white'})
    axes[1].set_title('CASIA-CXR — Class Proportion', fontweight='bold')
    _save(fig, 'casia_fig02_condition_distribution.png')

    # --- Demographics (age / gender) -------------------------------------------
    df['age_num'] = df['PatientAge'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    valid_age = df['age_num'].dropna()
    stats['age_summary'] = {
        'mean': float(valid_age.mean()),
        'median': float(valid_age.median()),
        'min': float(valid_age.min()) if len(valid_age) else None,
        'max': float(valid_age.max()) if len(valid_age) else None,
        'n_valid': int(len(valid_age))
    }
    print(f"\n  Age — n={len(valid_age):,}, mean={valid_age.mean():.1f}, "
          f"median={valid_age.median():.1f}")

    df['gender_clean'] = df['PatientGender'].astype(str).str.upper().str[0]
    df.loc[~df['gender_clean'].isin(['M', 'F']), 'gender_clean'] = np.nan
    g_counts = df['gender_clean'].value_counts()
    stats['gender_counts'] = g_counts.to_dict()
    print(f"  Gender — {g_counts.to_dict()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(valid_age, bins=40, color='#3498db', edgecolor='white')
    axes[0].axvline(valid_age.mean(), color='red', linestyle='--', label=f'Mean={valid_age.mean():.1f}')
    axes[0].set_title('Age Distribution (CASIA-CXR)', fontweight='bold')
    axes[0].set_xlabel('Age (years)'); axes[0].legend()

    axes[1].bar(g_counts.index, g_counts.values, color=['#3498db', '#e91e63'],
                edgecolor='white')
    axes[1].set_title('Gender Distribution', fontweight='bold')
    for i, v in enumerate(g_counts.values):
        axes[1].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    sns.violinplot(data=df.dropna(subset=['age_num']),
                   x='condition', y='age_num', hue='gender_clean',
                   split=True, ax=axes[2])
    axes[2].set_title('Age by Condition & Gender', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=30)
    _save(fig, 'casia_fig03_demographics.png')

    # --- Report length ---------------------------------------------------------
    df['findings_wc']   = df['Findings'].fillna('').apply(lambda x: len(str(x).split()))
    df['impression_wc'] = df['Impression'].fillna('').apply(lambda x: len(str(x).split()))
    df['indication_wc'] = df['Indication'].fillna('').apply(lambda x: len(str(x).split()))
    stats['length'] = {
        'findings_mean':   float(df['findings_wc'][df['findings_wc'] > 0].mean()),
        'findings_median': float(df['findings_wc'][df['findings_wc'] > 0].median()),
        'impression_mean': float(df['impression_wc'][df['impression_wc'] > 0].mean()),
        'impression_median': float(df['impression_wc'][df['impression_wc'] > 0].median())
    }
    print(f"\n  Findings  WC — mean={stats['length']['findings_mean']:.1f}, "
          f"median={stats['length']['findings_median']:.1f}")
    print(f"  Impression WC — mean={stats['length']['impression_mean']:.1f}, "
          f"median={stats['length']['impression_median']:.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].hist(df['findings_wc'][df['findings_wc'] > 0], bins=50,
                    color='#3498db', edgecolor='white')
    axes[0, 0].set_title('Findings Word Count (CASIA-CXR)', fontweight='bold')
    axes[0, 1].hist(df['impression_wc'][df['impression_wc'] > 0], bins=50,
                    color='#e67e22', edgecolor='white')
    axes[0, 1].set_title('Impression Word Count', fontweight='bold')
    sns.boxplot(data=df, x='condition', y='findings_wc', ax=axes[1, 0])
    axes[1, 0].set_title('Findings Length by Condition', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df, x='condition', y='impression_wc', ax=axes[1, 1])
    axes[1, 1].set_title('Impression Length by Condition', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=30)
    _save(fig, 'casia_fig04_report_length.png')

    # --- Most common French words ---------------------------------------------
    fr_stop = set("""le la les un une des de du au aux et ou ni mais
                     a à dans en sur sous par pour avec sans
                     ce cet cette ces son sa ses notre nos votre vos
                     je tu il elle on nous vous ils elles me te se
                     est sont etre être ont a avait avaient suis es
                     que qui quoi dont
                     plus moins très tres bien aussi non oui
                     d l n s y t""".split())

    def tokenise(text):
        text = re.sub(r'[^A-Za-zÀ-ÿ]+', ' ', str(text).lower())
        return [w for w in text.split() if len(w) > 2 and w not in fr_stop]

    all_words = []
    for t in df['Findings'].fillna(''):
        all_words.extend(tokenise(t))
    top_words = Counter(all_words).most_common(25)
    stats['top_25_findings_words'] = top_words

    fig, ax = plt.subplots(figsize=(10, 8))
    ws, cs = zip(*top_words[::-1])
    ax.barh(ws, cs, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(ws))),
            edgecolor='white')
    ax.set_title('Top-25 Most Frequent Words in CASIA-CXR Findings (French)',
                 fontweight='bold')
    ax.set_xlabel('Frequency')
    _save(fig, 'casia_fig05_top_words.png')

    # --- Co-occurrence matrix of conditions vs key terms ----------------------
    key_terms_fr = ['cardiomégalie', 'pneumonie', 'épanchement', 'pneumothorax',
                    'masse', 'opacité', 'condensation', 'nodule', 'consolidation',
                    'médiastinal', 'pleural', 'normal']
    coocc = pd.DataFrame(index=CONDITIONS, columns=key_terms_fr, dtype=int)
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        text_blob = (sub['Findings'].fillna('') + ' ' +
                     sub['Impression'].fillna('')).str.lower().str.cat(sep=' ')
        for term in key_terms_fr:
            coocc.loc[cond, term] = text_blob.count(term)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(coocc.astype(int), annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Frequency'}, ax=ax)
    ax.set_title('CASIA-CXR — Term × Condition Co-occurrence (French)',
                 fontweight='bold')
    _save(fig, 'casia_fig06_term_cooccurrence.png')
    stats['term_cooccurrence'] = coocc.to_dict()

    # --- Save text report -----------------------------------------------------
    rep_path = os.path.join(OUTPUT_DIR, 'casia_eda_statistics_report.txt')
    with open(rep_path, 'w', encoding='utf-8') as f:
        f.write("CASIA-CXR — EDA Statistics Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Records           : {stats['n_records']:,}\n")
        f.write(f"Columns           : {stats['n_columns']}\n\n")
        f.write("Condition counts:\n")
        for k, v in stats['condition_counts'].items():
            f.write(f"  {k:<18} {v:>5,}\n")
        f.write(f"\nAge — mean={stats['age_summary']['mean']:.1f}, "
                f"median={stats['age_summary']['median']:.1f}, "
                f"n_valid={stats['age_summary']['n_valid']:,}\n")
        f.write(f"Gender : {stats['gender_counts']}\n\n")
        f.write("Report length (words):\n")
        for k, v in stats['length'].items():
            f.write(f"  {k:<22} {v:.1f}\n")
        f.write("\nTop-25 most frequent findings words:\n")
        for w, c in top_words:
            f.write(f"  {w:<20} {c:>6,}\n")
    print(f"\n  Statistics report -> {rep_path}")

    return stats


# =============================================================================
# STEP 3: DATA CLEANSING
# =============================================================================
def clean_casia(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Rahm & Do (2000) cleansing taxonomy to CASIA-CXR."""
    print("\n" + "=" * 60)
    print("STEP 3: CASIA-CXR DATA CLEANSING")
    print("=" * 60)

    initial = len(df)
    log = [('Raw input', initial)]

    # 1. Drop rows where ExamID is null (these are placeholder lines)
    df = df.dropna(subset=['ExamID']).copy()
    log.append(('After dropping null ExamID', len(df)))
    print(f"  After dropping null ExamID : {len(df):,}")

    # 2. Fill text fields with empty string
    for col in ['Findings', 'Impression', 'Indication', 'Comparison']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # 3. Remove records with both Findings and Impression empty
    both_empty = (df['Findings'] == '') & (df['Impression'] == '')
    df = df[~both_empty].copy()
    log.append(('After removing empty text', len(df)))
    print(f"  After removing empty text  : {len(df):,}")

    # 4. Deduplicate exact ExamID
    df = df.drop_duplicates(subset='ExamID', keep='first')
    log.append(('After dedup by ExamID', len(df)))
    print(f"  After dedup by ExamID      : {len(df):,}")

    # 5. Token filter (>=5 tokens). NOTE: we do NOT deduplicate by full text
    #    because CASIA-CXR reports are heavily templated — different exams
    #    may legitimately share identical findings boilerplate.
    df['_combined'] = (df['Findings'] + ' ' + df['Impression']).str.strip()
    df['_tok'] = df['_combined'].apply(lambda x: len(x.split()))
    df = df[df['_tok'] >= 5].copy()
    log.append(('After truncation filter', len(df)))
    print(f"  After truncation filter    : {len(df):,}")

    # 6. Parse PatientAge ("057Y", "57", "57 ans" -> 57)
    def parse_age(v):
        if pd.isna(v): return np.nan
        m = re.match(r'\s*(\d+)', str(v))
        return float(m.group(1)) if m else np.nan
    df['patient_age'] = df['PatientAge'].apply(parse_age)

    # 7. Normalise gender; map junk to NaN
    df['patient_gender'] = (df['PatientGender'].astype(str)
                            .str.upper().str[0])
    df.loc[~df['patient_gender'].isin(['M', 'F']), 'patient_gender'] = np.nan

    # 8. Normalise French text (preserve accents)
    def normalise_fr(text):
        if not text or pd.isna(text): return ''
        t = str(text)
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'\.+', '.', t)
        t = re.sub(r"[’`']", "'", t)
        t = t.replace('/', '. ')
        return t.strip().lower()

    df['findings_clean']   = df['Findings'].apply(normalise_fr)
    df['impression_clean'] = df['Impression'].apply(normalise_fr)
    df['indication_clean'] = df['Indication'].apply(normalise_fr)

    # 9. Build a unified-schema view for downstream notebooks
    df = df.rename(columns={
        'ExamID':           'uid',
        'ImageID':          'image_id',
        'PatientID':        'patient_id',
        'ImageDir':         'image_dir',
        'StudyDate':        'study_date',
        'PatientPosition':  'patient_position',
        'PositionView':     'position_view',
        'Projection':       'projection',
        'ProjectionMethod': 'projection_method',
        'ImageWidth':       'image_width',
        'ImageHeight':      'image_height',
        'ReportID':         'report_id',
        'Findings':         'findings',
        'Indication':       'indication',
        'Comparison':       'comparison',
        'Impression':       'impression',
    })
    df['dataset_source'] = 'casia_cxr'

    cols_to_keep = ['uid', 'image_id', 'patient_id', 'image_dir',
                    'study_date', 'patient_age', 'patient_gender',
                    'patient_position', 'position_view',
                    'projection', 'projection_method',
                    'image_width', 'image_height', 'report_id',
                    'findings', 'indication', 'comparison', 'impression',
                    'findings_clean', 'impression_clean', 'indication_clean',
                    'condition', 'dataset_source']
    df = df[[c for c in cols_to_keep if c in df.columns]]

    out = os.path.join(DATA_DIR, 'casia_cxr_cleaned.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"  Final cleaned              : {len(df):,} "
          f"({100*len(df)/initial:.1f}% retained)\n  -> {out}")

    # Save cleaning log
    log_df = pd.DataFrame(log, columns=['Step', 'Records'])
    log_df['Removed'] = log_df['Records'].diff().fillna(0).abs().astype(int)
    log_df.to_csv(os.path.join(OUTPUT_DIR, 'casia_cleaning_log.csv'),
                  index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(log_df['Step'], log_df['Records'],
            color=plt.cm.Greens(np.linspace(0.3, 0.9, len(log_df))),
            edgecolor='white')
    ax.set_xlabel('Records remaining'); ax.invert_yaxis()
    ax.set_title('CASIA-CXR — Cleaning Pipeline', fontweight='bold')
    for i, c in enumerate(log_df['Records']):
        ax.text(c + 30, i, f'{c:,}', va='center', fontweight='bold')
    _save(fig, 'casia_fig07_cleaning_pipeline.png')

    return df


# =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer structured features mirroring IU-CXR's 25-feature set,
    adapted for French radiology reports.
    """
    print("\n" + "=" * 60)
    print("STEP 4: CASIA-CXR FEATURE ENGINEERING")
    print("=" * 60)

    df['text'] = df['findings_clean'].fillna('') + ' ' + df['impression_clean'].fillna('')

    # French clinical lexicons
    positive_fr = ['normal', 'absence', "pas d'", 'sans', 'aspect normal',
                   'inchangé', 'stable']
    negative_fr = ['opacité', 'condensation', 'épanchement', 'masse',
                   'nodule', 'cardiomégalie', 'œdème', 'pneumothorax',
                   'fracture', 'augmentation', 'élargissement', 'infiltrat',
                   'atteinte', 'pneumonie']

    entities_fr = ['poumon', 'cœur', 'cardiaque', 'médiastin', 'pleural',
                   'aorte', 'rachis', 'côte', 'diaphragm', 'hil', 'bronch',
                   'trachée', 'pulmonaire', 'thoracique', 'sternum',
                   'clavicule', 'péricard']

    severity_fr = {'discrète': 1, 'minime': 1, 'légère': 1, 'petite': 1,
                   'modérée': 2, 'moyenne': 2, 'partielle': 2,
                   'sévère': 3, 'important': 3, 'large': 3, 'marquée': 3,
                   'étendue': 3, 'massive': 4, 'severe': 3}

    diagnostic_fr = ['compatible avec', 'évocateur de', 'évoque',
                     'en faveur de', 'probable', 'suggérer',
                     'aspect de', 'pouvant correspondre']

    hedges_fr = ['possible', 'probable', 'peut', 'pourrait', 'semble',
                 'apparaît', 'douteux', 'questionnable']

    temporal_fr = ['antérieur', 'précédent', 'depuis', 'nouveau',
                   'ancien', 'chronique', 'aigu', 'évolutif', 'stable',
                   'inchangé']

    recommendations_fr = ['recommand', 'suggér', 'à recontrôler',
                          'corrélation clinique', 'compléter par',
                          'évaluation complémentaire', 'scanner']

    negations_fr = ["absence d'", 'absence de', 'pas de', 'pas d',
                    'sans', 'aucun', 'aucune']

    def count_terms(text, terms):
        t = str(text).lower()
        return sum(1 for x in terms if x in t)

    def sentiment(text):
        t = str(text).lower()
        p = count_terms(t, positive_fr)
        n = count_terms(t, negative_fr)
        s = p + n
        return n / s if s > 0 else 0.5

    df['F01_clinical_sentiment']      = df['text'].apply(sentiment)
    df['F02_clinical_entity_count']   = df['text'].apply(lambda t: count_terms(t, entities_fr))
    df['F03_severity_score']          = df['text'].apply(
        lambda t: max([v for k, v in severity_fr.items() if k in str(t).lower()] or [0]))
    df['F04_diagnostic_term_freq']    = df['text'].apply(lambda t: count_terms(t, diagnostic_fr))
    df['F05_impression_length']       = df['impression_clean'].fillna('').apply(lambda x: len(x.split()))

    df['F06_word_count']              = df['text'].apply(lambda x: len(str(x).split()))
    df['F07_sentence_count']          = df['text'].apply(
        lambda x: len([s for s in re.split(r'[.!?]', str(x)) if s.strip()]))
    fwc = df['findings_clean'].fillna('').apply(lambda x: len(x.split()))
    iwc = df['impression_clean'].fillna('').apply(lambda x: len(x.split()))
    df['F08_findings_impression_ratio'] = fwc / iwc.replace(0, 1)
    df['F09_has_measurements']        = df['text'].apply(
        lambda t: int(bool(re.search(r'\d+\s*(mm|cm|ml|cc)|\d+\.\d+|\d+\s*x\s*\d+',
                                     str(t), re.I))))
    df['F10_negation_count']          = df['text'].apply(
        lambda t: sum(str(t).lower().count(n) for n in negations_fr))

    # Pathology mapping in French
    pathology_map_fr = {
        'Normal':           ['normal', 'aspect normal'],
        'Cardiomegaly':     ['cardiomégalie', 'index cardio-thoracique'],
        'Mass':             ['masse', 'nodule', 'lésion'],
        'PleuralEffusion':  ['épanchement', 'pleural'],
        'Pneumonia':        ['pneumonie', 'condensation', 'foyer infectieux'],
        'Pneumothorax':     ['pneumothorax'],
    }
    df['F11_primary_pathology'] = df['text'].apply(
        lambda t: next((cat for cat, kws in pathology_map_fr.items()
                        if any(k in str(t).lower() for k in kws)), 'Other'))
    df['F12_multi_label_count'] = df['text'].apply(
        lambda t: sum(1 for cat, kws in pathology_map_fr.items()
                      if any(k in str(t).lower() for k in kws)))
    df['F12_is_multilabel'] = (df['F12_multi_label_count'] > 1).astype(int)

    # Linguistic complexity (basic Flesch-style)
    def fr_readability(text):
        words = str(text).split()
        sents = [s for s in re.split(r'[.!?]', str(text)) if s.strip()]
        if not words or not sents: return 0
        syls = sum(max(1, len(re.findall(r'[aeiouyàâäéèêëîïôöùûüœæ]+',
                                          w, re.I))) for w in words)
        return 0.39 * len(words) / len(sents) + 11.8 * syls / len(words) - 15.59

    df['F13_flesch_kincaid']    = df['text'].apply(fr_readability)
    common_abbrevs = ['CT', 'IRM', 'TDM', 'OAP', 'BPCO', 'AVC', 'AVP',
                      'ICT', 'PA', 'AP', 'CXR']
    df['F14_abbrev_density']    = df['text'].apply(
        lambda t: sum(1 for w in str(t).split() if w.upper() in common_abbrevs)
                  / max(len(str(t).split()), 1))
    df['F15_numerical_count']   = df['text'].apply(
        lambda t: len(re.findall(r'\b\d+\.?\d*\b', str(t))))
    df['F16_hedge_count']       = df['text'].apply(lambda t: count_terms(t, hedges_fr))
    df['F17_passive_ratio']     = df['text'].apply(
        lambda t: sum(str(t).lower().count(p) for p in ['est ', 'sont ', 'était ', 'été '])
                  / max(len(str(t).split()), 1))
    df['F18_temporal_count']    = df['text'].apply(lambda t: count_terms(t, temporal_fr))
    df['F19_has_recommendation']= df['text'].apply(
        lambda t: int(any(r in str(t).lower() for r in recommendations_fr)))

    # Patient-level features
    df['F20_patient_age']       = pd.to_numeric(df['patient_age'], errors='coerce').fillna(df['patient_age'].median() if df['patient_age'].notna().any() else 50)
    df['F21_is_male']           = (df['patient_gender'] == 'M').astype(int)

    # Composite features
    mx = lambda s: s.max() if s.max() > 0 else 1
    df['F22_composite_severity'] = (
        0.3 * df['F03_severity_score'] / mx(df['F03_severity_score']) +
        0.25 * df['F01_clinical_sentiment'] +
        0.2 * df['F02_clinical_entity_count'] / mx(df['F02_clinical_entity_count']) +
        0.15 * df['F12_multi_label_count'] / mx(df['F12_multi_label_count']) +
        0.1 * df['F16_hedge_count'] / mx(df['F16_hedge_count']))

    df['F23_pathology_confidence'] = (
        df['F04_diagnostic_term_freq'] * 0.4 +
        (1 - df['F16_hedge_count'] / mx(df['F16_hedge_count'])) * 0.3 +
        df['F02_clinical_entity_count'] / mx(df['F02_clinical_entity_count']) * 0.3)

    df['F24_report_complexity'] = (
        df['F06_word_count'] / mx(df['F06_word_count']) * 0.25 +
        df['F13_flesch_kincaid'] / max(abs(df['F13_flesch_kincaid']).max(), 1) * 0.25 +
        df['F15_numerical_count'] / mx(df['F15_numerical_count']) * 0.25 +
        df['F07_sentence_count'] / mx(df['F07_sentence_count']) * 0.25)

    df['F25_condition_label'] = df['condition']

    out = os.path.join(DATA_DIR, 'casia_cxr_features.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"  {sum(c.startswith('F') and c[1:3].isdigit() for c in df.columns)} "
          f"features engineered -> {out}")
    print(f"  Multi-class target distribution:")
    for k, v in df['F25_condition_label'].value_counts().items():
        print(f"    {k:<18} {v:>5,}")
    return df


# =============================================================================
# STEP 5: PREDICTIVE MODELING (Multi-class condition classifier)
# =============================================================================
def run_models(df: pd.DataFrame) -> dict:
    """Train LR / RF / XGB on structured features + TF-IDF for condition
    classification across the 5 CASIA-CXR pathology classes."""
    print("\n" + "=" * 60)
    print("STEP 5: CASIA-CXR PREDICTIVE MODELING (5-class)")
    print("=" * 60)

    feat_cols = [c for c in df.columns
                 if c.startswith('F') and c[1:3].isdigit()
                 and c not in ['F11_primary_pathology', 'F25_condition_label']]
    X_struct = df[feat_cols].fillna(0).values
    y = LabelEncoder().fit_transform(df['F25_condition_label'])
    classes = sorted(df['F25_condition_label'].unique())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = Pipeline([('sc', StandardScaler()),
                   ('lr', LogisticRegression(C=1.0, class_weight='balanced',
                                             max_iter=1000, multi_class='multinomial',
                                             random_state=42))])
    lr_f1 = cross_val_score(lr, X_struct, y, cv=cv, scoring='f1_macro')
    print(f"  LR  (structured)       — Macro-F1: {lr_f1.mean():.4f} ± {lr_f1.std():.4f}")

    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                max_depth=15, random_state=42, n_jobs=-1)
    rf_f1 = cross_val_score(rf, X_struct, y, cv=cv, scoring='f1_macro')
    print(f"  RF  (structured)       — Macro-F1: {rf_f1.mean():.4f} ± {rf_f1.std():.4f}")

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        objective='multi:softprob', num_class=len(classes),
                        random_state=42, eval_metric='mlogloss',
                        use_label_encoder=False, verbosity=0)
    xgb_f1 = cross_val_score(xgb, X_struct, y, cv=cv, scoring='f1_macro')
    print(f"  XGB (structured)       — Macro-F1: {xgb_f1.mean():.4f} ± {xgb_f1.std():.4f}")

    # TF-IDF on French Findings text
    tfidf_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                  min_df=3, max_df=0.95)),
        ('lr', LogisticRegression(C=1.0, class_weight='balanced',
                                  max_iter=2000, multi_class='multinomial',
                                  random_state=42))])
    txt = df['findings_clean'].fillna('').values
    tfidf_f1 = cross_val_score(tfidf_pipe, txt, y, cv=cv, scoring='f1_macro')
    print(f"  LR + TF-IDF (text)     — Macro-F1: {tfidf_f1.mean():.4f} ± {tfidf_f1.std():.4f}")

    results = {
        'classes': classes,
        'Logistic_Regression_structured': {'Macro_F1': float(lr_f1.mean()),
                                           'std': float(lr_f1.std())},
        'Random_Forest_structured':       {'Macro_F1': float(rf_f1.mean()),
                                           'std': float(rf_f1.std())},
        'XGBoost_structured':             {'Macro_F1': float(xgb_f1.mean()),
                                           'std': float(xgb_f1.std())},
        'LogisticRegression_TFIDF_text':  {'Macro_F1': float(tfidf_f1.mean()),
                                           'std': float(tfidf_f1.std())},
    }
    out = os.path.join(OUTPUT_DIR, 'casia_prediction_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {out}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ['LR-struct', 'RF-struct', 'XGB-struct', 'LR-TFIDF']
    vals  = [lr_f1.mean(), rf_f1.mean(), xgb_f1.mean(), tfidf_f1.mean()]
    stds  = [lr_f1.std(),  rf_f1.std(),  xgb_f1.std(),  tfidf_f1.std()]
    bars  = ax.bar(names, vals, yerr=stds, capsize=6,
                   color=sns.color_palette('Set2'), edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f'{v:.3f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Macro-F1 (5-fold CV)')
    ax.set_title('CASIA-CXR — 5-class Condition Classifier Comparison',
                 fontweight='bold')
    _save(fig, 'casia_fig08_model_comparison.png')

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  MedSum-AI — CASIA-CXR SECONDARY DATASET PIPELINE")
    print("=" * 70)

    df_raw     = load_casia_reports()
    stats      = run_eda(df_raw)
    df_clean   = clean_casia(df_raw)
    df_feat    = engineer_features(df_clean)
    results    = run_models(df_feat)

    print("\n" + "=" * 70)
    print("  CASIA-CXR PIPELINE COMPLETE!")
    print("  Outputs:")
    print("    - data/casia_cxr_combined.csv")
    print("    - data/casia_cxr_cleaned.csv")
    print("    - data/casia_cxr_features.csv")
    print("    - outputs/casia_eda_statistics_report.txt")
    print("    - outputs/casia_prediction_results.json")
    print("    - outputs/eda_figures_casia/casia_fig01..08_*.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
