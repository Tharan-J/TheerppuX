"""
Utility script to generate the comprehensive, presentation-ready Jupyter Notebook
`notebooks/stage1_model_evaluation.ipynb` adhering to all 8 required sections.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_notebook():
    cells = []

    # --- Header Cell ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TheerppuX — Stage 1: Multilingual Indian Legal Document Understanding\n",
                "## Empirical Benchmark & Model Selection for Tamil (`ta`) and Malayalam (`ml`)\n",
                "\n",
                "> **Target Domain:** Indian District & High Court Legal Judgments\n",
                "> **Primary Objective:** Comparative evaluation of multilingual translation configurations preserving legal terminology, statutory citations, numbers, dates, and factual integrity.\n",
                "\n",
                "---\n"
            ],
        }
    )

    # --- Setup & Imports Cell ---
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup environment and import analysis tools\n",
                "import sys\n",
                "import json\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add src to path\n",
                "PROJECT_ROOT = Path(\"..\").resolve()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "from src.config import load_config, SUPPORTED_TARGET_LANGUAGES\n",
                "from src.document.loader import DocumentLoader\n",
                "from src.preprocessing.cleaner import TextCleaner\n",
                "from src.analysis.report import ReportGenerator\n",
                "from src.analysis.consistency import CrossLanguageConsistencyChecker\n",
                "\n",
                "# Set publication aesthetic\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
                "plt.rcParams['figure.dpi'] = 120\n",
                "print(\"✓ Environment initialized successfully.\")"
            ],
        }
    )

    # --- Section 1: Objective ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Experiment Objective\n",
                "\n",
                "In Indian jurisprudence, judgments are predominantly authored in English while proceedings, litigants, and subordinate courts operate in regional languages. Translating English legal documents into South Indian Dravidian languages (**Tamil** and **Malayalam**) presents unique challenges:\n",
                "\n",
                "1. **Morphological Richness:** Tamil and Malayalam are agglutinative languages with complex postpositions and compounding.\n",
                "2. **Critical Fact Preservation:** Statutory sections (e.g., *Section 302 IPC* vs *Section 304*), dates, case numbers, and monetary figures (*₹50,000* vs *₹5,000*) admit zero tolerance for distortion.\n",
                "3. **Inadequacy of Generic BLEU:** A high n-gram overlap score can mask catastrophic legal errors such as party-role inversion (*plaintiff* $\\leftrightarrow$ *defendant*) or outcome reversal (*convicted* $\\leftrightarrow$ *acquitted*).\n",
                "\n",
                "**Goal:** Empirically evaluate three translation pipelines ($P1, P2, P3$) to select the optimal model configuration for downstream legal information extraction and grounded summarization."
            ],
        }
    )

    # --- Section 2: Dataset Summary ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Characterization\n",
                "\n",
                "The evaluation corpus comprises structured Indian legal documents with human-annotated reference translations in Tamil and Malayalam. **No Personally Identifiable Information (PII)** is exposed in benchmark representations."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inspect and summarize the evaluation dataset\n",
                "loader = DocumentLoader(ocr_enabled=False)\n",
                "cleaner = TextCleaner()\n",
                "raw_dir = PROJECT_ROOT / \"data\" / \"raw\"\n",
                "ref_ta_dir = PROJECT_ROOT / \"data\" / \"references\" / \"tamil\"\n",
                "ref_ml_dir = PROJECT_ROOT / \"data\" / \"references\" / \"malayalam\"\n",
                "\n",
                "case_records = []\n",
                "for f in sorted(raw_dir.glob(\"*.txt\")):\n",
                "    doc = loader.load(f, document_id=f.stem)\n",
                "    cleaned = cleaner.clean_document(doc)\n",
                "    has_ta = (ref_ta_dir / f\"{f.stem}.txt\").exists()\n",
                "    has_ml = (ref_ml_dir / f\"{f.stem}.txt\").exists()\n",
                "    case_records.append({\n",
                "        \"Case ID\": f.stem,\n",
                "        \"Format\": doc.file_type.upper(),\n",
                "        \"Pages\": doc.total_pages,\n",
                "        \"Words\": cleaned.total_words,\n",
                "        \"Chars\": len(cleaned.full_text),\n",
                "        \"Human Ref (Tamil)\": \"✓ Available\" if has_ta else \"✗ Missing\",\n",
                "        \"Human Ref (Malayalam)\": \"✓ Available\" if has_ml else \"✗ Missing\",\n",
                "        \"Type\": \"Digital Synthetic / Demo\"\n",
                "    })\n",
                "\n",
                "df_dataset = pd.DataFrame(case_records)\n",
                "df_dataset"
            ],
        }
    )

    # --- Section 3: Model Configurations ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Model & Pipeline Configurations\n",
                "\n",
                "We evaluate three distinct architectural configurations under identical experimental controls (same tokenization, segmentation, and prompt context):\n",
                "\n",
                "| Pipeline | Model Architecture | Domain Strategy | Checkpoint Identifier |\n",
                "| :--- | :--- | :--- | :--- |\n",
                "| **P1 (Baseline)** | Helsinki-NLP OPUS-MT (`MarianMT`) | Generic multilingual baseline | `opus-mt-en-dra` / `opus-mt-en-ml` |\n",
                "| **P2 (IndicTrans2)** | AI4Bharat IndicTrans2 (`Seq2SeqLM`) | Indic-specialized transfer learning | `indictrans2-en-indic-dist-200M` |\n",
                "| **P3 (Legal-Aware)** | IndicTrans2 + Legal Engine | Entity masking + Terminology validation + Post-repair | `indictrans2-en-indic-dist-200M` |"
            ],
        }
    )

    # --- Section 4: Automatic Metrics ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Quantitative Evaluation Metrics\n",
                "\n",
                "We compare generic NMT metrics (**BLEU-4**, **chrF++**, **ROUGE-L**, **BERTScore F1**) against legal-specific fidelity metrics (**Entity F1**, **Number Accuracy**, **Legal Terminology Accuracy**, and **Critical Error Rate**)."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load evaluation results from experiment outputs\n",
                "outputs_dir = PROJECT_ROOT / \"data\" / \"outputs\"\n",
                "\n",
                "def load_metrics_for_case(case_id: str):\n",
                "    metrics_file = outputs_dir / case_id / \"evaluation\" / \"metrics.json\"\n",
                "    if metrics_file.exists():\n",
                "        with open(metrics_file, \"r\", encoding=\"utf-8\") as f:\n",
                "            return json.load(f)\n",
                "    return None\n",
                "\n",
                "# Gather metrics across available outputs\n",
                "all_metrics = {}\n",
                "for case_dir in sorted(outputs_dir.glob(\"*\")):\n",
                "    if case_dir.is_dir():\n",
                "        m = load_metrics_for_case(case_dir.name)\n",
                "        if m:\n",
                "            all_metrics[case_dir.name] = m\n",
                "\n",
                "print(f\"Loaded evaluation results for cases: {list(all_metrics.keys())}\")"
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display Comparative Metric Table for Case 001\n",
                "case_key = \"case_001\" if \"case_001\" in all_metrics else list(all_metrics.keys())[0] if all_metrics else None\n",
                "if case_key and case_key in all_metrics:\n",
                "    c_metrics = all_metrics[case_key]\n",
                "    metric_names = [\n",
                "        (\"BLEU-4 (SacreBLEU)\", \"bleu\"),\n",
                "        (\"chrF++ (Morphological)\", \"chrf_plus_plus\"),\n",
                "        (\"ROUGE-L (Supplementary)\", \"rouge_l\"),\n",
                "        (\"BERTScore F1 (Semantic)\", \"bertscore_f1\"),\n",
                "        (\"Entity F1 (%)\", \"entity_f1\"),\n",
                "        (\"Number Accuracy (%)\", \"number_accuracy\"),\n",
                "        (\"Legal Term Accuracy (%)\", \"legal_term_accuracy\"),\n",
                "        (\"Critical Error Rate (%)\", \"critical_error_rate_pct\"),\n",
                "        (\"Inference Latency (sec)\", \"latency_seconds\"),\n",
                "    ]\n",
                "    rows = []\n",
                "    for label, key in metric_names:\n",
                "        rows.append({\n",
                "            \"Evaluation Metric\": label,\n",
                "            \"P1 (Baseline OPUS-MT)\": c_metrics.get(\"baseline\", {}).get(key, \"-\"),\n",
                "            \"P2 (IndicTrans2)\": c_metrics.get(\"indictrans2\", {}).get(key, \"-\"),\n",
                "            \"P3 (Legal-Aware IndicTrans2)\": c_metrics.get(\"legal_aware\", {}).get(key, \"-\"),\n",
                "        })\n",
                "    df_comp = pd.DataFrame(rows)\n",
                "    display(df_comp)\n",
                "else:\n",
                "    print(\"Run experiments first via `python -m src.cli experiment` to populate metrics.\")"
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize Model Performance Comparison across Core Dimensions\n",
                "if case_key and case_key in all_metrics:\n",
                "    categories = ['BLEU-4', 'chrF++', 'Entity F1', 'Number Acc', 'Legal Term Acc']\n",
                "    keys = ['bleu', 'chrf_plus_plus', 'entity_f1', 'number_accuracy', 'legal_term_accuracy']\n",
                "    \n",
                "    p1_scores = [float(all_metrics[case_key].get('baseline', {}).get(k, 0)) for k in keys]\n",
                "    p2_scores = [float(all_metrics[case_key].get('indictrans2', {}).get(k, 0)) for k in keys]\n",
                "    p3_scores = [float(all_metrics[case_key].get('legal_aware', {}).get(k, 0)) for k in keys]\n",
                "    \n",
                "    x = np.arange(len(categories))\n",
                "    width = 0.25\n",
                "    \n",
                "    fig, ax = plt.subplots(figsize=(10, 5))\n",
                "    ax.bar(x - width, p1_scores, width, label='P1: Baseline OPUS-MT', color='#6c757d')\n",
                "    ax.bar(x, p2_scores, width, label='P2: IndicTrans2', color='#0d6efd')\n",
                "    ax.bar(x + width, p3_scores, width, label='P3: Legal-Aware IndicTrans2', color='#198754')\n",
                "    \n",
                "    ax.set_ylabel('Score / Percentage (%)', fontsize=12)\n",
                "    ax.set_title(f'Legal Translation Benchmark Comparison ({case_key})', fontsize=14, fontweight='bold')\n",
                "    ax.set_xticks(x)\n",
                "    ax.set_xticklabels(categories, fontsize=11)\n",
                "    ax.legend(frameon=True, facecolor='white', loc='lower right')\n",
                "    ax.set_ylim(0, 110)\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ],
        }
    )

    # --- Section 5: Error Analysis ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Qualitative Error Analysis (E1–E12 Taxonomy)\n",
                "\n",
                "We inspect concrete examples comparing the **Source English Legal Clause**, **Human Reference**, **P1**, **P2**, and **P3** to diagnose specific failure modes."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load and display side-by-side translation examples with error annotations\n",
                "if case_key and (outputs_dir / case_key / \"translations\").exists():\n",
                "    trans_dir = outputs_dir / case_key / \"translations\"\n",
                "    p1_file = list(trans_dir.glob(\"baseline_*.json\"))\n",
                "    p2_file = list(trans_dir.glob(\"indictrans2_*.json\"))\n",
                "    p3_file = list(trans_dir.glob(\"legal_aware_*.json\"))\n",
                "    \n",
                "    chunks_file = outputs_dir / case_key / \"chunks\" / \"chunks.json\"\n",
                "    with open(chunks_file, \"r\", encoding=\"utf-8\") as f:\n",
                "        chunks_data = json.load(f)\n",
                "        \n",
                "    t1 = json.load(open(p1_file[0]))[\"translations\"] if p1_file else []\n",
                "    t2 = json.load(open(p2_file[0]))[\"translations\"] if p2_file else []\n",
                "    t3 = json.load(open(p3_file[0]))[\"translations\"] if p3_file else []\n",
                "    \n",
                "    print(\"=\" * 85)\n",
                "    print(\"QUALITATIVE COMPARATIVE INSPECTION (SAMPLE CHUNKS)\")\n",
                "    print(\"=\" * 85)\n",
                "    for idx in range(min(3, len(chunks_data))):\n",
                "        print(f\"\\n[CHUNK {idx+1} — {chunks_data[idx].get('section_type', 'BODY')}]\")\n",
                "        print(f\"SOURCE (EN)    : {chunks_data[idx]['text']}\")\n",
                "        print(f\"P1 (BASELINE)  : {t1[idx] if idx < len(t1) else 'N/A'}\")\n",
                "        print(f\"P2 (INDICTRANS): {t2[idx] if idx < len(t2) else 'N/A'}\")\n",
                "        print(f\"P3 (LEGAL-AWR) : {t3[idx] if idx < len(t3) else 'N/A'}\")\n",
                "        print(\"-\" * 85)"
            ],
        }
    )

    # --- Section 6: Case-Level Analysis ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Case-Level Granular Breakdown\n",
                "\n",
                "Aggregate metrics can obscure variance between criminal appeals (Section 302 IPC) and commercial negotiable instrument disputes (Section 138 NI Act). Below is the per-case comparative performance breakdown."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tabulate performance across multiple cases\n",
                "case_summary = []\n",
                "for c_id, metrics in all_metrics.items():\n",
                "    for m_name in ['baseline', 'indictrans2', 'legal_aware']:\n",
                "        m_dict = metrics.get(m_name, {})\n",
                "        case_summary.append({\n",
                "            \"Case ID\": c_id,\n",
                "            \"Model Pipeline\": m_name,\n",
                "            \"BLEU-4\": m_dict.get(\"bleu\", \"-\"),\n",
                "            \"chrF++\": m_dict.get(\"chrf_plus_plus\", \"-\"),\n",
                "            \"Entity F1\": m_dict.get(\"entity_f1\", \"-\"),\n",
                "            \"Number Acc (%)\": m_dict.get(\"number_accuracy\", \"-\"),\n",
                "            \"Legal Term Acc (%)\": m_dict.get(\"legal_term_accuracy\", \"-\"),\n",
                "            \"Critical Errors\": m_dict.get(\"critical_errors_count\", 0),\n",
                "        })\n",
                "df_case_breakdown = pd.DataFrame(case_summary)\n",
                "df_case_breakdown"
            ],
        }
    )

    # --- Section 7: Tamil vs Malayalam ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Tamil vs Malayalam Cross-Language Performance & Consistency\n",
                "\n",
                "Comparing model performance across both Dravidian target languages reveals differences driven by script vocabulary size and morphological inflection patterns."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cross-Language Comparison Table\n",
                "ta_metrics = all_metrics.get(\"case_001\", {}).get(\"legal_aware\", {})\n",
                "ml_metrics = all_metrics.get(\"case_002\", {}).get(\"legal_aware\", {})\n",
                "\n",
                "lang_comp_rows = [\n",
                "    {\"Criterion\": \"BLEU-4\", \"Tamil (ta)\": ta_metrics.get(\"bleu\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"bleu\", \"-\")},\n",
                "    {\"Criterion\": \"chrF++\", \"Tamil (ta)\": ta_metrics.get(\"chrf_plus_plus\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"chrf_plus_plus\", \"-\")},\n",
                "    {\"Criterion\": \"BERTScore F1\", \"Tamil (ta)\": ta_metrics.get(\"bertscore_f1\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"bertscore_f1\", \"-\")},\n",
                "    {\"Criterion\": \"Entity Preservation F1\", \"Tamil (ta)\": ta_metrics.get(\"entity_f1\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"entity_f1\", \"-\")},\n",
                "    {\"Criterion\": \"Number Preservation (%)\", \"Tamil (ta)\": ta_metrics.get(\"number_accuracy\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"number_accuracy\", \"-\")},\n",
                "    {\"Criterion\": \"Legal Terminology Acc (%)\", \"Tamil (ta)\": ta_metrics.get(\"legal_term_accuracy\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"legal_term_accuracy\", \"-\")},\n",
                "    {\"Criterion\": \"Critical Error Rate (%)\", \"Tamil (ta)\": ta_metrics.get(\"critical_error_rate_pct\", \"-\"), \"Malayalam (ml)\": ml_metrics.get(\"critical_error_rate_pct\", \"-\")},\n",
                "]\n",
                "df_lang_comp = pd.DataFrame(lang_comp_rows)\n",
                "display(df_lang_comp)"
            ],
        }
    )

    # --- Section 8: Model Selection Scorecard ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Final Model Selection Scorecard\n",
                "\n",
                "To determine the primary translation engine for Stage 2 without arbitrary hardcoded bias, we implement a **configurable multi-criteria decision matrix**."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Configurable Multi-Criteria Scorecard\n",
                "def compute_scorecard(metrics_dict: dict, weights: dict = None):\n",
                "    if weights is None:\n",
                "        weights = {\n",
                "            \"legal_term_accuracy\": 0.25,\n",
                "            \"number_accuracy\": 0.25,\n",
                "            \"entity_f1\": 0.20,\n",
                "            \"chrf_plus_plus\": 0.15,\n",
                "            \"bleu\": 0.15,\n",
                "        }\n",
                "    \n",
                "    scorecard = []\n",
                "    for model_key in ['baseline', 'indictrans2', 'legal_aware']:\n",
                "        m = metrics_dict.get(model_key, {})\n",
                "        weighted_score = 0.0\n",
                "        for k, w in weights.items():\n",
                "            val = float(m.get(k, 0))\n",
                "            weighted_score += val * w\n",
                "        \n",
                "        # Penalize critical errors\n",
                "        crit_err = float(m.get('critical_error_rate_pct', 0))\n",
                "        final_score = max(0.0, weighted_score - (crit_err * 2.0))\n",
                "        \n",
                "        scorecard.append({\n",
                "            \"Pipeline\": model_key.upper(),\n",
                "            \"Weighted Fidelity Score\": round(weighted_score, 2),\n",
                "            \"Critical Error Penalty\": round(crit_err * 2.0, 2),\n",
                "            \"Composite Legal Score\": round(final_score, 2),\n",
                "        })\n",
                "    return pd.DataFrame(scorecard).sort_values(\"Composite Legal Score\", ascending=False)\n",
                "\n",
                "if case_key and case_key in all_metrics:\n",
                "    df_scorecard = compute_scorecard(all_metrics[case_key])\n",
                "    display(df_scorecard)\n",
                "    winner = df_scorecard.iloc[0]['Pipeline']\n",
                "    print(f\"\\n🏆 Selected Optimal Pipeline for Stage 2 Integration: {winner}\")"
            ],
        }
    )

    # --- Section 9: Human Evaluation Analysis ---
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Human Expert Evaluation Module\n",
                "\n",
                "Automatic metrics are validated against expert human evaluations using the standardized 5-point Likert scale (*Factual Accuracy, Legal Fidelity, Fluency, Completeness, Terminology Accuracy*)."
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze Human Evaluation CSV if completed annotations are present\n",
                "eval_csv_path = outputs_dir / \"case_001\" / \"evaluation\" / \"human_evaluation_template.csv\"\n",
                "if eval_csv_path.exists():\n",
                "    human_stats = ReportGenerator.analyze_human_evaluation_csv(eval_csv_path)\n",
                "    print(f\"Human Evaluation Status: {human_stats.get('status', 'Completed')}\")\n",
                "    if \"stats_by_model\" in human_stats:\n",
                "        display(pd.DataFrame(human_stats[\"stats_by_model\"]))\n",
                "else:\n",
                "    print(\"Human evaluation template available at data/outputs/*/evaluation/human_evaluation_template.csv\")"
            ],
        }
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = PROJECT_ROOT / "notebooks" / "stage1_model_evaluation.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated complete Jupyter Notebook at: {nb_path}")


if __name__ == "__main__":
    build_notebook()
