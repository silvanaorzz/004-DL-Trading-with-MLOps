# 🧮 Advanced Trading Project — Deep Learning Trading with MLOps  
### TA Grading Instructions (ChatGPT Support Version)

---

## 🎯 Overview

This document provides detailed grading instructions for evaluating the **Advanced Trading Project** submissions.  
Each student submits:
- A **GitHub repository** containing their code (feature engineering, CNN models, MLFlow tracking, FastAPI endpoint, backtesting, etc.)
- An **Executive Report** explaining their strategy, methodology, results, and conclusions.

The grading process emphasizes both **ML engineering implementation (40%)** and **analytical/report quality (60%)**.

---

## 🧩 Grading Workflow

1. **Load Student Materials**
   - Review the GitHub repository for structure, code quality, and deliverables.
   - Review the Executive Report (PDF, Markdown, or Notebook section).
   - Note any missing or incomplete deliverables.

2. **Evaluate by Category** using the weighted rubric below.

3. **Provide concise feedback** (2–4 sentences per category).

4. **Compute the final numeric grade (0–100)** based on weighted averages.

5. **Summarize Overall Evaluation** in 3–5 sentences.

---

## ⚖️ Weighted Rubric (Total = 100 points)

### **A. Code Implementation (40 points total)**

| Subcategory | Weight | Evaluation Criteria |
|--------------|---------|---------------------|
| **A1. Code Structure & Documentation** | 10 | Organized repo with clear modular structure, readable code, and a setup-ready README. |
| **A2. Feature Engineering Implementation** | 5 | At least 20 features including momentum, volatility, and volume; normalization and missing data handling. |
| **A3. Model Architecture (CNN)** | 10 | Functional CNN in TensorFlow/Keras; proper convolution/pooling layers; class weighting applied. |
| **A4. MLFlow Integration** | 7 | Tracks experiments, hyperparameters, metrics, and model artifacts; allows comparison of CNN variants. |
| **A5. FastAPI Predict Endpoint** | 8 | Working `/predict` endpoint; loads model and returns correct predictions; clean implementation. |

---

### **B. Executive Report (60 points total)**

| Subcategory | Weight | Evaluation Criteria |
|--------------|---------|---------------------|
| **B1. Strategy Overview & Objectives** | 5 | Clearly describes the problem, rationale for CNN-based trading, and goals. |
| **B2. Feature Engineering & Target Definition** | 10 | Explains chosen indicators and target labels; discusses normalization and class imbalance handling. |
| **B3. Model Design & Training** | 10 | Details CNN architecture, optimizer, loss, and training setup; explains class weighting and data splits. |
| **B4. MLFlow Experimentation Summary** | 10 | Shows experiment tracking results, model comparison tables, and final model justification. |
| **B5. Data Drift Monitoring** | 10 | Includes methodology (KS-test, Chi-squared), Streamlit dashboard concept, and interpretation of drift results. |
| **B6. Backtesting & Performance Analysis** | 10 | Provides equity curve, realistic trading costs, performance ratios (Sharpe, Sortino, etc.), and insights on profitability. |
| **B7. Conclusions & Recommendations** | 5 | Summarizes key findings, limitations, and potential improvements. |

---

## 🧭 Scoring Guide

| Range | Description |
|--------|--------------|
| **90–100 (Excellent)** | Meets or exceeds expectations with strong implementation and explanations. |
| **75–89 (Good)** | Generally correct and complete with minor issues or unclear areas. |
| **60–74 (Fair)** | Functional but missing detail, weak justifications, or partial implementations. |
| **<60 (Poor)** | Major requirements missing or incorrect; limited understanding or documentation. |

---

## 💬 Feedback Template

**Project:** [Student Name or Repo Name]  
**Total Grade:** XX / 100  

### A. Code Implementation (40%) — [Score: X/40]  
*Brief 2–4 sentence qualitative feedback.*

### B. Executive Report (60%) — [Score: X/60]  
*Brief 2–4 sentence qualitative feedback.*

### 💡 Overall Comments  
*3–5 sentences summarizing the project’s strengths, weaknesses, and technical rigor.*

---

## 🧰 TA Notes for Grading

- Prioritize **depth and correctness** over surface-level polish.  
- Verify CNN implementation and MLFlow experiment tracking.  
- Confirm **chronological data splits** and no data leakage.  
- Check that the **FastAPI `/predict`** endpoint works and loads the trained model properly.  
- Review the **data drift section** for correct use of KS-test or other statistical drift detection.  
- In the report, emphasize interpretation quality and connection between model accuracy and trading profitability.  
- Reward analytical thinking, documentation, and professional communication.

---