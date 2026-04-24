# 🩸 Blood Type Distribution Prediction Model

> Analyzing and modeling blood group distribution patterns across populations using ML

---

## 📌 Problem Statement
Analyze patterns in **blood group distribution** across populations using demographic and regional data, and build models to predict blood type percentage and classify blood groups.

---

## 📊 Dataset
- **Source:** Population-based blood group distribution dataset (demographic/regional data)
- **Size:** 600 records across 5 regions
- **Features:** Region, Age Group, Population Size, Urban/Rural, Literacy Rate, Healthcare Index

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas & NumPy | Data preprocessing, EDA |
| Scikit-learn | ML models, evaluation metrics |
| Matplotlib | Visualization |

---

## 🔄 Project Pipeline

```
Raw Population Data
      ↓
Data Cleaning (mean/median imputation)
      ↓
EDA (distributions, correlations, boxplots)
      ↓
Feature Engineering (Label encoding, Pop_Bin, Literacy×Health)
      ↓
Model A: Regression (predict blood type %)
Model B: Classification (predict blood type)
      ↓
Evaluation (R², MAE, RMSE, Accuracy)
      ↓
Feature Importance & Insights
```

---

## 🤖 Models Used

### Regression (predicting blood type % in population)
- **Linear Regression** — baseline
- **Decision Tree Regressor** (max_depth=6)

### Classification (predicting blood type)
- **Decision Tree Classifier** (max_depth=6)
- **Random Forest Classifier** (100 estimators)

---

## 📈 Results

### Regression
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | ~2.1 | ~1.7 | ~0.88 |
| Decision Tree | ~1.4 | ~1.0 | ~0.93 |

### Classification
| Model | Accuracy |
|-------|----------|
| Decision Tree | ~0.52 |
| Random Forest | ~0.58 |

> Note: Classification accuracy is limited because blood type is largely genetically determined and weakly correlated with regional demographics — a realistic real-world finding.

---

## 🔍 Key Insights
- **O+** (~38%) and **B+** (~22%) are the most prevalent blood types — consistent with global statistics
- Blood type percentage varies more by **blood type itself** than by region or demographics
- Regional variation exists but is smaller than expected — supports genetic basis of blood typing
- Healthcare Index and Literacy Rate show weak correlation with distribution patterns

---

## 📊 Visualizations Generated
1. Blood type distribution (pie chart)
2. Blood type % by region (stacked bar)
3. EDA boxplot — blood type % spread
4. Regression model comparison
5. Actual vs Predicted (Linear Regression)
6. Residuals distribution
7. Age group sample distribution
8. Feature importance (Random Forest)

---

## ▶️ How to Run

**Option 1 — Google Colab (recommended):**
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `blood_type_distribution.py`
3. Run all cells

**Option 2 — Local:**
```bash
pip install numpy pandas scikit-learn matplotlib
python blood_type_distribution.py
```

---

## 👩‍💻 Author
**Nandani Goyal**
B.Tech Biotechnology (Minor: AI/ML) | JIIT Delhi
- LinkedIn: [linkedin.com/in/nandani-goyal543](https://linkedin.com/in/nandani-goyal543)
- Email: nandanigoyal543@gmail.com
