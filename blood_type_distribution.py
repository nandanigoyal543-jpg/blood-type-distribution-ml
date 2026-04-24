# ============================================================
# Blood Type Distribution Prediction Model
# Author: Nandani Goyal
# Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, classification_report,
                              confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Generate / Load Dataset
# Simulates population-based blood group distribution data
# Replace with: df = pd.read_csv('blood_type_data.csv')
# ============================================================

np.random.seed(42)
n = 600

# Regions across India
regions   = ['North', 'South', 'East', 'West', 'Central']
age_groups = ['0-18', '19-35', '36-55', '56+']

# Global blood type frequencies (approximate)
blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
bt_probs    = [0.27, 0.06, 0.22, 0.02, 0.38, 0.07, 0.05, 0.01]
# Renormalize to sum to 1
bt_probs    = np.array(bt_probs) / sum(bt_probs)

region_col    = np.random.choice(regions, n)
age_col       = np.random.choice(age_groups, n)
blood_type_col = np.random.choice(blood_types, n, p=bt_probs)

# Numeric features
population_size  = np.random.randint(5000, 500000, n)
urban_rural      = np.random.choice([0, 1], n)          # 0=Rural, 1=Urban
literacy_rate    = np.random.uniform(0.5, 0.99, n)
healthcare_index = np.random.uniform(1, 10, n)

# Target 1: Percentage of population with that blood type (regression)
base_pct = {
    'A+': 27, 'A-': 6, 'B+': 22, 'B-': 2,
    'O+': 38, 'O-': 7, 'AB+': 5, 'AB-': 1
}
blood_pct = np.array([base_pct[bt] for bt in blood_type_col])
blood_pct = blood_pct + np.random.normal(0, 2, n)
blood_pct = np.clip(blood_pct, 0.5, 60)

df = pd.DataFrame({
    'Region':          region_col,
    'Age_Group':       age_col,
    'Blood_Type':      blood_type_col,
    'Population_Size': population_size,
    'Urban_Rural':     urban_rural,
    'Literacy_Rate':   literacy_rate,
    'Healthcare_Index': healthcare_index,
    'Blood_Type_Pct':  blood_pct
})

# Inject missing values
for col in ['Literacy_Rate', 'Healthcare_Index', 'Population_Size']:
    df.loc[df.sample(frac=0.04).index, col] = np.nan

print("=" * 55)
print("  BLOOD TYPE DISTRIBUTION PREDICTION MODEL")
print("=" * 55)
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nBlood type distribution in dataset:")
print(df['Blood_Type'].value_counts())


# ============================================================
# STEP 2: Data Preprocessing & Cleaning
# ============================================================

print("\n--- STEP 2: Data Cleaning ---")
print(f"Missing values:\n{df.isnull().sum()}")

df['Population_Size'].fillna(df['Population_Size'].median(), inplace=True)
df['Literacy_Rate'].fillna(df['Literacy_Rate'].mean(), inplace=True)
df['Healthcare_Index'].fillna(df['Healthcare_Index'].mean(), inplace=True)

print("✓ Missing values handled")


# ============================================================
# STEP 3: Feature Engineering & Encoding
# ============================================================

print("\n--- STEP 3: Feature Engineering ---")

# Encode categorical columns
le_region = LabelEncoder()
le_age    = LabelEncoder()
le_bt     = LabelEncoder()

df['Region_Enc']     = le_region.fit_transform(df['Region'])
df['Age_Group_Enc']  = le_age.fit_transform(df['Age_Group'])
df['Blood_Type_Enc'] = le_bt.fit_transform(df['Blood_Type'])

# Population bins
df['Pop_Bin'] = pd.cut(df['Population_Size'],
                        bins=[0, 50000, 150000, 350000, 600000],
                        labels=[0, 1, 2, 3])
df['Pop_Bin'] = df['Pop_Bin'].astype(float).fillna(0).astype(int)

# Interaction feature
df['Literacy_Health'] = df['Literacy_Rate'] * df['Healthcare_Index']

print("✓ Encoded: Region, Age_Group, Blood_Type")
print("✓ New features: Pop_Bin, Literacy_Health")


# ============================================================
# STEP 4: MODEL A — Regression (predict blood type percentage)
# ============================================================

print("\n--- STEP 4A: Regression Task (Blood Type %) ---")

reg_features = ['Region_Enc', 'Age_Group_Enc', 'Blood_Type_Enc',
                'Population_Size', 'Urban_Rural', 'Literacy_Rate',
                'Healthcare_Index', 'Pop_Bin', 'Literacy_Health']

X_reg = df[reg_features].copy().fillna(df[reg_features].median())
y_reg = df['Blood_Type_Pct']

scaler_reg  = StandardScaler()
X_reg_sc    = scaler_reg.fit_transform(X_reg)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_reg_sc, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree Reg":  DecisionTreeRegressor(max_depth=6, random_state=42)
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    mae    = mean_absolute_error(y_te, y_pred)
    r2     = r2_score(y_te, y_pred)
    reg_results[name] = {'model': model, 'y_pred': y_pred,
                         'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"\n{name}: RMSE={rmse:.3f} | MAE={mae:.3f} | R²={r2:.3f}")


# ============================================================
# STEP 5: MODEL B — Classification (predict blood type group)
# ============================================================

print("\n--- STEP 4B: Classification Task (Blood Type) ---")

clf_features = ['Region_Enc', 'Age_Group_Enc', 'Population_Size',
                'Urban_Rural', 'Literacy_Rate', 'Healthcare_Index',
                'Pop_Bin', 'Literacy_Health']

X_clf = df[clf_features].copy().fillna(df[clf_features].median())
y_clf = df['Blood_Type_Enc']

scaler_clf  = StandardScaler()
X_clf_sc    = scaler_clf.fit_transform(X_clf)

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_clf_sc, y_clf, test_size=0.2, random_state=42)

clf_models = {
    "Decision Tree Clf":  DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest Clf":  RandomForestClassifier(n_estimators=100, random_state=42)
}

clf_results = {}
for name, model in clf_models.items():
    model.fit(Xc_tr, yc_tr)
    yc_pred  = model.predict(Xc_te)
    acc      = accuracy_score(yc_te, yc_pred)
    clf_results[name] = {'model': model, 'y_pred': yc_pred, 'Accuracy': acc}
    print(f"\n{name}: Accuracy={acc:.3f}")


# ============================================================
# STEP 6: Visualizations
# ============================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Blood Type Distribution Analysis — ML Results",
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors_bt = ['#EF5350','#42A5F5','#66BB6A','#FFA726',
             '#AB47BC','#26C6DA','#FF7043','#78909C']

# --- Plot 1: Blood type distribution (pie chart) ---
ax1 = fig.add_subplot(gs[0, 0])
bt_counts = df['Blood_Type'].value_counts()
ax1.pie(bt_counts.values, labels=bt_counts.index,
        colors=colors_bt[:len(bt_counts)],
        autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 8})
ax1.set_title('Blood Type Distribution', fontweight='bold')

# --- Plot 2: Blood type by region (stacked bar) ---
ax2 = fig.add_subplot(gs[0, 1])
region_bt = df.groupby(['Region', 'Blood_Type']).size().unstack(fill_value=0)
region_bt_pct = region_bt.div(region_bt.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(region_bt_pct))
for i, bt in enumerate(region_bt_pct.columns):
    ax2.bar(region_bt_pct.index, region_bt_pct[bt],
            bottom=bottom, label=bt,
            color=colors_bt[i % len(colors_bt)], edgecolor='white')
    bottom += region_bt_pct[bt].values
ax2.set_title('Blood Type % by Region', fontweight='bold')
ax2.set_xlabel('Region'); ax2.set_ylabel('Percentage (%)')
ax2.legend(fontsize=7, loc='upper right', ncol=2)
ax2.tick_params(axis='x', rotation=30)

# --- Plot 3: EDA — Blood type % distribution boxplot ---
ax3 = fig.add_subplot(gs[0, 2])
bt_pct_bytype = [df[df['Blood_Type'] == bt]['Blood_Type_Pct'].values
                 for bt in blood_types]
bp = ax3.boxplot(bt_pct_bytype, patch_artist=True, labels=blood_types)
for patch, color in zip(bp['boxes'], colors_bt):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_title('Blood Type % Distribution\n(EDA Boxplot)', fontweight='bold')
ax3.set_xlabel('Blood Type'); ax3.set_ylabel('% in Population')
ax3.tick_params(axis='x', rotation=30)
ax3.grid(True, alpha=0.3)

# --- Plot 4: Regression — Model comparison ---
ax4 = fig.add_subplot(gs[1, 0])
r_names   = list(reg_results.keys())
r_metrics = {'RMSE': [reg_results[m]['RMSE'] for m in r_names],
             'MAE':  [reg_results[m]['MAE']  for m in r_names],
             'R²':   [reg_results[m]['R2']   for m in r_names]}
x = np.arange(len(r_names))
w = 0.25
for i, (metric, vals) in enumerate(r_metrics.items()):
    ax4.bar(x + i*w, vals, w, label=metric,
            color=['#EF5350','#42A5F5','#66BB6A'][i])
ax4.set_title('Regression Model Comparison', fontweight='bold')
ax4.set_xticks(x + w); ax4.set_xticklabels(['Lin Reg', 'Dec Tree'], rotation=15)
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')

# --- Plot 5: Actual vs Predicted (Linear Regression) ---
ax5 = fig.add_subplot(gs[1, 1])
lr_pred = reg_results['Linear Regression']['y_pred']
ax5.scatter(y_te, lr_pred, alpha=0.4, color='#3F51B5', s=15)
lim = [min(y_te.min(), lr_pred.min()), max(y_te.max(), lr_pred.max())]
ax5.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect fit')
ax5.set_title('Actual vs Predicted\n(Linear Regression)', fontweight='bold')
ax5.set_xlabel('Actual Blood Type %')
ax5.set_ylabel('Predicted Blood Type %')
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# --- Plot 6: Residuals (Decision Tree Regression) ---
ax6 = fig.add_subplot(gs[1, 2])
dt_pred   = reg_results['Decision Tree Reg']['y_pred']
residuals = y_te - dt_pred
ax6.hist(residuals, bins=30, color='#607D8B', edgecolor='white', alpha=0.8)
ax6.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax6.set_title('Residuals (Decision Tree)', fontweight='bold')
ax6.set_xlabel('Residual'); ax6.set_ylabel('Frequency')
ax6.grid(True, alpha=0.3)

# --- Plot 7: Age group distribution ---
ax7 = fig.add_subplot(gs[2, 0])
age_counts = df['Age_Group'].value_counts()
ax7.bar(age_counts.index, age_counts.values,
        color=['#26C6DA','#FFA726','#EF5350','#66BB6A'],
        edgecolor='white')
ax7.set_title('Samples by Age Group', fontweight='bold')
ax7.set_xlabel('Age Group'); ax7.set_ylabel('Count')
ax7.grid(True, alpha=0.3, axis='y')

# --- Plot 8: Feature importance (Random Forest) ---
ax8 = fig.add_subplot(gs[2, 1:])
rf_clf   = clf_results['Random Forest Clf']['model']
imp      = rf_clf.feature_importances_
idx      = np.argsort(imp)[::-1]
fn       = [clf_features[i] for i in idx]
ax8.barh(fn, imp[idx], color='#009688')
ax8.set_title('Feature Importance (Random Forest Classifier)', fontweight='bold')
ax8.set_xlabel('Importance Score')
ax8.grid(True, alpha=0.3, axis='x')

plt.savefig('blood_type_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Plot saved as 'blood_type_results.png'")

print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)
lr  = reg_results['Linear Regression']
dtr = reg_results['Decision Tree Reg']
rf  = clf_results['Random Forest Clf']
print(f"  Linear Regression → R²: {lr['R2']:.3f} | RMSE: {lr['RMSE']:.3f}")
print(f"  Decision Tree Reg → R²: {dtr['R2']:.3f} | RMSE: {dtr['RMSE']:.3f}")
print(f"  Random Forest Clf → Accuracy: {rf['Accuracy']:.3f}")
print("\n  Key insight: O+ and B+ are the most common blood")
print("  types. Regional variation exists but is less")
print("  significant than blood type base frequency.")
print("=" * 55)
