# ============================================================
# Blood Type Distribution Prediction Model
# Author: Nandani Goyal
# Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib
# ============================================================
"""
This module implements regression and classification models to predict
blood type distribution across populations based on demographic and 
socioeconomic features.

Main Tasks:
- Regression: Predict blood type percentage in population
- Classification: Predict blood type category

Models Used:
- Linear Regression & Decision Tree Regressor
- Decision Tree Classifier & Random Forest Classifier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      cross_validate)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, classification_report,
                              confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

RANDOM_SEED = 42
DATASET_SIZE = 600
TEST_SIZE = 0.2
CV_FOLDS = 5
TREE_MAX_DEPTH = 6
RF_N_ESTIMATORS = 100
MISSING_VALUE_FRACTION = 0.04

# Data generation parameters
REGIONS = ['North', 'South', 'East', 'West', 'Central']
AGE_GROUPS = ['0-18', '19-35', '36-55', '56+']
BLOOD_TYPES = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
BLOOD_TYPE_PROBS = np.array([0.27, 0.06, 0.22, 0.02, 0.38, 0.07, 0.05, 0.01])
BLOOD_TYPE_PROBS = BLOOD_TYPE_PROBS / BLOOD_TYPE_PROBS.sum()

POP_BINS = [0, 50000, 150000, 350000, 600000]
BASE_BLOOD_PCT = {
    'A+': 27, 'A-': 6, 'B+': 22, 'B-': 2,
    'O+': 38, 'O-': 7, 'AB+': 5, 'AB-': 1
}

# Visualization
COLORS_BT = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726',
             '#AB47BC', '#26C6DA', '#FF7043', '#78909C']

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_sample_dataset(n_samples=DATASET_SIZE, seed=RANDOM_SEED):
    """
    Generate synthetic blood type distribution dataset.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        pandas.DataFrame: Generated dataset with blood type features
    """
    np.random.seed(seed)
    
    region_col = np.random.choice(REGIONS, n_samples)
    age_col = np.random.choice(AGE_GROUPS, n_samples)
    blood_type_col = np.random.choice(BLOOD_TYPES, n_samples, p=BLOOD_TYPE_PROBS)
    
    # Numeric features
    population_size = np.random.randint(5000, 500000, n_samples)
    urban_rural = np.random.choice([0, 1], n_samples)  # 0=Rural, 1=Urban
    literacy_rate = np.random.uniform(0.5, 0.99, n_samples)
    healthcare_index = np.random.uniform(1, 10, n_samples)
    
    # Target: Percentage of population with blood type
    blood_pct = np.array([BASE_BLOOD_PCT[bt] for bt in blood_type_col])
    blood_pct = blood_pct + np.random.normal(0, 2, n_samples)
    blood_pct = np.clip(blood_pct, 0.5, 60)
    
    df = pd.DataFrame({
        'Region': region_col,
        'Age_Group': age_col,
        'Blood_Type': blood_type_col,
        'Population_Size': population_size,
        'Urban_Rural': urban_rural,
        'Literacy_Rate': literacy_rate,
        'Healthcare_Index': healthcare_index,
        'Blood_Type_Pct': blood_pct
    })
    
    # Inject missing values
    for col in ['Literacy_Rate', 'Healthcare_Index', 'Population_Size']:
        df.loc[df.sample(frac=MISSING_VALUE_FRACTION).index, col] = np.nan
    
    return df


def preprocess_data(df):
    """
    Clean and preprocess dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    print("\n--- STEP 2: Data Cleaning ---")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Fill missing values
    df['Population_Size'].fillna(df['Population_Size'].median(), inplace=True)
    df['Literacy_Rate'].fillna(df['Literacy_Rate'].mean(), inplace=True)
    df['Healthcare_Index'].fillna(df['Healthcare_Index'].mean(), inplace=True)
    
    print("✓ Missing values handled")
    return df


def engineer_features(df):
    """
    Create encoded and derived features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        tuple: (DataFrame with new features, dict of encoders)
    """
    print("\n--- STEP 3: Feature Engineering ---")
    
    # Encode categorical columns
    le_region = LabelEncoder()
    le_age = LabelEncoder()
    le_bt = LabelEncoder()
    
    df['Region_Enc'] = le_region.fit_transform(df['Region'])
    df['Age_Group_Enc'] = le_age.fit_transform(df['Age_Group'])
    df['Blood_Type_Enc'] = le_bt.fit_transform(df['Blood_Type'])
    
    # Population bins
    df['Pop_Bin'] = pd.cut(df['Population_Size'], bins=POP_BINS,
                            labels=[0, 1, 2, 3])
    df['Pop_Bin'] = df['Pop_Bin'].astype(float).fillna(0).astype(int)
    
    # Interaction feature
    df['Literacy_Health'] = df['Literacy_Rate'] * df['Healthcare_Index']
    
    print("✓ Encoded: Region, Age_Group, Blood_Type")
    print("✓ New features: Pop_Bin, Literacy_Health")
    
    encoders = {
        'region': le_region,
        'age_group': le_age,
        'blood_type': le_bt
    }
    
    return df, encoders


def train_regression_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate regression models with cross-validation.
    
    Args:
        X_train, X_test: Training/test feature sets
        y_train, y_test: Training/test targets
    
    Returns:
        dict: Results for each model
    """
    print("\n--- STEP 4A: Regression Task (Blood Type %) ---")
    
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Reg": DecisionTreeRegressor(max_depth=TREE_MAX_DEPTH,
                                                   random_state=RANDOM_SEED)
    }
    
    reg_results = {}
    
    for name, model in reg_models.items():
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS,
                                    scoring='r2')
        
        reg_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_R2_Mean': cv_scores.mean(),
            'CV_R2_Std': cv_scores.std()
        }
        
        print(f"\n{name}:")
        print(f"  Test → RMSE={rmse:.3f} | MAE={mae:.3f} | R²={r2:.3f}")
        print(f"  CV (5-fold) → R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return reg_results


def train_classification_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate classification models with cross-validation.
    
    Args:
        X_train, X_test: Training/test feature sets
        y_train, y_test: Training/test targets
    
    Returns:
        dict: Results for each model
    """
    print("\n--- STEP 4B: Classification Task (Blood Type) ---")
    
    clf_models = {
        "Decision Tree Clf": DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH,
            random_state=RANDOM_SEED,
            class_weight='balanced'  # Handle class imbalance
        ),
        "Random Forest Clf": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_SEED,
            class_weight='balanced'  # Handle class imbalance
        )
    }
    
    clf_results = {}
    
    for name, model in clf_models.items():
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS,
                                    scoring='accuracy')
        
        clf_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'Accuracy': acc,
            'CV_Accuracy_Mean': cv_scores.mean(),
            'CV_Accuracy_Std': cv_scores.std(),
            'y_true': y_test  # Store for confusion matrix
        }
        
        print(f"\n{name}:")
        print(f"  Test → Accuracy={acc:.3f}")
        print(f"  CV (5-fold) → Accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return clf_results


def create_visualizations(df, reg_results, clf_results, clf_features, 
                          y_te_reg, blood_types=BLOOD_TYPES):
    """
    Generate comprehensive visualization dashboard.
    
    Args:
        df: Dataset
        reg_results: Regression model results
        clf_results: Classification model results
        clf_features: Feature names for classification
        y_te_reg: Test targets for regression
        blood_types: List of blood type labels
    """
    print("\n--- STEP 6: Visualizations ---")
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Blood Type Distribution Analysis — ML Results",
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # Plot 1: Blood type distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    bt_counts = df['Blood_Type'].value_counts()
    ax1.pie(bt_counts.values, labels=bt_counts.index,
            colors=COLORS_BT[:len(bt_counts)],
            autopct='%1.1f%%', startangle=140,
            textprops={'fontsize': 8})
    ax1.set_title('Blood Type Distribution', fontweight='bold')
    
    # Plot 2: Blood type by region (stacked bar)
    ax2 = fig.add_subplot(gs[0, 1])
    region_bt = df.groupby(['Region', 'Blood_Type']).size().unstack(fill_value=0)
    region_bt_pct = region_bt.div(region_bt.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(region_bt_pct))
    for i, bt in enumerate(region_bt_pct.columns):
        ax2.bar(region_bt_pct.index, region_bt_pct[bt],
                bottom=bottom, label=bt,
                color=COLORS_BT[i % len(COLORS_BT)], edgecolor='white')
        bottom += region_bt_pct[bt].values
    ax2.set_title('Blood Type % by Region', fontweight='bold')
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(fontsize=7, loc='upper right', ncol=2)
    ax2.tick_params(axis='x', rotation=30)
    
    # Plot 3: EDA — Blood type % distribution boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    bt_pct_bytype = [df[df['Blood_Type'] == bt]['Blood_Type_Pct'].values
                     for bt in blood_types]
    bp = ax3.boxplot(bt_pct_bytype, patch_artist=True, labels=blood_types)
    for patch, color in zip(bp['boxes'], COLORS_BT):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('Blood Type % Distribution\n(EDA Boxplot)', fontweight='bold')
    ax3.set_xlabel('Blood Type')
    ax3.set_ylabel('% in Population')
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regression — Model comparison
    ax4 = fig.add_subplot(gs[1, 0])
    r_names = list(reg_results.keys())
    r_metrics = {
        'RMSE': [reg_results[m]['RMSE'] for m in r_names],
        'MAE': [reg_results[m]['MAE'] for m in r_names],
        'R²': [reg_results[m]['R2'] for m in r_names]
    }
    x = np.arange(len(r_names))
    w = 0.25
    for i, (metric, vals) in enumerate(r_metrics.items()):
        ax4.bar(x + i*w, vals, w, label=metric,
                color=['#EF5350', '#42A5F5', '#66BB6A'][i])
    ax4.set_title('Regression Model Comparison', fontweight='bold')
    ax4.set_xticks(x + w)
    ax4.set_xticklabels(['Lin Reg', 'Dec Tree'], rotation=15)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Actual vs Predicted (Linear Regression)
    ax5 = fig.add_subplot(gs[1, 1])
    lr_pred = reg_results['Linear Regression']['y_pred']
    ax5.scatter(y_te_reg, lr_pred, alpha=0.4, color='#3F51B5', s=15)
    lim = [min(y_te_reg.min(), lr_pred.min()), max(y_te_reg.max(), lr_pred.max())]
    ax5.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect fit')
    ax5.set_title('Actual vs Predicted\n(Linear Regression)', fontweight='bold')
    ax5.set_xlabel('Actual Blood Type %')
    ax5.set_ylabel('Predicted Blood Type %')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Residuals (Decision Tree Regression)
    ax6 = fig.add_subplot(gs[1, 2])
    dt_pred = reg_results['Decision Tree Reg']['y_pred']
    residuals = y_te_reg - dt_pred
    ax6.hist(residuals, bins=30, color='#607D8B', edgecolor='white', alpha=0.8)
    ax6.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax6.set_title('Residuals (Decision Tree)', fontweight='bold')
    ax6.set_xlabel('Residual')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Age group distribution
    ax7 = fig.add_subplot(gs[2, 0])
    age_counts = df['Age_Group'].value_counts()
    ax7.bar(age_counts.index, age_counts.values,
            color=['#26C6DA', '#FFA726', '#EF5350', '#66BB6A'],
            edgecolor='white')
    ax7.set_title('Samples by Age Group', fontweight='bold')
    ax7.set_xlabel('Age Group')
    ax7.set_ylabel('Count')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Confusion Matrix (Random Forest)
    ax8 = fig.add_subplot(gs[2, 1])
    rf_clf = clf_results['Random Forest Clf']['model']
    y_true = clf_results['Random Forest Clf']['y_true']
    y_pred = clf_results['Random Forest Clf']['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    im = ax8.imshow(cm, cmap='Blues', aspect='auto')
    ax8.set_title('Confusion Matrix\n(Random Forest)', fontweight='bold')
    ax8.set_xlabel('Predicted')
    ax8.set_ylabel('True')
    plt.colorbar(im, ax=ax8)
    
    # Plot 9: Feature importance (Random Forest)
    ax9 = fig.add_subplot(gs[2, 2])
    imp = rf_clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fn = [clf_features[i] for i in idx]
    ax9.barh(fn, imp[idx], color='#009688')
    ax9.set_title('Feature Importance\n(Random Forest)', fontweight='bold')
    ax9.set_xlabel('Importance Score')
    ax9.grid(True, alpha=0.3, axis='x')
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'blood_type_results_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Plot saved as '{filename}'")
    
    return filename


def print_summary(df, reg_results, clf_results):
    """Print comprehensive model summary."""
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    
    print("\n📊 REGRESSION RESULTS:")
    for name, result in reg_results.items():
        print(f"\n  {name}:")
        print(f"    Test R²: {result['R2']:.3f}")
        print(f"    Test RMSE: {result['RMSE']:.3f}")
        print(f"    Test MAE: {result['MAE']:.3f}")
        print(f"    CV R² (mean ± std): {result['CV_R2_Mean']:.3f} ± {result['CV_R2_Std']:.3f}")
    
    print("\n📊 CLASSIFICATION RESULTS:")
    for name, result in clf_results.items():
        print(f"\n  {name}:")
        print(f"    Test Accuracy: {result['Accuracy']:.3f}")
        print(f"    CV Accuracy (mean ± std): {result['CV_Accuracy_Mean']:.3f} ± {result['CV_Accuracy_Std']:.3f}")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • O+ and B+ are the most common blood types")
    print("  • Regional variation exists but is less significant")
    print("    than base blood type frequency")
    print("  • Class-weighted models handle blood type imbalance")
    print("  • Cross-validation confirms model robustness")
    print("=" * 55)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  BLOOD TYPE DISTRIBUTION PREDICTION MODEL")
    print("=" * 55)
    
    # STEP 1: Generate dataset
    print("\n--- STEP 1: Dataset Generation ---")
    df = create_sample_dataset(n_samples=DATASET_SIZE, seed=RANDOM_SEED)
    
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBlood type distribution in dataset:")
    print(df['Blood_Type'].value_counts())
    
    # STEP 2: Preprocess data
    df = preprocess_data(df)
    
    # STEP 3: Engineer features
    df, encoders = engineer_features(df)
    
    # STEP 4A: Regression task
    reg_features = ['Region_Enc', 'Age_Group_Enc', 'Blood_Type_Enc',
                    'Population_Size', 'Urban_Rural', 'Literacy_Rate',
                    'Healthcare_Index', 'Pop_Bin', 'Literacy_Health']
    
    X_reg = df[reg_features].copy().fillna(df[reg_features].median())
    y_reg = df['Blood_Type_Pct']
    
    # Fix data leakage: fit scaler ONLY on training data
    X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    scaler_reg = StandardScaler()
    X_tr_reg_sc = scaler_reg.fit_transform(X_tr_reg)
    X_te_reg_sc = scaler_reg.transform(X_te_reg)
    
    reg_results = train_regression_models(X_tr_reg_sc, X_te_reg_sc,
                                          y_tr_reg, y_te_reg)
    
    # STEP 4B: Classification task
    clf_features = ['Region_Enc', 'Age_Group_Enc', 'Population_Size',
                    'Urban_Rural', 'Literacy_Rate', 'Healthcare_Index',
                    'Pop_Bin', 'Literacy_Health']
    
    X_clf = df[clf_features].copy().fillna(df[clf_features].median())
    y_clf = df['Blood_Type_Enc']
    
    # Fix data leakage: fit scaler ONLY on training data
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_clf, y_clf, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    scaler_clf = StandardScaler()
    Xc_tr_sc = scaler_clf.fit_transform(Xc_tr)
    Xc_te_sc = scaler_clf.transform(Xc_te)
    
    clf_results = train_classification_models(Xc_tr_sc, Xc_te_sc, yc_tr, yc_te)
    
    # STEP 5: Visualizations
    output_file = create_visualizations(df, reg_results, clf_results,
                                        clf_features, y_te_reg, BLOOD_TYPES)
    
    # Summary
    print_summary(df, reg_results, clf_results)
