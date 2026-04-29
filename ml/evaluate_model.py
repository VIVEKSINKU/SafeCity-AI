"""
SafeCity AI — Comprehensive Model Evaluation
Generates detailed metrics, confusion matrices, ROC curves,
precision-recall curves, and feature importance analysis.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joblib
import os
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, learning_curve)
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              f1_score, precision_score, recall_score,
                              log_loss, cohen_kappa_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, '..', 'data', 'unified_crime_data.csv')
XGB_PATH  = os.path.join(BASE, 'xgb_model.pkl')
RF_PATH   = os.path.join(BASE, 'rf_model.pkl')
ENC_PATH  = os.path.join(BASE, 'label_encoder.pkl')
OUT_DIR   = os.path.join(BASE, '..')

CITY_LAT, CITY_LNG = 31.3260, 75.5762

FEATURES = [
    'Latitude','Longitude','Hour','DayOfWeek','Month',
    'Is_Weekend','Is_Night','Is_Evening','Is_Morning',
    'Hour_sin','Hour_cos','Month_sin','Month_cos',
    'Day_sin','Day_cos','Dist_Centre','Lat_Zone','Lng_Zone',
]

# ── Colour palette (dark theme) ──────────────────────────────────────────────
BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
ACCENT1 = '#58a6ff'   # XGBoost
ACCENT2 = '#3fb950'   # Random Forest
ACCENT3 = '#f78166'   # highlights
TEXT    = '#e6edf3'
SUBTEXT = '#8b949e'
COLORS_9 = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#79c0ff',
            '#ffa657','#ff7b72','#7ee787','#a5d6ff']

def style():
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': PANEL,
        'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT,
        'axes.titlecolor': TEXT, 'xtick.color': SUBTEXT,
        'ytick.color': SUBTEXT, 'text.color': TEXT,
        'grid.color': BORDER, 'grid.linestyle': '--', 'grid.alpha': 0.5,
        'font.family': 'DejaVu Sans',
        'legend.facecolor': PANEL, 'legend.edgecolor': BORDER,
        'legend.labelcolor': TEXT,
    })

# ── Feature engineering ──────────────────────────────────────────────────────
def engineer(df):
    df = df.copy()
    df['Is_Weekend']  = (df['DayOfWeek'] >= 5).astype(int)
    df['Is_Night']    = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
    df['Is_Evening']  = ((df['Hour'] >= 18) & (df['Hour'] < 22)).astype(int)
    df['Is_Morning']  = ((df['Hour'] >= 6)  & (df['Hour'] < 12)).astype(int)
    df['Hour_sin']    = np.sin(2*np.pi*df['Hour']      /24)
    df['Hour_cos']    = np.cos(2*np.pi*df['Hour']      /24)
    df['Month_sin']   = np.sin(2*np.pi*df['Month']     /12)
    df['Month_cos']   = np.cos(2*np.pi*df['Month']     /12)
    df['Day_sin']     = np.sin(2*np.pi*df['DayOfWeek'] /7)
    df['Day_cos']     = np.cos(2*np.pi*df['DayOfWeek'] /7)
    df['Dist_Centre'] = np.sqrt((df['Latitude']-CITY_LAT)**2+(df['Longitude']-CITY_LNG)**2)
    df['Lat_Zone']    = pd.cut(df['Latitude'],  bins=6, labels=False).fillna(0).astype(int)
    df['Lng_Zone']    = pd.cut(df['Longitude'], bins=6, labels=False).fillna(0).astype(int)
    return df

def augment(df, target=200):
    rng = np.random.RandomState(42)
    parts = [df]
    for ct, grp in df.groupby('Crime_Type'):
        needed = max(0, target - len(grp))
        if needed == 0: continue
        s = grp.sample(needed, replace=True, random_state=42).copy()
        s['Latitude']  += rng.normal(0, 0.003, needed)
        s['Longitude'] += rng.normal(0, 0.003, needed)
        s['Hour']       = (s['Hour'] + rng.randint(-1,2,needed)) % 24
        parts.append(s)
    return pd.concat(parts, ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD AND PREPARE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SAFECITY AI  --  COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)

df_raw = pd.read_csv(DATA_PATH)
df     = augment(df_raw)
df     = engineer(df)

le  = joblib.load(ENC_PATH)
xgb = joblib.load(XGB_PATH)
rf  = joblib.load(RF_PATH)

classes   = le.classes_
n_classes = len(classes)

X    = df[FEATURES].fillna(0)
y    = le.transform(df['Crime_Type'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for the training data used in some evaluations
min_cnt = pd.Series(y_train).value_counts().min()
k = max(1, min(5, min_cnt-1))
sm = SMOTE(random_state=42, k_neighbors=k)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Predictions
xgb_pred  = xgb.predict(X_test)
rf_pred   = rf.predict(X_test)
xgb_prob  = xgb.predict_proba(X_test)
rf_prob   = rf.predict_proba(X_test)

# Binarize for ROC/PR curves
y_test_bin = label_binarize(y_test, classes=range(n_classes))

# ══════════════════════════════════════════════════════════════════════════════
#  DETAILED METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(name, y_true, y_pred, y_prob):
    acc   = accuracy_score(y_true, y_pred)
    f1_ma = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_we = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec   = recall_score(y_true, y_pred, average='macro', zero_division=0)
    auc   = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    ll    = log_loss(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    return {
        'Model': name, 'Accuracy': acc, 'Macro F1': f1_ma,
        'Weighted F1': f1_we, 'Macro Precision': prec,
        'Macro Recall': rec, 'ROC-AUC': auc, 'Log Loss': ll,
        'Cohen Kappa': kappa, 'MCC': mcc,
    }

xgb_m = compute_metrics('XGBoost',       y_test, xgb_pred, xgb_prob)
rf_m  = compute_metrics('Random Forest',  y_test, rf_pred,  rf_prob)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv = cross_val_score(xgb, X_res, y_res, cv=cv, scoring='accuracy', n_jobs=-1)
rf_cv  = cross_val_score(rf,  X_res, y_res, cv=cv, scoring='accuracy', n_jobs=-1)

baseline_acc = max(pd.Series(y_test).value_counts()) / len(y_test)

# ── Print console report ─────────────────────────────────────────────────────
print(f"\nDataset: {len(df_raw)} raw -> {len(df)} augmented -> {len(X_test)} test samples")
print(f"Classes: {n_classes} | Features: {len(FEATURES)}")
print(f"Baseline (majority class): {baseline_acc:.2%}\n")

metrics_df = pd.DataFrame([xgb_m, rf_m]).set_index('Model').T
print(metrics_df.to_string(float_format='{:.4f}'.format))

print(f"\n5-Fold CV Accuracy:")
print(f"  XGBoost       : {xgb_cv.mean():.2%} (+/- {xgb_cv.std():.4f})")
print(f"  Random Forest : {rf_cv.mean():.2%} (+/- {rf_cv.std():.4f})")

print(f"\n--- XGBoost Classification Report ---")
print(classification_report(y_test, xgb_pred, target_names=classes, zero_division=0))

print(f"\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_pred, target_names=classes, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS (8-panel figure)
# ══════════════════════════════════════════════════════════════════════════════
style()

fig = plt.figure(figsize=(28, 32), facecolor=BG)
fig.suptitle('SafeCity AI  --  Comprehensive Model Evaluation',
             fontsize=24, fontweight='bold', color=TEXT, y=0.995)

gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30,
              left=0.06, right=0.97, top=0.97, bottom=0.03)

# ── Panel 1: Overall Metrics Bar Chart ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
metric_names = ['Accuracy','Macro F1','Weighted F1','Macro Precision','Macro Recall','ROC-AUC']
xgb_vals = [xgb_m[m] for m in metric_names]
rf_vals  = [rf_m[m]  for m in metric_names]
x = np.arange(len(metric_names))
w = 0.35
b1 = ax1.bar(x - w/2, xgb_vals, w, label='XGBoost',      color=ACCENT1, alpha=0.9, zorder=3)
b2 = ax1.bar(x + w/2, rf_vals,  w, label='Random Forest', color=ACCENT2, alpha=0.9, zorder=3)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.1%}',
                 ha='center', va='bottom', fontsize=7.5, color=TEXT, fontweight='bold')
ax1.axhline(baseline_acc, color=ACCENT3, ls='--', lw=1.5, alpha=0.7, label=f'Baseline ({baseline_acc:.1%})')
ax1.set_xticks(x); ax1.set_xticklabels(metric_names, fontsize=9)
ax1.set_ylim(0, 1.12); ax1.set_ylabel('Score')
ax1.set_title('1. Overall Performance Metrics', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=9); ax1.grid(axis='y', zorder=0)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

# ── Panel 2: Advanced Metrics (Kappa, MCC, LogLoss) ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
adv_names = ['Cohen Kappa', 'MCC', 'Log Loss']
xgb_adv = [xgb_m[m] for m in adv_names]
rf_adv  = [rf_m[m]  for m in adv_names]
x2 = np.arange(len(adv_names))
b1 = ax2.bar(x2 - w/2, xgb_adv, w, label='XGBoost',      color=ACCENT1, alpha=0.9, zorder=3)
b2 = ax2.bar(x2 + w/2, rf_adv,  w, label='Random Forest', color=ACCENT2, alpha=0.9, zorder=3)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2, h + (0.01 if h >= 0 else -0.03),
                 f'{h:.3f}', ha='center', va='bottom', fontsize=9, color=TEXT, fontweight='bold')
ax2.set_xticks(x2); ax2.set_xticklabels(adv_names, fontsize=10)
ax2.set_title('2. Advanced Metrics (Kappa, MCC, Log Loss)', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=9); ax2.grid(axis='y', zorder=0)
ax2.set_ylabel('Score')

# ── Panel 3: Per-Class F1 Comparison ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
xgb_rep = classification_report(y_test, xgb_pred, target_names=classes, output_dict=True, zero_division=0)
rf_rep  = classification_report(y_test, rf_pred,  target_names=classes, output_dict=True, zero_division=0)
xgb_f1 = [xgb_rep[c]['f1-score'] for c in classes]
rf_f1  = [rf_rep[c]['f1-score']  for c in classes]
x3 = np.arange(n_classes)
ax3.bar(x3 - 0.2, xgb_f1, 0.38, label='XGBoost',      color=ACCENT1, alpha=0.9, zorder=3)
ax3.bar(x3 + 0.2, rf_f1,  0.38, label='Random Forest', color=ACCENT2, alpha=0.9, zorder=3)
ax3.set_xticks(x3); ax3.set_xticklabels(classes, rotation=30, ha='right', fontsize=8)
ax3.set_ylim(0, 1.15); ax3.set_ylabel('F1-Score')
ax3.set_title('3. Per-Class F1-Score Comparison', fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=9); ax3.grid(axis='y', zorder=0)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

# ── Panel 4: Confusion Matrices (side by side) ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
cm_xgb  = confusion_matrix(y_test, xgb_pred)
cm_rf   = confusion_matrix(y_test, rf_pred)
cm_xgb_n = cm_xgb.astype(float) / cm_xgb.sum(axis=1, keepdims=True)
cm_rf_n  = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True)

# Show RF confusion matrix (winner)
im = ax4.imshow(cm_rf_n, cmap='Greens', vmin=0, vmax=1, aspect='auto')
ax4.set_xticks(range(n_classes)); ax4.set_yticks(range(n_classes))
ax4.set_xticklabels(classes, rotation=35, ha='right', fontsize=7.5)
ax4.set_yticklabels(classes, fontsize=7.5)
for i in range(n_classes):
    for j in range(n_classes):
        val = cm_rf_n[i, j]
        ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                 color='white' if val > 0.5 else TEXT, fontsize=7, fontweight='bold')
plt.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
ax4.set_xlabel('Predicted', fontsize=10); ax4.set_ylabel('True', fontsize=10)
ax4.set_title('4. Random Forest - Normalised Confusion Matrix', fontsize=13, fontweight='bold', pad=10)

# ── Panel 5: ROC Curves (per class, RF) ──────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_prob[:, i])
    auc_val = roc_auc_score(y_test_bin[:, i], rf_prob[:, i])
    ax5.plot(fpr, tpr, color=COLORS_9[i], lw=1.8, label=f'{cls} (AUC={auc_val:.3f})')
ax5.plot([0,1], [0,1], 'w--', lw=1, alpha=0.3)
ax5.set_xlabel('False Positive Rate', fontsize=10)
ax5.set_ylabel('True Positive Rate', fontsize=10)
ax5.set_title('5. ROC Curves - Random Forest (per class)', fontsize=13, fontweight='bold', pad=10)
ax5.legend(fontsize=7, loc='lower right'); ax5.grid(zorder=0)

# ── Panel 6: Precision-Recall Curves (RF) ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
for i, cls in enumerate(classes):
    prec_arr, rec_arr, _ = precision_recall_curve(y_test_bin[:, i], rf_prob[:, i])
    ap = average_precision_score(y_test_bin[:, i], rf_prob[:, i])
    ax6.plot(rec_arr, prec_arr, color=COLORS_9[i], lw=1.8, label=f'{cls} (AP={ap:.3f})')
ax6.set_xlabel('Recall', fontsize=10)
ax6.set_ylabel('Precision', fontsize=10)
ax6.set_title('6. Precision-Recall Curves - Random Forest', fontsize=13, fontweight='bold', pad=10)
ax6.legend(fontsize=7, loc='lower left'); ax6.grid(zorder=0)

# ── Panel 7: Cross-Validation Stability ──────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
folds = [f'Fold {i+1}' for i in range(5)]
ax7.plot(folds, xgb_cv, 'o-', color=ACCENT1, lw=2.5, ms=8, label=f'XGBoost (mean={xgb_cv.mean():.2%})', zorder=3)
ax7.plot(folds, rf_cv,  's-', color=ACCENT2, lw=2.5, ms=8, label=f'RF (mean={rf_cv.mean():.2%})', zorder=3)
ax7.axhline(xgb_cv.mean(), color=ACCENT1, ls=':', lw=1.5, alpha=0.5)
ax7.axhline(rf_cv.mean(),  color=ACCENT2, ls=':', lw=1.5, alpha=0.5)
ax7.fill_between(folds, xgb_cv.mean()-xgb_cv.std(), xgb_cv.mean()+xgb_cv.std(),
                 alpha=0.15, color=ACCENT1)
ax7.fill_between(folds, rf_cv.mean()-rf_cv.std(), rf_cv.mean()+rf_cv.std(),
                 alpha=0.15, color=ACCENT2)
ax7.set_ylabel('Accuracy', fontsize=10)
ax7.set_title('7. 5-Fold Cross-Validation Stability', fontsize=13, fontweight='bold', pad=10)
ax7.legend(fontsize=9); ax7.grid(zorder=0)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))

# ── Panel 8: Feature Importance (XGB + RF side by side) ──────────────────────
ax8 = fig.add_subplot(gs[3, 1])
xgb_fi = dict(zip(FEATURES, xgb.feature_importances_))
rf_fi  = dict(zip(FEATURES, rf.feature_importances_))
# Sort by RF importance
sorted_feats = sorted(FEATURES, key=lambda f: rf_fi[f], reverse=True)[:12]
y_pos = np.arange(len(sorted_feats))
ax8.barh(y_pos - 0.18, [xgb_fi[f] for f in sorted_feats], 0.35,
         label='XGBoost', color=ACCENT1, alpha=0.9, zorder=3)
ax8.barh(y_pos + 0.18, [rf_fi[f] for f in sorted_feats], 0.35,
         label='Random Forest', color=ACCENT2, alpha=0.9, zorder=3)
ax8.set_yticks(y_pos); ax8.set_yticklabels(sorted_feats, fontsize=9)
ax8.invert_yaxis()
ax8.set_title('8. Feature Importance Comparison (Top 12)', fontsize=13, fontweight='bold', pad=10)
ax8.set_xlabel('Importance', fontsize=10)
ax8.legend(fontsize=9); ax8.grid(axis='x', zorder=0)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(OUT_DIR, 'model_evaluation.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nEvaluation graph saved: {out}")
print("Done.")
