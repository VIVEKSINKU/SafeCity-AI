import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE

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

# ── colour palette ────────────────────────────────────────────────────────────
BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
ACCENT1 = '#58a6ff'   # blue  – XGBoost
ACCENT2 = '#3fb950'   # green – Random Forest
ACCENT3 = '#f78166'   # red   – Baseline
TEXT    = '#e6edf3'
SUBTEXT = '#8b949e'

def style():
    plt.rcParams.update({
        'figure.facecolor'  : BG,
        'axes.facecolor'    : PANEL,
        'axes.edgecolor'    : BORDER,
        'axes.labelcolor'   : TEXT,
        'axes.titlecolor'   : TEXT,
        'xtick.color'       : SUBTEXT,
        'ytick.color'       : SUBTEXT,
        'text.color'        : TEXT,
        'grid.color'        : BORDER,
        'grid.linestyle'    : '--',
        'grid.alpha'        : 0.5,
        'font.family'       : 'DejaVu Sans',
        'legend.facecolor'  : PANEL,
        'legend.edgecolor'  : BORDER,
        'legend.labelcolor' : TEXT,
    })

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

# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data and models ...")
df_raw = pd.read_csv(DATA_PATH)
df     = augment(df_raw)
df     = engineer(df)

le  = joblib.load(ENC_PATH)
xgb = joblib.load(XGB_PATH)
rf  = joblib.load(RF_PATH)

X    = df[FEATURES].fillna(0)
y    = le.transform(df['Crime_Type'])
classes = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

min_cnt = pd.Series(y_train).value_counts().min()
k = max(1, min(5, min_cnt-1))
sm = SMOTE(random_state=42, k_neighbors=k)
X_res, y_res = sm.fit_resample(X_train, y_train)

xgb_pred  = xgb.predict(X_test)
rf_pred   = rf.predict(X_test)
xgb_prob  = xgb.predict_proba(X_test)
rf_prob   = rf.predict_proba(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
rf_acc  = accuracy_score(y_test, rf_pred)

xgb_auc = roc_auc_score(y_test, xgb_prob,  multi_class='ovr', average='macro')
rf_auc  = roc_auc_score(y_test, rf_prob,   multi_class='ovr', average='macro')

xgb_rep = classification_report(y_test, xgb_pred, target_names=classes,
                                 output_dict=True, zero_division=0)
rf_rep  = classification_report(y_test, rf_pred,  target_names=classes,
                                 output_dict=True, zero_division=0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv = cross_val_score(xgb, X_res, y_res, cv=cv, scoring='accuracy', n_jobs=-1)
rf_cv  = cross_val_score(rf,  X_res, y_res, cv=cv, scoring='accuracy', n_jobs=-1)

# Baseline: majority class
baseline_acc = max(pd.Series(y_test).value_counts()) / len(y_test)

# ─── PRINT REPORT ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("              CRIME PREDICTION MODEL -- ACCURACY REPORT")
print("="*65)
print(f"\n{'Metric':<30} {'XGBoost':>12} {'Random Forest':>14}")
print("-"*60)
print(f"{'Test Accuracy':<30} {xgb_acc:>11.2%} {rf_acc:>13.2%}")
print(f"{'ROC-AUC (macro-OvR)':<30} {xgb_auc:>11.4f} {rf_auc:>13.4f}")
print(f"{'CV Accuracy (mean)':<30} {xgb_cv.mean():>11.2%} {rf_cv.mean():>13.2%}")
print(f"{'CV Accuracy (std)':<30} {xgb_cv.std():>11.4f} {rf_cv.std():>13.4f}")
print(f"{'Macro Precision':<30} {xgb_rep['macro avg']['precision']:>11.2%} {rf_rep['macro avg']['precision']:>13.2%}")
print(f"{'Macro Recall':<30} {xgb_rep['macro avg']['recall']:>11.2%} {rf_rep['macro avg']['recall']:>13.2%}")
print(f"{'Macro F1-Score':<30} {xgb_rep['macro avg']['f1-score']:>11.2%} {rf_rep['macro avg']['f1-score']:>13.2%}")
print(f"{'Baseline (majority class)':<30} {baseline_acc:>11.2%} {'':>13}")
print(f"{'Training Samples (after aug)':<30} {len(X_res):>11,}")
print(f"{'Test Samples':<30} {len(X_test):>11,}")
print(f"{'Features Used':<30} {len(FEATURES):>11}")
print("-"*60)
print("\nPer-class Comparison (F1-Score):")
print(f"  {'Class':<20} {'XGBoost':>10} {'RF':>10}")
print(f"  {'-'*42}")
for c in classes:
    xf = xgb_rep[c]['f1-score']
    rf_f = rf_rep[c]['f1-score']
    winner = '<< RF' if rf_f > xf else ('<< XGB' if xf > rf_f else '')
    print(f"  {c:<20} {xf:>9.2%} {rf_f:>10.2%}  {winner}")
print("="*65)

# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
style()

fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.suptitle('Crime Prediction Model — Accuracy Report & Comparison',
             fontsize=20, fontweight='bold', color=TEXT, y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
              left=0.06, right=0.97, top=0.93, bottom=0.05)

# ── 1. Overall metrics bar chart ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
metrics      = ['Test Accuracy', 'CV Accuracy', 'Macro F1', 'ROC-AUC']
xgb_vals     = [xgb_acc, xgb_cv.mean(), xgb_rep['macro avg']['f1-score'], xgb_auc]
rf_vals      = [rf_acc,  rf_cv.mean(),  rf_rep['macro avg']['f1-score'],  rf_auc]
baseline_val = [baseline_acc, baseline_acc, baseline_acc, 0.5]

x     = np.arange(len(metrics))
width = 0.28
bars1 = ax1.bar(x - width, xgb_vals,      width, label='XGBoost',      color=ACCENT1, alpha=0.9, zorder=3)
bars2 = ax1.bar(x,         rf_vals,        width, label='Random Forest', color=ACCENT2, alpha=0.9, zorder=3)
bars3 = ax1.bar(x + width, baseline_val,  width, label='Baseline',      color=ACCENT3, alpha=0.6, zorder=3)

for bar in list(bars1)+list(bars2)+list(bars3):
    h = bar.get_height()
    ax1.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.0%}',
             ha='center', va='bottom', fontsize=8, color=TEXT, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.set_ylim(0, 1.12)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
ax1.grid(axis='y', zorder=0)
ax1.set_facecolor(PANEL)

# ── 2. Score gauge / big numbers ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
ax2.set_xlim(0,1); ax2.set_ylim(0,1)
ax2.axis('off')
ax2.set_title('Headline Scores', fontsize=13, fontweight='bold', pad=10)

def draw_gauge(ax, cx, cy, radius, value, color, label, sub):
    theta = np.linspace(np.pi, np.pi*(1 - value), 100)
    ax.plot(np.cos(theta)*radius+cx, np.sin(theta)*radius+cy,
            color=color, lw=10, solid_capstyle='round')
    ax.plot(np.cos(np.linspace(np.pi,0,100))*radius+cx,
            np.sin(np.linspace(np.pi,0,100))*radius+cy,
            color=BORDER, lw=10, solid_capstyle='round', zorder=1)
    ax.plot(np.cos(np.linspace(np.pi,0,100))*radius+cx,
            np.sin(np.linspace(np.pi,0,100))*radius+cy,
            color=PANEL,  lw=8,  solid_capstyle='round', zorder=0)
    ax.plot(np.cos(np.linspace(np.pi, np.pi*(1-value),100))*radius+cx,
            np.sin(np.linspace(np.pi, np.pi*(1-value),100))*radius+cy,
            color=color, lw=8, solid_capstyle='round', zorder=2)
    ax.text(cx, cy+0.01, f'{value:.1%}', ha='center', va='center',
            fontsize=16, fontweight='bold', color=color)
    ax.text(cx, cy-0.12, label, ha='center', fontsize=9, color=TEXT, fontweight='bold')
    ax.text(cx, cy-0.22, sub,   ha='center', fontsize=7.5, color=SUBTEXT)

draw_gauge(ax2, 0.28, 0.62, 0.22, xgb_acc,  ACCENT1, 'XGBoost', 'Test Accuracy')
draw_gauge(ax2, 0.72, 0.62, 0.22, rf_acc,   ACCENT2, 'Random Forest', 'Test Accuracy')
draw_gauge(ax2, 0.28, 0.18, 0.15, xgb_auc,  ACCENT1, 'XGBoost AUC', 'ROC-AUC')
draw_gauge(ax2, 0.72, 0.18, 0.15, rf_auc,   ACCENT2, 'RF AUC', 'ROC-AUC')

# ── 3. Per-class F1 comparison ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
xgb_f1 = [xgb_rep[c]['f1-score'] for c in classes]
rf_f1  = [rf_rep[c]['f1-score']  for c in classes]
x3     = np.arange(len(classes))
ax3.bar(x3 - 0.2, xgb_f1, 0.38, label='XGBoost',      color=ACCENT1, alpha=0.9, zorder=3)
ax3.bar(x3 + 0.2, rf_f1,  0.38, label='Random Forest', color=ACCENT2, alpha=0.9, zorder=3)
ax3.set_xticks(x3)
ax3.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel('F1-Score', fontsize=11)
ax3.set_title('Per-Class F1-Score Comparison', fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
ax3.grid(axis='y', zorder=0)
ax3.set_facecolor(PANEL)

# ── 4. CV fold accuracy ────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
folds = [f'Fold {i+1}' for i in range(5)]
ax4.plot(folds, xgb_cv, 'o-', color=ACCENT1, lw=2.5, ms=7, label='XGBoost', zorder=3)
ax4.plot(folds, rf_cv,  's-', color=ACCENT2, lw=2.5, ms=7, label='RF', zorder=3)
ax4.axhline(xgb_cv.mean(), color=ACCENT1, ls=':', lw=1.5, alpha=0.7)
ax4.axhline(rf_cv.mean(),  color=ACCENT2, ls=':', lw=1.5, alpha=0.7)
ax4.fill_between(folds, xgb_cv, xgb_cv.mean(), alpha=0.12, color=ACCENT1)
ax4.fill_between(folds, rf_cv,  rf_cv.mean(),  alpha=0.12, color=ACCENT2)
ax4.set_title('5-Fold CV Accuracy', fontsize=13, fontweight='bold', pad=10)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
ax4.legend(fontsize=10)
ax4.grid(zorder=0)
ax4.set_facecolor(PANEL)

# ── 5. Confusion matrix – RF (best model) ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
cm = confusion_matrix(y_test, rf_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax5.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax5.set_xticks(range(len(classes)))
ax5.set_yticks(range(len(classes)))
ax5.set_xticklabels(classes, rotation=35, ha='right', fontsize=8)
ax5.set_yticklabels(classes, fontsize=8)
ax5.set_title('Random Forest — Normalised Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
ax5.set_xlabel('Predicted Label', fontsize=10)
ax5.set_ylabel('True Label', fontsize=10)
for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm_norm[i, j]
        ax5.text(j, i, f'{val:.2f}', ha='center', va='center',
                 color='white' if val > 0.5 else TEXT, fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax5, fraction=0.03, pad=0.02)

# ── 6. Feature importance – XGBoost ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
fi   = dict(zip(FEATURES, xgb.feature_importances_))
fi_s = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
names, vals = zip(*fi_s)
colors = [ACCENT1 if i == 0 else ACCENT2 if i < 3 else SUBTEXT for i in range(len(names))]
bars = ax6.barh(range(len(names)), vals, color=colors, alpha=0.9, zorder=3)
ax6.set_yticks(range(len(names)))
ax6.set_yticklabels(names, fontsize=9)
ax6.invert_yaxis()
ax6.set_title('XGBoost Feature Importance\n(Top 10)', fontsize=12, fontweight='bold', pad=8)
ax6.set_xlabel('Importance', fontsize=10)
ax6.grid(axis='x', zorder=0)
ax6.set_facecolor(PANEL)
for bar, val in zip(bars, vals):
    ax6.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8, color=TEXT)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'model_accuracy_report.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nGraph saved: {out_path}")
plt.close()
print("Done.")
