"""
STEP 2 (Tahap 2): Training & Evaluasi Baseline Random Forest - 7 Kelas.

PERBAIKAN PENTING vs versi sebelumnya (analisis gap CV 0.98 vs test 0.76):
  Root cause gap tersebut adalah SMOTE diterapkan pada train set SEBELUM
  cross-validation dilakukan -> setiap fold CV berisi campuran data
  asli + sintetis yang saling "bocor" (titik sintetis dibuat dari
  tetangga yang bisa saja jatuh di fold validasi), sehingga skor CV
  jadi optimis palsu. Fold CV mengevaluasi model pada data yang mirip
  sekali dengan data latihnya sendiri (interpolasi SMOTE), bukan pada
  data benar-benar belum pernah dilihat.

  Perbaikan: SMOTE dipindah ke DALAM Pipeline (imblearn.pipeline.Pipeline)
  yang dipasangkan ke StratifiedKFold lewat cross_validate. Dengan begitu,
  di setiap fold, SMOTE HANYA dipasang (fit_resample) pada bagian train
  fold tersebut - fold validasi selalu murni data asli yang tidak pernah
  disentuh SMOTE. Ini membuat skor CV jadi estimator yang jujur/sebanding
  dengan skor test set hold-out.

Implementasi tetap mengikuti Bagian 4 & 10, "Ringkasan Arah Skripsi Kopi":
  - Undersampling kelas Normal (rasio awal ~34:1 terhadap kelas cacat
    terkecil) - dilakukan SEBELUM split train/test & CV karena undersampling
    (membuang baris) tidak menimbulkan leakage seperti SMOTE (tidak
    membuat titik baru dari kombinasi informasi lintas-fold).
  - class_weight='balanced' pada Random Forest.
  - SMOTE pada level fitur hasil ekstraksi (bukan raw pixel) - HANYA di
    dalam pipeline training, tidak pernah menyentuh data validasi/test.
  - Evaluasi utama pakai macro-F1 (bukan accuracy semata).
  - Stratified split & stratified CV supaya proporsi 7 kelas terjaga.

Skrip ini adalah baseline Stage 2 (Jalur 2 pada diagram alur data) yang
dijalankan langsung di atas fitur hasil extract_features.py (crop bersih
Aset A) - BUKAN pipeline few-shot/self-training penuh, yang menyusul
setelah Stage 1 (YOLO) tersedia.

Cara pakai:
    python train_evaluate_rf.py --csv data/features_dataset_7kelas.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.utils import resample

from utils.label_mapping import FINAL_CLASSES

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ─────────────────────────────────────────────────────────────────────────
# Penanganan imbalance
# ─────────────────────────────────────────────────────────────────────────

def undersample_majority(df, label_col='class_name', majority_class='Normal',
                          target_size=1200, random_state=42):
    """Turunkan jumlah sampel kelas mayoritas (Normal) ke target_size (Bagian 4)."""
    is_major = df[label_col] == majority_class
    df_major = df[is_major]
    df_minor = df[~is_major]

    if len(df_major) > target_size:
        df_major = resample(df_major, replace=False, n_samples=target_size,
                             random_state=random_state)

    return pd.concat([df_major, df_minor], axis=0).reset_index(drop=True)


def make_smote_rf_pipeline(y_train, random_state=42, n_estimators=300,
                            max_depth=None, min_samples_leaf=1):
    """
    Bangun Pipeline SMOTE -> RandomForest. SMOTE HANYA dieksekusi pada
    data yang di-fit ke pipeline ini (mis. train fold di dalam CV, atau
    seluruh train set untuk model final) - tidak pernah pada data yang
    hanya di-predict (validasi/test), karena itu perilaku standar
    imblearn.pipeline.Pipeline (berbeda dari sklearn.pipeline.Pipeline
    biasa yang tidak mendukung resampler).
    """
    if not HAS_SMOTE:
        print("[WARNING] imbalanced-learn belum terpasang (`pip install imbalanced-learn`)."
              " Melanjutkan TANPA SMOTE, hanya class_weight='balanced'.")
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight='balanced',
            random_state=random_state, n_jobs=-1
        )

    counts = pd.Series(y_train).value_counts()
    min_count = counts.min()
    k_neighbors = max(1, min(5, min_count - 1))
    if min_count <= 1:
        print("[WARNING] Ada kelas dengan <=1 sampel, SMOTE dilewati untuk pipeline ini.")
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight='balanced',
            random_state=random_state, n_jobs=-1
        )

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    # class_weight='balanced' TETAP dipasang meski sudah di-SMOTE: SMOTE
    # menyamakan JUMLAH sampel, tapi menjaga balanced weight memberi margin
    # ekstra agar kelas minoritas asli (sebelum oversampling) tidak
    # tenggelam oleh noise sintetis saat splitting pohon.
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight='balanced',
        random_state=random_state, n_jobs=-1
    )
    return ImbPipeline([('smote', smote), ('rf', clf)])


# ─────────────────────────────────────────────────────────────────────────
# Training & evaluasi
# ─────────────────────────────────────────────────────────────────────────

def get_feature_columns(df):
    non_feature = {'filename', 'subclass_name', 'class_name', 'class_num', 'is_valid'}
    return [c for c in df.columns if c not in non_feature and not c.startswith('_')]


def train_and_evaluate(csv_path, output_dir='outputs', undersample_target=1200,
                        use_smote=True, n_splits=5, random_state=42,
                        n_estimators=300, max_depth=18, min_samples_leaf=2):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df['is_valid'] == True].reset_index(drop=True)  # noqa: E712
    print(f"Total sampel valid: {len(df)}")
    print("\nDistribusi kelas awal:")
    print(df['class_name'].value_counts())

    # ── Undersampling kelas Normal (aman dilakukan sebelum split: hanya
    #    membuang baris, tidak membuat titik baru dari info lintas-fold) ──
    df_balanced = undersample_majority(df, target_size=undersample_target,
                                        random_state=random_state)
    print(f"\nDistribusi kelas setelah undersampling Normal (target={undersample_target}):")
    print(df_balanced['class_name'].value_counts())

    feature_cols = get_feature_columns(df_balanced)
    X = df_balanced[feature_cols].values
    y = df_balanced['class_name'].values

    # ── Stratified train/test split (test set = evaluasi akhir, TIDAK PERNAH
    #    disentuh SMOTE atau undersampling tambahan apa pun) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    print(f"\nUkuran train set (sebelum SMOTE): {len(X_train)}")
    print(f"Ukuran test set (hold-out murni): {len(X_test)}")

    # ── Stratified K-Fold CV YANG BENAR: SMOTE di dalam pipeline, fit HANYA
    #    pada train-fold; validasi-fold selalu data asli. Ini memperbaiki
    #    gap CV-vs-test yang ditemukan sebelumnya (CV 0.98 vs test 0.76). ──
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_macro_f1, cv_acc = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        pipe = make_smote_rf_pipeline(
            y_train[tr_idx], random_state=random_state, n_estimators=n_estimators,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        ) if use_smote else RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight='balanced',
            random_state=random_state, n_jobs=-1
        )
        pipe.fit(X_train[tr_idx], y_train[tr_idx])
        pred = pipe.predict(X_train[val_idx])   # val_idx = data ASLI, tidak di-SMOTE
        fold_f1 = f1_score(y_train[val_idx], pred, average='macro')
        fold_acc = accuracy_score(y_train[val_idx], pred)
        cv_macro_f1.append(fold_f1)
        cv_acc.append(fold_acc)
        print(f"  Fold {fold}: macro-F1 = {fold_f1:.4f} | accuracy = {fold_acc:.4f}")
    print(f"CV macro-F1 rata-rata (leak-free): {np.mean(cv_macro_f1):.4f} "
          f"(+/- {np.std(cv_macro_f1):.4f})")
    print(f"CV accuracy rata-rata (leak-free): {np.mean(cv_acc):.4f} (+/- {np.std(cv_acc):.4f})")
    print("-> Bandingkan dengan skor test set di bawah: seharusnya sekarang jauh lebih")
    print("   dekat (gap kecil) dibanding versi lama yang SMOTE-nya bocor ke CV.")

    # ── Model final: pipeline SMOTE+RF di-fit ke SELURUH train (SMOTE di
    #    sini tidak masalah karena tidak ada "fold lain" yang bisa bocor -
    #    leakage CV hanya relevan saat mengestimasi performa, bukan saat
    #    melatih model final), dievaluasi di test set asli murni. ──
    final_pipe = make_smote_rf_pipeline(
        y_train, random_state=random_state, n_estimators=n_estimators,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf
    ) if use_smote else RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight='balanced',
        random_state=random_state, n_jobs=-1
    )
    final_pipe.fit(X_train, y_train)
    final_clf = final_pipe.named_steps['rf'] if hasattr(final_pipe, 'named_steps') else final_pipe
    y_pred = final_pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n{'=' * 60}\nHASIL EVALUASI TEST SET (hold-out)\n{'=' * 60}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro-F1   : {macro_f1:.4f}")

    labels_order = [c for c in FINAL_CLASSES if c in set(y_test) | set(y_pred)]
    report = classification_report(y_test, y_pred, labels=labels_order, digits=3, zero_division=0)
    print("\nClassification report per kelas:")
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}")
    print(f"Classification report disimpan di: {report_path}")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order,
                yticklabels=labels_order)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix - Random Forest (7 Kelas)')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix disimpan di: {cm_path}")

    # ── Feature importance ──
    importances = pd.Series(final_clf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(8, 10))
    importances.head(25).iloc[::-1].plot(kind='barh')
    plt.title('Top 25 Feature Importance - Random Forest')
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Feature importance disimpan di: {fi_path}")

    importances.to_csv(os.path.join(output_dir, "feature_importance.csv"))

    # ── Simpan model untuk dipakai app.py (WAJIB - sebelumnya terlewat,
    #    menyebabkan app.py tidak pernah menemukan file model & selalu
    #    fallback ke Rule-based saja). PENTING: yang disimpan adalah
    #    final_clf (RandomForest murni), BUKAN final_pipe - karena SMOTE
    #    hanya boleh dijalankan saat training, tidak pernah saat inference. ──
    model_path = os.path.join(output_dir, "coffee_bean_rf_7class.pkl")
    bundle = {
        'model': final_clf,
        'feature_columns': feature_cols,
        'classes': list(final_clf.classes_),
    }
    joblib.dump(bundle, model_path)
    print(f"\nModel disimpan di: {model_path}")
    print("-> app.py akan otomatis mendeteksi file ini dan mengaktifkan mode Machine Learning.")

    return {
        'model': final_clf,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'cv_macro_f1_mean': float(np.mean(cv_macro_f1)),
        'cv_macro_f1_std': float(np.std(cv_macro_f1)),
        'feature_columns': feature_cols,
        'model_path': model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Training & evaluasi Random Forest 7 kelas")
    parser.add_argument('--csv', type=str, default='data/features_dataset_7kelas.csv')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--undersample-target', type=int, default=1200,
                         help="Jumlah maksimum sampel kelas Normal setelah undersampling")
    parser.add_argument('--no-smote', action='store_true', help="Nonaktifkan SMOTE")
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--max-depth', type=int, default=18,
                         help="Batasi kedalaman pohon agar tidak overfit ke noise SMOTE")
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] File CSV tidak ditemukan: {args.csv}")
        print("Jalankan extract_features.py terlebih dahulu.")
        return

    train_and_evaluate(
        csv_path=args.csv,
        output_dir=args.output_dir,
        undersample_target=args.undersample_target,
        use_smote=not args.no_smote,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )


if __name__ == "__main__":
    main()