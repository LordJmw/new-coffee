# utils/visualization.py
"""
Visualisasi tambahan untuk Tab "Penilaian Model" di app.py.

CATATAN PENTING soal create_decision_boundary_plot():
Random Forest produksi dilatih di ruang fitur ~90 dimensi (ALL_FEATURE_NAMES
dari utils/features_combine.py), jadi decision boundary-nya sendiri tidak
bisa digambar langsung dalam 2D. Pendekatan yang dipakai di sini adalah
teknik umum untuk memvisualisasikan model non-linear di ruang fitur tinggi:

    1. Reduksi fitur ASLI ke 2D lewat PCA (2 komponen varians terbesar).
    2. Latih SURROGATE RandomForest baru khusus di ruang PCA 2D tsb - ini
       BUKAN model produksi, hanya aproksimasi bentuk keputusan model asli
       pada proyeksi 2D, murni untuk keperluan visualisasi/interpretasi.
    3. Gambar contourf region keputusan surrogate + scatter titik data asli
       (warna = label sebenarnya) di atasnya.

Karena surrogate dilatih ulang di ruang 2D (bukan proyeksi dari model asli),
batas yang tergambar adalah APROKSIMASI kasar, bukan representasi eksak dari
RF 90-dimensi yang sesungguhnya dipakai untuk prediksi. Fungsi ini murni
alat bantu eksplorasi visual, bukan bukti akurasi model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3",
]


def create_decision_boundary_plot(X, y, feature_names=None, classes=None,
                                   random_state=42, mesh_step=0.02,
                                   n_estimators=200, figsize=(8, 6)):
    """
    Bangun matplotlib Figure berisi decision boundary surrogate 2D (lihat
    catatan modul di atas) untuk sekumpulan sampel fitur + label kelas.

    Parameters
    ----------
    X : array (n_samples, n_features) - fitur ASLI, boleh langsung fitur
        ~90 dimensi dari CSV/model (direduksi PCA secara internal).
    y : array (n_samples,) - label kelas final (string), dipakai untuk
        warna titik & melatih surrogate.
    feature_names : opsional, tidak dipakai untuk plotting - disediakan
        supaya pemanggil boleh lewatkan nama kolom tanpa error (mis. untuk
        anotasi/logging di luar fungsi ini).
    classes : opsional, urutan kelas untuk warna & legenda. Default:
        urutan terurut (sorted) dari nilai unik y.

    Returns
    -------
    fig : matplotlib.figure.Figure - siap dipakai lewat st.pyplot(fig).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if classes is None:
        classes = sorted(set(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)

    surrogate = RandomForestClassifier(
        n_estimators=n_estimators, class_weight='balanced',
        random_state=random_state, n_jobs=-1
    )
    surrogate.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, max(mesh_step * (x_max - x_min), 1e-6)),
        np.arange(y_min, y_max, max(mesh_step * (y_max - y_min), 1e-6)),
    )

    grid_pred = surrogate.predict(np.c_[xx.ravel(), yy.ravel()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    grid_idx = np.array([class_to_idx.get(c, -1) for c in grid_pred]).reshape(xx.shape)

    cmap_bg = ListedColormap([_PALETTE[i % len(_PALETTE)] for i in range(len(classes))])

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, grid_idx, alpha=0.25, cmap=cmap_bg,
                levels=np.arange(-0.5, len(classes), 1))

    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=c, s=25, alpha=0.85,
            color=_PALETTE[i % len(_PALETTE)],
            edgecolors='black', linewidths=0.3,
        )

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% varians)")
    ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% varians)")
    ax.set_title("Decision Boundary (Surrogate RF, Proyeksi PCA 2D)")
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    fig.tight_layout()

    return fig