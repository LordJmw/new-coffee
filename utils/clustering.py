# utils/clustering.py
"""
Segmentasi warna K-Means pada citra biji yang sudah di-crop (ROI hasil
bounding box kontur utama). Dipakai Tab "Analisis Warna" di app.py untuk:
  1. Visualisasi warna dominan biji (kmeans_img).
  2. Statistik per-klaster (proporsi area + warna RGB) yang juga jadi
     dasar logika rule-based deteksi SOUR (rasio R/G klaster dominan).
"""

import numpy as np
from sklearn.cluster import KMeans


def analyze_color_kmeans(cropped_rgb, k=3, random_state=42):
    """
    Klaster piksel citra crop biji ke k warna dominan.

    Parameters
    ----------
    cropped_rgb : array RGB hasil crop bounding box biji (H x W x 3).
    k : jumlah klaster warna yang diinginkan (bisa dikurangi otomatis
        kalau jumlah piksel lebih sedikit dari k).

    Returns
    -------
    kmeans_img : array RGB (ukuran sama dengan cropped_rgb, uint8) - tiap
        piksel diganti warna centroid klasternya (visualisasi segmentasi).
    color_centers : array (k_eff, 3) uint8 - RGB tiap centroid, terurut
        dari klaster dengan area TERBESAR ke terkecil.
    cluster_stats : dict {0: {'color': (r, g, b), 'percentage': float}, ...}
        indeks & urutan sama dengan color_centers (0 = klaster dominan).

    Kalau cropped_rgb kosong/tidak valid, return (None, None, None) supaya
    pemanggil (app.py) bisa fallback ke pesan "klaster tidak muncul".
    """
    if cropped_rgb is None or cropped_rgb.size == 0:
        return None, None, None

    h, w = cropped_rgb.shape[:2]
    if h == 0 or w == 0:
        return None, None, None

    pixels = cropped_rgb.reshape(-1, 3).astype(np.float32)
    n_pixels = pixels.shape[0]
    k_eff = max(1, min(k, n_pixels))

    kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

    # Urutkan klaster dari yang paling dominan (persentase area terbesar)
    # supaya klaster 0 di UI selalu warna paling representatif dari biji.
    counts = np.bincount(labels, minlength=k_eff)
    order = np.argsort(-counts)  # descending berdasarkan jumlah piksel
    rank_of_old_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}

    color_centers = centers[order]
    cluster_stats = {}
    for new_idx, old_idx in enumerate(order):
        pct = float(counts[old_idx]) / n_pixels * 100.0
        r, g, b = (int(c) for c in centers[old_idx])
        cluster_stats[new_idx] = {'color': (r, g, b), 'percentage': pct}

    # Bangun citra tervisualisasi: tiap piksel -> warna centroid klasternya
    remapped_labels = np.array([rank_of_old_idx[old] for old in labels])
    kmeans_img = color_centers[remapped_labels].reshape(h, w, 3).astype(np.uint8)

    return kmeans_img, color_centers, cluster_stats