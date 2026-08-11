# utils/features_texture.py
"""
Fitur tekstur — bagian paling penting dari penyempurnaan ini.

Alasan: kelas versi lama yang stagnan (Severe Insect Damage, 78%) dan
kelas-kelas baru yang mirip secara bentuk/warna tapi beda tekstur permukaan
(Fungus Damage berbintik vs Husk berserat vs Shell licin vs Insect Damage
berlubang) TIDAK bisa dipisahkan hanya dari geometri kontur + rata-rata
warna. Dua deskriptor tekstur ditambahkan:

1. GLCM (Gray-Level Co-occurrence Matrix) — cv2/skimage.feature.graycomatrix
   pada 4 sudut (0,45,90,135) & 2 jarak (1,3 px), diambil rata-ratanya agar
   rotation-robust. Menghasilkan: contrast, homogeneity, energy (ASM),
   correlation, dissimilarity. Menangkap kekasaran & keteraturan pola
   permukaan biji secara global.

2. LBP (Local Binary Pattern, uniform) — skimage.feature.local_binary_pattern.
   LBP invarian terhadap perubahan monoton kecerahan (illumination-robust
   secara desain, bukan cuma dinormalisasi di preprocessing), cocok untuk
   mikro-tekstur seperti lubang kecil/bintik jamur yang kontrasnya lokal.
   Histogram LBP uniform (10 bin untuk P=8,R=1) dipakai sebagai fitur.

3. Kontrol tambahan: rasio kepadatan tepi (Canny edge density) di dalam
   region biji, menggantikan hitungan HoughLinesP mentah pada versi lama
   yang gampang berubah oleh noise -- dipakai sebagai proxy "center cut
   lines" / retakan permukaan yang lebih stabil.

REVISI (fokus Insect Damage & Immature/Discoloration):
- LBP dibuat MULTI-SKALA: (P=8, R=1) untuk mikro-tekstur sangat halus (titik
  masuk serangga sekecil beberapa piksel, retakan rambut halus akibat
  withered/immature), DITAMBAH (P=16, R=2) untuk pola tekstur skala lebih
  besar (kerutan withered, bintik fungus yang lebih lebar). Kelas Insect
  Damage & Immature seringkali hanya beda di skala tekstur ini, bukan di
  bentuk/warna global.
- GLCM ditambah properti 'ASM' (Angular Second Moment, komponen dasar
  Haralick) dan jarak piksel 1,2,4 (bukan cuma 1,3) agar kontras/keteraturan
  permukaan tertangkap di beberapa skala.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ['contrast', 'homogeneity', 'energy', 'ASM', 'correlation', 'dissimilarity']

# Skala halus: menangkap titik/lubang serangga kecil & retakan rambut halus
LBP_FINE_P, LBP_FINE_R = 8, 1
LBP_FINE_N_BINS = LBP_FINE_P + 2

# Skala sedang: menangkap kerutan withered & bintik fungus yang lebih besar
LBP_COARSE_P, LBP_COARSE_R = 16, 2
LBP_COARSE_N_BINS = LBP_COARSE_P + 2

TEXTURE_FEATURE_NAMES = (
    [f'glcm_{p}' for p in GLCM_PROPS]
    + [f'lbp_fine_bin{i}' for i in range(LBP_FINE_N_BINS)]
    + [f'lbp_coarse_bin{i}' for i in range(LBP_COARSE_N_BINS)]
    + ['edge_density']
)


def _crop_to_bbox(gray_img, mask, bbox, pad=2):
    x, y, w, h = bbox
    img_h, img_w = gray_img.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    return gray_img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def extract_texture_features(gray_img, bean_mask, bbox):
    """
    Ekstrak fitur tekstur dari region biji.
    gray_img: citra grayscale (idealnya sudah melalui normalisasi iluminasi
    di preprocessing, meski LBP & GLCM relatif tahan variasi kecerahan).
    bean_mask, bbox: hasil segmentasi dari features_shape.extract_shape_features.
    """
    empty = {name: 0.0 for name in TEXTURE_FEATURE_NAMES}
    if bean_mask is None or bbox is None:
        return empty

    crop_gray, crop_mask = _crop_to_bbox(gray_img, bean_mask, bbox)
    if crop_gray.size == 0 or not np.any(crop_mask == 255):
        return empty

    # Background di luar biji diisi nilai median biji supaya tidak
    # mendominasi statistik GLCM/LBP dengan kontras artifisial mask-vs-latar.
    bean_vals = crop_gray[crop_mask == 255]
    fill_val = int(np.median(bean_vals)) if bean_vals.size > 0 else 0
    masked_gray = crop_gray.copy()
    masked_gray[crop_mask != 255] = fill_val

    # ── GLCM ──
    glcm = graycomatrix(
        masked_gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=256, symmetric=True, normed=True
    )
    glcm_feats = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        glcm_feats[f'glcm_{prop}'] = float(np.mean(vals))

    # ── LBP multi-skala (uniform), histogram dihitung hanya di dalam mask biji ──
    lbp_fine = local_binary_pattern(masked_gray, P=LBP_FINE_P, R=LBP_FINE_R, method='uniform')
    lbp_fine_vals = lbp_fine[crop_mask == 255]
    hist_fine, _ = np.histogram(lbp_fine_vals, bins=np.arange(LBP_FINE_N_BINS + 1), density=True)
    lbp_feats = {f'lbp_fine_bin{i}': float(hist_fine[i]) for i in range(LBP_FINE_N_BINS)}

    lbp_coarse = local_binary_pattern(masked_gray, P=LBP_COARSE_P, R=LBP_COARSE_R, method='uniform')
    lbp_coarse_vals = lbp_coarse[crop_mask == 255]
    hist_coarse, _ = np.histogram(lbp_coarse_vals, bins=np.arange(LBP_COARSE_N_BINS + 1), density=True)
    lbp_feats.update({f'lbp_coarse_bin{i}': float(hist_coarse[i]) for i in range(LBP_COARSE_N_BINS)})

    # ── Edge density (proxy retakan/center-cut, lebih stabil dari HoughLinesP) ──
    edges = cv2.Canny(masked_gray, 50, 150)
    bean_pixel_count = int(np.count_nonzero(crop_mask == 255))
    edge_in_bean = int(np.count_nonzero((edges > 0) & (crop_mask == 255)))
    edge_density = (edge_in_bean / bean_pixel_count) if bean_pixel_count > 0 else 0.0

    features = {**glcm_feats, **lbp_feats, 'edge_density': float(edge_density)}
    return features