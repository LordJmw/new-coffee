# utils/features_color.py
"""
Fitur warna multi color-space.

Perbedaan vs versi lama (yang hanya red_ratio/green_ratio dari RGB mentah):
- RGB mentah rentan terhadap variasi iluminasi -> ditambah preprocessing
  normalisasi (lihat preprocessing.py), TAPI fitur warna sendiri juga
  dibuat lebih tahan-iluminasi:
    * Normalized chromaticity (r,g dari r=R/(R+G+B), g=G/(R+G+B)) -> invarian
      terhadap perubahan intensitas cahaya seragam (illumination-invariant
      under diagonal/scaling model).
    * HSV: Hue tidak berubah oleh perubahan Value (kecerahan) -> baik untuk
      membedakan Sour (kekuningan/kecoklatan) vs Normal vs Black.
    * Lab: a*/b* adalah kanal warna murni, L* (lightness) dipisah agar model
      bisa belajar warna tanpa tercampur variasi pencahayaan.
- Color moments (mean, std, skewness) per kanal H, S, a*, b* -> menangkap
  variasi warna permukaan (mis. Fungus Damage yang berbintik, Partial
  Black/Sour yang warnanya tidak seragam), bukan cuma rata-rata.

REVISI (analisis kesalahan Insect Damage & Immature/Discoloration):
- Histogram Hue (8 bin, dinormalisasi) -> mean/std satu angka tidak cukup
  menangkap distribusi warna yang tidak simetris (mis. Immature yang
  hue-nya condong ke hijau-kekuningan pucat dibanding Normal yang lebih
  coklat-kekuningan pekat). Histogram memberi RF resolusi distribusi, bukan
  cuma rata-rata.
- Fitur "patchiness" (grid 3x3 di dalam bounding box, ambil std antar-blok
  dari L*, a*, b*, Hue) -> menangkap warna yang TIDAK merata di permukaan
  biji (Partial Sour/Partial Black/Fungus Damage berbintik/Immature yang
  pudar tidak rata), sesuatu yang tidak tertangkap oleh rata-rata warna
  global karena rata-rata blok pudar & blok normal bisa saling menutupi.
"""

import cv2
import numpy as np


def _skewness(x):
    x = x.astype(np.float64)
    if x.size == 0:
        return 0.0
    std = x.std()
    if std < 1e-6:
        return 0.0
    return float(np.mean(((x - x.mean()) / std) ** 3))


HUE_HIST_BINS = 8
GRID_SIZE = 3  # grid 3x3 untuk fitur patchiness

COLOR_FEATURE_NAMES = (
    [
        'red_ratio', 'green_ratio', 'blue_ratio',           # normalized chromaticity (illum-invariant)
        'mean_intensity',                                     # grayscale lightness (illum-sensitive, dipertahankan utk baseline)
        'hue_mean', 'hue_std', 'sat_mean', 'sat_std', 'val_mean', 'val_std',
        'L_mean', 'a_mean', 'a_std', 'a_skew', 'b_mean', 'b_std', 'b_skew',
    ]
    + [f'hue_hist{i}' for i in range(HUE_HIST_BINS)]
    + ['patch_L_std', 'patch_a_std', 'patch_b_std', 'patch_hue_std']
)


def _grid_patchiness(channel_2d, mask_2d, grid_size=GRID_SIZE):
    """
    Bagi bounding-box channel jadi grid_size x grid_size blok, hitung rata-rata
    tiap blok (hanya piksel di dalam mask), lalu kembalikan std antar-blok.
    std tinggi -> warna tidak merata di permukaan biji (indikasi cacat parsial/
    berbintik: Partial Sour, Partial Black, Fungus Damage, atau Immature yang
    pudar tidak rata).
    """
    h, w = channel_2d.shape
    if h < grid_size or w < grid_size:
        return 0.0
    block_means = []
    ys = np.linspace(0, h, grid_size + 1, dtype=int)
    xs = np.linspace(0, w, grid_size + 1, dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            block_mask = mask_2d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if not np.any(block_mask):
                continue
            block_ch = channel_2d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            block_means.append(float(np.mean(block_ch[block_mask])))
    if len(block_means) < 2:
        return 0.0
    return float(np.std(block_means))


def extract_color_features(original_rgb, original_gray, bean_mask, bbox=None):
    """
    Ekstrak fitur warna dalam region biji (bean_mask == 255).
    original_rgb sebaiknya citra yang SUDAH dinormalisasi iluminasinya
    (lihat preprocessing.normalize_illumination), agar fitur warna lebih
    stabil lintas domain capture.
    bbox (x, y, w, h): jika diberikan, fitur patchiness dihitung hanya pada
    crop bounding-box biji (lebih presisi & lebih cepat daripada full-frame).
    """
    empty = {name: 0.0 for name in COLOR_FEATURE_NAMES}
    if bean_mask is None or not np.any(bean_mask == 255):
        return empty

    mask_bool = bean_mask == 255

    # ── Normalized chromaticity dari RGB ──
    mean_val = cv2.mean(original_rgb, mask=bean_mask)
    r_mean, g_mean, b_mean = mean_val[0], mean_val[1], mean_val[2]
    total_rgb = r_mean + g_mean + b_mean
    red_ratio = (r_mean / total_rgb * 100) if total_rgb > 0 else 0
    green_ratio = (g_mean / total_rgb * 100) if total_rgb > 0 else 0
    blue_ratio = (b_mean / total_rgb * 100) if total_rgb > 0 else 0

    mean_intensity = float(np.mean(original_gray[mask_bool])) if np.any(mask_bool) else 0.0

    # ── HSV ──
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0][mask_bool], hsv[..., 1][mask_bool], hsv[..., 2][mask_bool]

    # ── Lab ──
    lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = lab[..., 0][mask_bool], lab[..., 1][mask_bool], lab[..., 2][mask_bool]

    # ── Histogram Hue (distribusi, bukan cuma mean/std) ──
    hue_hist, _ = np.histogram(h_ch, bins=HUE_HIST_BINS, range=(0, 180), density=True)
    hue_hist = hue_hist / (hue_hist.sum() + 1e-9)

    # ── Patchiness (grid 3x3, dihitung pada crop bounding-box bila tersedia) ──
    if bbox is not None:
        x, y, w, h = bbox
        img_h, img_w = original_rgb.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        crop_mask = mask_bool[y1:y2, x1:x2]
        crop_hsv_h = hsv[y1:y2, x1:x2, 0]
        crop_lab_l = lab[y1:y2, x1:x2, 0]
        crop_lab_a = lab[y1:y2, x1:x2, 1]
        crop_lab_b = lab[y1:y2, x1:x2, 2]
    else:
        crop_mask, crop_hsv_h = mask_bool, hsv[..., 0]
        crop_lab_l, crop_lab_a, crop_lab_b = lab[..., 0], lab[..., 1], lab[..., 2]

    patch_L_std = _grid_patchiness(crop_lab_l, crop_mask)
    patch_a_std = _grid_patchiness(crop_lab_a, crop_mask)
    patch_b_std = _grid_patchiness(crop_lab_b, crop_mask)
    patch_hue_std = _grid_patchiness(crop_hsv_h, crop_mask)

    features = {
        'red_ratio': float(red_ratio),
        'green_ratio': float(green_ratio),
        'blue_ratio': float(blue_ratio),
        'mean_intensity': mean_intensity,
        'hue_mean': float(np.mean(h_ch)), 'hue_std': float(np.std(h_ch)),
        'sat_mean': float(np.mean(s_ch)), 'sat_std': float(np.std(s_ch)),
        'val_mean': float(np.mean(v_ch)), 'val_std': float(np.std(v_ch)),
        'L_mean': float(np.mean(l_ch)),
        'a_mean': float(np.mean(a_ch)), 'a_std': float(np.std(a_ch)), 'a_skew': _skewness(a_ch),
        'b_mean': float(np.mean(b_ch)), 'b_std': float(np.std(b_ch)), 'b_skew': _skewness(b_ch),
        'patch_L_std': patch_L_std, 'patch_a_std': patch_a_std,
        'patch_b_std': patch_b_std, 'patch_hue_std': patch_hue_std,
    }
    for i in range(HUE_HIST_BINS):
        features[f'hue_hist{i}'] = float(hue_hist[i])
    return features