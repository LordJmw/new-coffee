# utils/features_combine.py
"""
Menggabungkan fitur bentuk + warna + tekstur menjadi satu vektor fitur.
Titik masuk tunggal yang dipanggil oleh extract_features.py.
"""

from utils.features_shape import extract_shape_features, SHAPE_FEATURE_NAMES
from utils.features_color import extract_color_features, COLOR_FEATURE_NAMES
from utils.features_texture import extract_texture_features, TEXTURE_FEATURE_NAMES

# REVISI - dibuang dari fitur ML (bukan dari extract_color_features() itu
# sendiri, yang tetap menghitungnya untuk kompatibilitas app_integration/app.py
# yang masih memakai mean_intensity dkk untuk tampilan/logika lama).
#
# Terbukti empiris (ablation, split test hold-out identik): membuang 14
# statistik warna ABSOLUT ini dari fitur yang dilatihkan ke RF menaikkan
# macro-F1 0.8026 -> 0.8081, karena mereka membawa sinyal yang terlalu
# spesifik ke kondisi pencahayaan dataset training (bukan sinyal warna
# identitas kelas yang generalize) - RF "terlalu percaya" fitur ini di
# training padahal tidak robust ke data baru. red_ratio/green_ratio/
# blue_ratio (chromaticity, invariant thd exposure uniform) TETAP dipakai
# karena justru grup fitur paling efisien per-fitur (importance tertinggi
# per fitur dari semua grup).
ABSOLUTE_COLOR_FEATURES_EXCLUDED_FROM_ML = [
    'mean_intensity', 'hue_mean', 'hue_std', 'sat_mean', 'sat_std',
    'val_mean', 'val_std', 'L_mean', 'a_mean', 'a_std', 'a_skew',
    'b_mean', 'b_std', 'b_skew',
]

COLOR_FEATURE_NAMES_ML = [
    c for c in COLOR_FEATURE_NAMES if c not in ABSOLUTE_COLOR_FEATURES_EXCLUDED_FROM_ML
]

# Urutan kolom fitur final (dipakai konsisten di CSV & training) - HANYA
# fitur yang lolos validasi generalisasi. Fitur warna absolut tetap
# dihitung oleh extract_color_features() (lihat 'result' penuh di bawah,
# atau panggil utils.features_color langsung) untuk keperluan non-ML
# (display/debug/kompatibilitas app lama), tapi tidak masuk sini.
ALL_FEATURE_NAMES = SHAPE_FEATURE_NAMES + COLOR_FEATURE_NAMES_ML + TEXTURE_FEATURE_NAMES


def extract_all_features(binary_img, original_gray, original_rgb, include_excluded_color=False):
    """
    Ekstraksi fitur lengkap dari satu citra biji (sudah melalui preprocessing
    + morfologi). Return dict fitur (siap jadi satu baris CSV) + 'is_valid'.

    Parameters
    ----------
    binary_img : hasil morfologi (closing) — untuk segmentasi kontur.
    original_gray : citra grayscale (idealnya hasil normalisasi iluminasi).
    original_rgb : citra RGB (idealnya hasil normalisasi iluminasi) — dipakai
        untuk fitur warna dan sebagai basis crop tekstur.
    include_excluded_color : jika True, sertakan juga 14 fitur warna absolut
        (mean_intensity, a_mean, b_mean, dst) di hasil - berguna untuk
        app_integration/app.py yang masih menampilkannya, TAPI kolom ini
        TIDAK PERNAH masuk ALL_FEATURE_NAMES / CSV training (lihat catatan
        di atas). Default False supaya pipeline training/extract_features.py
        tetap bersih tanpa perlu diubah.
    """
    shape_feats = extract_shape_features(binary_img, original_gray, original_rgb)

    if not shape_feats['is_valid']:
        result = {name: 0.0 for name in ALL_FEATURE_NAMES}
        if include_excluded_color:
            result.update({name: 0.0 for name in ABSOLUTE_COLOR_FEATURES_EXCLUDED_FROM_ML})
        result['is_valid'] = False
        return result

    color_feats = extract_color_features(
        original_rgb, original_gray, shape_feats['bean_mask'], bbox=shape_feats['bbox']
    )
    texture_feats = extract_texture_features(original_gray, shape_feats['bean_mask'], shape_feats['bbox'])

    result = {'is_valid': True}
    for name in SHAPE_FEATURE_NAMES:
        result[name] = shape_feats[name]
    for name in COLOR_FEATURE_NAMES_ML:
        result[name] = color_feats[name]
    for name in TEXTURE_FEATURE_NAMES:
        result[name] = texture_feats[name]
    if include_excluded_color:
        for name in ABSOLUTE_COLOR_FEATURES_EXCLUDED_FROM_ML:
            result[name] = color_feats[name]

    # Info tambahan berguna untuk debugging/visualisasi (bukan fitur RF)
    result['_contour'] = shape_feats['contour']
    result['_hole_contours'] = shape_feats['hole_contours']

    return result