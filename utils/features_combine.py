# utils/features_combine.py
"""
Menggabungkan fitur bentuk + warna + tekstur menjadi satu vektor fitur.
Titik masuk tunggal yang dipanggil oleh extract_features.py.
"""

from utils.features_shape import extract_shape_features, SHAPE_FEATURE_NAMES
from utils.features_color import extract_color_features, COLOR_FEATURE_NAMES
from utils.features_texture import extract_texture_features, TEXTURE_FEATURE_NAMES

# Urutan kolom fitur final (dipakai konsisten di CSV & training)
ALL_FEATURE_NAMES = SHAPE_FEATURE_NAMES + COLOR_FEATURE_NAMES + TEXTURE_FEATURE_NAMES


def extract_all_features(binary_img, original_gray, original_rgb):
    """
    Ekstraksi fitur lengkap dari satu citra biji (sudah melalui preprocessing
    + morfologi). Return dict fitur (siap jadi satu baris CSV) + 'is_valid'.

    Parameters
    ----------
    binary_img : hasil morfologi (closing) — untuk segmentasi kontur.
    original_gray : citra grayscale (idealnya hasil normalisasi iluminasi).
    original_rgb : citra RGB (idealnya hasil normalisasi iluminasi) — dipakai
        untuk fitur warna dan sebagai basis crop tekstur.
    """
    shape_feats = extract_shape_features(binary_img, original_gray, original_rgb)

    if not shape_feats['is_valid']:
        result = {name: 0.0 for name in ALL_FEATURE_NAMES}
        result['is_valid'] = False
        return result

    color_feats = extract_color_features(
        original_rgb, original_gray, shape_feats['bean_mask'], bbox=shape_feats['bbox']
    )
    texture_feats = extract_texture_features(original_gray, shape_feats['bean_mask'], shape_feats['bbox'])

    result = {'is_valid': True}
    for name in SHAPE_FEATURE_NAMES:
        result[name] = shape_feats[name]
    for name in COLOR_FEATURE_NAMES:
        result[name] = color_feats[name]
    for name in TEXTURE_FEATURE_NAMES:
        result[name] = texture_feats[name]

    # Info tambahan berguna untuk debugging/visualisasi (bukan fitur RF)
    result['_contour'] = shape_feats['contour']
    result['_hole_contours'] = shape_feats['hole_contours']

    return result