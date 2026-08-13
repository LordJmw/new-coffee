# utils/features_shape.py
"""
Fitur bentuk/geometri.

Perbedaan vs versi lama:
- Tetap pakai area, perimeter, circularity, solidity, extent, aspect_ratio
  (sudah terbukti bekerja untuk 6-kelas lama).
- Ditambah 7 Hu Moments (log-scaled) -> deskriptor bentuk invarian
  translasi/rotasi/skala. Berguna memisahkan kelas yang bedanya di bentuk
  utuh biji, bukan warna/tekstur permukaan: Broken/Cut/Shell (Physical
  Damage) vs Normal, dan bentuk biji layu (Withered) yang non-simetris.
- Deteksi lubang serangga dual-track dipindah ke sini apa adanya dari
  features.py lama (sudah tervalidasi), diperluas dengan agregat
  (total luas lubang & rasio thd luas biji) supaya granularitas
  slight vs severe insect damage tidak sepenuhnya hilang walau digabung
  jadi satu kelas final "Insect Damage".
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Deteksi lubang serangga — dual-track (dipertahankan dari versi lama)
# ─────────────────────────────────────────────────────────────────────────

def detect_insect_holes_large(contours, hierarchy, main_idx, main_area):
    """Track A: rongga besar yang menembus biji (background terlihat dari dalam)."""
    hole_contours, hole_areas = [], []
    for i in range(len(contours)):
        if hierarchy[0][i][3] != main_idx:
            continue
        h_area = cv2.contourArea(contours[i])
        if h_area < 150:
            continue
        ratio = h_area / main_area if main_area > 0 else 0
        if not (0.015 <= ratio <= 0.60):
            continue
        M = cv2.moments(contours[i])
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        bx, by, bw, bh = cv2.boundingRect(contours[main_idx])
        margin = 5
        if not (bx + margin < cx < bx + bw - margin and by + margin < cy < by + bh - margin):
            continue
        hole_contours.append(contours[i])
        hole_areas.append(h_area)
    return hole_contours, hole_areas


def detect_insect_holes_small(original_gray, original_rgb, main_contour, main_area, bean_mask):
    """
    Track B: titik gelap kecil (titik masuk serangga).

    REVISI (jawab pertanyaan "apakah pakai Otsu?"): sebelumnya Track B TIDAK
    memakai Otsu sama sekali - hanya threshold statistik flat (median - k*std)
    yang diterapkan RATA ke seluruh biji. Itu sebabnya kontur lubang yang
    divisualisasikan sering "tidak pas" di app.py: satu angka threshold
    global tidak mengikuti gradien pencahayaan lokal di sekitar lubang,
    sehingga tepi kontur jadi blok kasar, bukan menempel ke tepi lubang asli.

    Sekarang dipakai 2 langkah agar kontur lebih presisi:
      1. Black-hat morphological transform pada grayscale (dalam bbox biji) -
         menonjolkan struktur gelap KECIL relatif terhadap latar lokal di
         sekitarnya (bukan relatif ke rata-rata seluruh biji), sehingga
         tahan terhadap gradasi warna alami permukaan biji.
      2. Otsu diterapkan PADA RESPON BLACK-HAT tersebut (bukan pada
         grayscale mentah, dan bukan pada seluruh frame) -> ambang otomatis
         yang mengikuti kontras lokal tiap biji, hasil kontur lebih menempel
         ke tepi lubang asli dibanding threshold statistik flat sebelumnya.
    """
    img_h, img_w = original_gray.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(main_contour)
    bean_pixels = original_gray[bean_mask == 255]
    if bean_pixels.size < 100:
        return [], [], []

    median_intensity = float(np.median(bean_pixels))
    std_intensity = float(np.std(bean_pixels))

    # ── Black-hat: kernel selebar lubang serangga tipikal (biji 224x224) ──
    pad = 6
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(img_w, bx + bw + pad), min(img_h, by + bh + pad)
    crop_gray = original_gray[y1:y2, x1:x2]
    crop_mask = bean_mask[y1:y2, x1:x2]
    if crop_gray.size == 0:
        return [], [], []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(crop_gray, cv2.MORPH_BLACKHAT, kernel)
    # Nolkan respon di luar mask biji supaya tidak ikut mempengaruhi ambang Otsu
    blackhat_masked = blackhat.copy()
    blackhat_masked[crop_mask != 255] = 0

    if blackhat_masked.max() < 8:  # nyaris tidak ada struktur gelap lokal -> tidak ada lubang
        return [], [], []

    # Otsu HANYA pada respon black-hat di dalam mask biji (bukan seluruh frame)
    bean_blackhat_vals = blackhat_masked[crop_mask == 255]
    otsu_thresh, _ = cv2.threshold(
        bean_blackhat_vals.reshape(-1, 1).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    local_contrast_mask = blackhat_masked >= otsu_thresh

    # Gabungkan dengan syarat kegelapan absolut (relatif median seluruh biji)
    # supaya hanya tekstur permukaan yang JUGA benar-benar gelap yang lolos -
    # black-hat sendirian bisa terlalu sensitif pada variasi tekstur
    # permukaan yang halus (bukan lubang sungguhan), jadi dua sinyal
    # (kontras lokal + kegelapan absolut) harus sepakat.
    absolute_dark_thresh = max(30, min(110, median_intensity - 1.3 * std_intensity))
    absolute_dark_mask = crop_gray < absolute_dark_thresh

    dark_mask = np.zeros_like(crop_gray, dtype=np.uint8)
    dark_mask[local_contrast_mask & absolute_dark_mask & (crop_mask == 255)] = 255

    k = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k, iterations=1)
    spot_contours_crop, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hole_contours, hole_areas, hole_meta = [], [], []
    for sc_crop in spot_contours_crop:
        s_area = cv2.contourArea(sc_crop)
        if not (3 <= s_area <= 80):
            continue
        if s_area / main_area > 0.03:
            continue
        s_perim = cv2.arcLength(sc_crop, True)
        if s_perim == 0:
            continue
        s_circ = (4 * np.pi * s_area) / (s_perim ** 2)
        if s_circ < 0.25:
            continue
        M = cv2.moments(sc_crop)
        if M['m00'] == 0:
            continue
        cx_crop, cy_crop = M['m10'] / M['m00'], M['m01'] / M['m00']
        cx, cy = cx_crop + x1, cy_crop + y1  # balik ke koordinat frame penuh
        margin = 8
        if not (bx + margin < cx < bx + bw - margin and by + margin < cy < by + bh - margin):
            continue

        spot_mask_single = np.zeros_like(crop_gray, dtype=np.uint8)
        cv2.drawContours(spot_mask_single, [sc_crop], -1, 255, -1)
        spot_pixels = crop_gray[spot_mask_single == 255]
        if spot_pixels.size == 0:
            continue
        crop_rgb = original_rgb[y1:y2, x1:x2]
        spot_rgb = crop_rgb[spot_mask_single == 255]
        if spot_rgb.size > 0 and spot_rgb.mean(axis=0).max() > 130:
            continue  # terlalu terang untuk lubang serangga asli

        contrast = float(median_intensity - np.mean(spot_pixels))
        sc_full = sc_crop + np.array([[x1, y1]])  # translasi kontur ke koordinat frame penuh

        hole_contours.append(sc_full)
        hole_areas.append(s_area)
        hole_meta.append({'circularity': float(s_circ), 'contrast': contrast})

    return hole_contours, hole_areas, hole_meta


def detect_insect_holes(contours, hierarchy, main_idx, main_area,
                         original_gray, original_rgb, main_contour, bean_mask):
    """Gabungan Track A + Track B dengan deduplication. Return juga hole_meta
    (circularity & contrast per lubang Track B; Track A diberi nilai default
    karena rongga besar Track A sudah pasti valid secara geometris)."""
    holes_a, areas_a = detect_insect_holes_large(contours, hierarchy, main_idx, main_area)
    meta_a = [{'circularity': 1.0, 'contrast': 255.0} for _ in holes_a]
    holes_b, areas_b, meta_b = detect_insect_holes_small(
        original_gray, original_rgb, main_contour, main_area, bean_mask)

    if holes_a:
        filtered_b, filtered_areas_b, filtered_meta_b = [], [], []
        for hb, ab, mb in zip(holes_b, areas_b, meta_b):
            M = cv2.moments(hb)
            if M['m00'] == 0:
                continue
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            inside_a = any(cv2.pointPolygonTest(ha, (cx, cy), False) >= 0 for ha in holes_a)
            if not inside_a:
                filtered_b.append(hb)
                filtered_areas_b.append(ab)
                filtered_meta_b.append(mb)
        holes_b, areas_b, meta_b = filtered_b, filtered_areas_b, filtered_meta_b

    return holes_a + holes_b, areas_a + areas_b, meta_a + meta_b


# ─────────────────────────────────────────────────────────────────────────
# Segmentasi biji utama (dipakai bersama oleh semua modul fitur)
# ─────────────────────────────────────────────────────────────────────────

def find_main_bean_contour(binary_img):
    """
    Cari kontur biji utama (kontur terluar dengan area terbesar).
    Return: (contours, hierarchy, main_idx, main_area) — main_idx = -1 jika gagal.
    """
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return contours, hierarchy, -1, 0

    main_idx, max_area = -1, 0
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area
                main_idx = i

    if main_idx == -1 or max_area < 500:
        return contours, hierarchy, -1, 0

    return contours, hierarchy, main_idx, max_area


# ─────────────────────────────────────────────────────────────────────────
# Fitur bentuk utama
# ─────────────────────────────────────────────────────────────────────────

SHAPE_FEATURE_NAMES = [
    'area', 'perimeter', 'circularity', 'solidity', 'convexity', 'extent',
    'aspect_ratio', 'eccentricity',
    'holes_count', 'holes_area_ratio', 'holes_mean_area',
    'holes_circularity_mean', 'holes_contrast_mean', 'holes_contrast_max',
    'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
]


def extract_shape_features(binary_img, original_gray, original_rgb):
    """
    Ekstrak fitur bentuk dari citra biner hasil morfologi.
    Return dict fitur + info kontur (dipakai modul warna/tekstur lain agar
    tidak perlu segmentasi ulang).
    """
    empty = {name: 0.0 for name in SHAPE_FEATURE_NAMES}
    empty.update({'is_valid': False, 'contour': None, 'bean_mask': None,
                  'bbox': None, 'hole_contours': [], 'hole_areas': [], 'hole_meta': []})

    contours, hierarchy, main_idx, max_area = find_main_bean_contour(binary_img)
    if main_idx == -1:
        return empty

    cnt = contours[main_idx]
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    solidity = float(max_area) / hull_area if hull_area > 0 else 0
    # Convexity: rasio KELILING hull vs kontur asli - komplementer thd solidity
    # (yang berbasis LUAS). Solidity bisa "tidak sensitif" pada cekungan kecil
    # tapi rapat (area yang hilang sedikit), sedangkan convexity langsung
    # menangkap kekasaran/gerigi tepi kontur (perimeter jadi lebih panjang
    # dari hull-nya) - berguna untuk Broken/Cut/Shell yang tepi patahannya
    # tajam & tidak beraturan meski areanya tidak banyak berkurang.
    convexity = float(hull_perimeter) / perimeter if perimeter > 0 else 0
    extent = float(max_area) / (w * h) if w * h > 0 else 0
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0

    # Eccentricity dari ellipse-fit: lebih tepat dari aspect_ratio (bounding-box)
    # karena TIDAK bergantung pada orientasi/rotasi biji di dalam frame -
    # penting karena biji hasil crop YOLO (domain target) orientasinya acak,
    # beda dari kondisi lab yang mungkin lebih terkontrol.
    eccentricity = 0.0
    if len(cnt) >= 5:  # fitEllipse butuh minimal 5 titik
        try:
            (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
            ma, MA = min(ma, MA), max(ma, MA)
            if MA > 0:
                ratio_sq = (ma / MA) ** 2
                eccentricity = float(np.sqrt(max(0.0, 1 - ratio_sq)))
        except cv2.error:
            eccentricity = 0.0

    img_h, img_w = original_rgb.shape[:2]
    bean_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(bean_mask, [cnt], -1, 255, -1)

    hole_contours, hole_areas, hole_meta = detect_insect_holes(
        contours, hierarchy, main_idx, max_area, original_gray, original_rgb, cnt, bean_mask
    )
    holes_count = len(hole_contours)
    holes_total_area = float(sum(hole_areas)) if hole_areas else 0.0
    holes_area_ratio = holes_total_area / max_area if max_area > 0 else 0.0
    holes_mean_area = holes_total_area / holes_count if holes_count > 0 else 0.0

    # Deskriptor tambahan lubang: lubang serangga asli cenderung BULAT
    # (circularity tinggi) & kontras gelap TAJAM thd jaringan sekitar,
    # berbeda dari false-positive garis lipatan/bayangan (memanjang, kontras
    # halus). Nilai ini membantu RF membedakan tanpa perlu threshold keras.
    if hole_meta:
        circs = [m['circularity'] for m in hole_meta]
        contrasts = [m['contrast'] for m in hole_meta]
        holes_circularity_mean = float(np.mean(circs))
        holes_contrast_mean = float(np.mean(contrasts))
        holes_contrast_max = float(np.max(contrasts))
    else:
        holes_circularity_mean = 0.0
        holes_contrast_mean = 0.0
        holes_contrast_max = 0.0

    # Hu Moments: log-scale agar rentang nilai stabil untuk RF (nilai asli bisa
    # sangat kecil/besar), tanda dipertahankan dengan sign(x)*log10(|x|).
    hu_raw = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu_log = [float(np.sign(v) * np.log10(abs(v) + 1e-12)) for v in hu_raw]

    features = {
        'is_valid': True,
        'area': float(max_area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'solidity': float(solidity),
        'convexity': float(convexity),
        'extent': float(extent),
        'aspect_ratio': float(aspect_ratio),
        'eccentricity': eccentricity,
        'holes_count': int(holes_count),
        'holes_area_ratio': float(holes_area_ratio),
        'holes_mean_area': float(holes_mean_area),
        'holes_circularity_mean': holes_circularity_mean,
        'holes_contrast_mean': holes_contrast_mean,
        'holes_contrast_max': holes_contrast_max,
        'contour': cnt,
        'bean_mask': bean_mask,
        'bbox': (x, y, w, h),
        'hole_contours': hole_contours,
        'hole_areas': hole_areas,
        'hole_meta': hole_meta,
    }
    for i, v in enumerate(hu_log, start=1):
        features[f'hu{i}'] = v

    return features