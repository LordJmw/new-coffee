# utils/visualization.py
"""
Helper visualisasi untuk app.py.

PENTING - jawaban untuk pertanyaan "kenapa visualisasi lubang tidak pas":

1. Segmentasi biji-vs-background (kontur hijau) memang pakai Otsu GLOBAL
   (satu ambang untuk seluruh frame 224x224) - lihat
   utils/preprocessing.apply_threshold(). Ini cukup untuk memisahkan biji
   dari latar putih karena kontrasnya besar & konsisten.

2. Deteksi lubang serangga (kontur merah) TIDAK memakai Otsu global yang
   sama. Versi sebelumnya bahkan tidak memakai Otsu sama sekali (cuma
   ambang statistik flat: median - k*std, rata ke seluruh biji) - itu
   sebabnya kontur yang tergambar sering "blok kasar", tidak menempel ke
   tepi lubang asli: satu angka ambang tidak mengikuti gradien pencahayaan
   lokal berbeda-beda di tiap titik permukaan biji.

   Revisi saat ini (features_shape.detect_insect_holes_small) memakai
   black-hat morphological transform + Otsu LOKAL (dihitung hanya dari
   respons black-hat di dalam mask biji, bukan grayscale mentah / bukan
   seluruh frame) supaya ambang mengikuti kontras lokal tiap biji.

   CATATAN JUJUR: permukaan biji kopi kering secara alami punya kerutan/
   center-cut yang juga menciptakan kontras gelap lokal, sehingga deteksi
   berbasis threshold klasik (apa pun variannya) tetap akan menghasilkan
   sejumlah false-positive pada biji Normal/Withered. Ini bukan bug yang
   "bisa dihilangkan total" dengan tweak threshold - ini keterbatasan
   mendasar computer vision klasik pada citra resolusi rendah (224x224).
   Karena itu, pipeline SAAT INI TIDAK menggunakan holes_count sebagai
   aturan keputusan tunggal (beda dari classify_coffee_bean() versi lama
   yang langsung men-declare "Severe Insect Damage" bila holes>=1) -
   holes_count/circularity/contrast hanyalah SALAH SATU dari puluhan fitur
   yang dipelajari Random Forest, sehingga false-positive tunggal tidak
   otomatis salah klasifikasi.

   Untuk TAMPILAN visual saja (bukan untuk fitur RF), fungsi di bawah
   memfilter lubang yang digambar dengan confidence score, supaya app.py
   tidak menampilkan kontur yang lemah/meragukan ke pengguna.
"""

import cv2
import numpy as np


def hole_confidence(meta: dict) -> float:
    """
    Skor kepercayaan 0-1 kasar untuk satu lubang terdeteksi, dari circularity,
    contrast, DAN (REVISI) elongation & solidity - lubang asli mendekati
    bulat & solid, fragmen center-cut/keretakan cenderung memanjang &
    kurang solid meski kontrasnya gelap tajam. Ditambahkan setelah
    percobaan menaruh filter ini di lapisan deteksi (features_shape.py)
    terbukti menghapus sinyal keretakan asli yang dipakai RF untuk kelas
    lain (Broken/Cut/Shell/Black) - jadi elongation/solidity HANYA dipakai
    di sini (murni tampilan), tidak lagi jadi gate keras di deteksi.
    Hanya untuk keperluan TAMPILAN (filter kontur mana yang digambar) -
    bukan dipakai sebagai fitur RF (RF memakai nilai mentahnya langsung).
    """
    circ_score = np.clip(meta.get('circularity', 0.0) / 0.85, 0, 1)
    contrast_score = np.clip(meta.get('contrast', 0.0) / 80.0, 0, 1)
    elongation = meta.get('elongation', 1.0)
    elong_score = np.clip(1.0 - (elongation - 1.0) / 1.2, 0, 1)  # 1.0->1.0 ; >=2.2->~0
    solidity_score = np.clip((meta.get('solidity', 1.0) - 0.5) / 0.4, 0, 1)  # 0.5->0 ; 0.9->1.0
    return float(0.35 * circ_score + 0.25 * contrast_score
                 + 0.25 * elong_score + 0.15 * solidity_score)


def draw_detection_overlay(rgb_img, shape_feats, min_hole_confidence=0.5,
                            bean_color=(0, 255, 0), hole_color=(255, 40, 40),
                            thickness=1):
    """
    Gambar kontur biji + kontur lubang (yang confidence-nya >= threshold)
    di atas rgb_img. Pengganti langsung 'viz_image' yang dulu dikembalikan
    oleh features.py lama - dipisah jadi fungsi sendiri karena visualisasi
    adalah kebutuhan UI (app.py), bukan bagian dari vektor fitur RF.

    Parameters
    ----------
    rgb_img : citra RGB dasar untuk digambar (biasanya pre['rgb']).
    shape_feats : dict hasil features_shape.extract_shape_features(), harus
        berisi 'contour', 'hole_contours', 'hole_meta'.
    min_hole_confidence : lubang dengan confidence di bawah ini tidak
        digambar (tapi TETAP terhitung di fitur holes_count untuk RF -
        filter ini murni kosmetik).
    """
    viz = rgb_img.copy()
    if not shape_feats.get('is_valid', False) or shape_feats.get('contour') is None:
        return viz

    cv2.drawContours(viz, [shape_feats['contour']], -1, bean_color, thickness)

    hole_contours = shape_feats.get('hole_contours', [])
    hole_meta = shape_feats.get('hole_meta', [])
    drawn = 0
    for hc, meta in zip(hole_contours, hole_meta):
        if hole_confidence(meta) >= min_hole_confidence:
            cv2.drawContours(viz, [hc], -1, hole_color, thickness)
            drawn += 1

    return viz


def crop_bean(rgb_img, bbox, pad=5):
    """Crop bounding-box biji dari rgb_img (pengganti 'cropped_rgb' lama)."""
    if bbox is None:
        return rgb_img
    x, y, w, h = bbox
    img_h, img_w = rgb_img.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    cropped = rgb_img[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else rgb_img