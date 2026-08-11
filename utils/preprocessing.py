# utils/preprocessing.py
"""
Preprocessing pipeline Tahap 2.

Perbedaan vs versi lama:
- Ditambah normalisasi iluminasi (gray-world white balance + CLAHE pada
  kanal L di ruang Lab) SEBELUM segmentasi & ekstraksi fitur warna, karena
  domain gap utama pada penelitian ini adalah variasi pencahayaan/device
  (Bagian 8, "Ringkasan Arah Skripsi Kopi").
- Segmentasi & fitur bentuk dihitung dari citra ternormalisasi, TAPI citra
  RGB asli (sebelum normalisasi) tetap disimpan ('rgb_raw') supaya fitur
  warna absolut (untuk few-shot/self-training domain adaptation) masih bisa
  dibandingkan dengan versi ternormalisasi bila dibutuhkan pada eksperimen.
"""

import cv2
import numpy as np


def load_and_convert(image_array):
    """M1: Konversi input ke format RGB yang sesuai."""
    if len(image_array.shape) == 2:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = image_array.copy()
    return img_rgb


def downsample_image(image_array, target_size=(224, 224), interpolation_method="Area-based"):
    """M2: Resize gambar untuk konsistensi koordinat fitur."""
    interp_dict = {
        "Nearest Neighbor": cv2.INTER_NEAREST,
        "Bilinear interpolation": cv2.INTER_LINEAR,
        "Bicubic interpolation": cv2.INTER_CUBIC,
        "Area-based": cv2.INTER_AREA,
        "Lanczos": cv2.INTER_LANCZOS4,
    }
    cv2_interp = interp_dict.get(interpolation_method, cv2.INTER_AREA)
    img_resized = cv2.resize(image_array, target_size, interpolation=cv2_interp)
    return img_resized


def gray_world_white_balance(img_rgb):
    """
    Koreksi white balance sederhana (gray-world assumption).
    Mengurangi efek warna dari perbedaan sumber cahaya (kuning vs putih vs
    fluorescent) antar sesi pemotretan / device.
    """
    img_f = img_rgb.astype(np.float32)
    mean_per_channel = img_f.reshape(-1, 3).mean(axis=0)
    gray_mean = mean_per_channel.mean()
    # Hindari pembagian oleh nol / penguatan ekstrem pada channel nyaris hitam
    scale = gray_mean / np.clip(mean_per_channel, 10, None)
    scale = np.clip(scale, 0.5, 2.0)
    balanced = img_f * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_clahe_lab(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE pada kanal L (lightness) di ruang Lab.
    Menyamakan kontras lokal tanpa mendistorsi informasi warna (a*, b*),
    sehingga fitur warna tetap valid setelah normalisasi iluminasi.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def normalize_illumination(img_rgb, use_white_balance=True, use_clahe=True):
    """Pipeline normalisasi iluminasi gabungan (M2.5, opsional tapi disarankan aktif)."""
    out = img_rgb
    if use_white_balance:
        out = gray_world_white_balance(out)
    if use_clahe:
        out = apply_clahe_lab(out)
    return out


def rgb_to_grayscale(image_array):
    """M3: Konversi RGB ke Grayscale."""
    if len(image_array.shape) == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array.copy()
    return img_gray


def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    """M3: Reduksi Noise dengan Gaussian Blur."""
    if kernel_size[0] % 2 == 0:
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1)
    img_blur = cv2.GaussianBlur(gray_image, kernel_size, 0)
    return img_blur


def apply_threshold(gray_image, method='otsu'):
    """M3: Thresholding untuk Segmentasi Biji."""
    if method == 'otsu':
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary


def preprocess_pipeline(image_array, target_size=(224, 224), blur_kernel=(5, 5),
                         threshold_method='otsu', interpolation_method="Area-based",
                         normalize_illum=True):
    """
    Pipeline preprocessing single-scale (koordinat kontur sinkron dengan
    koordinat warna). Ditambah normalisasi iluminasi opsional (default aktif)
    sebelum grayscale/threshold, agar segmentasi lebih stabil lintas domain.
    """
    img_original = load_and_convert(image_array)
    img_rgb_raw = downsample_image(img_original, target_size, interpolation_method)

    img_rgb = normalize_illumination(img_rgb_raw) if normalize_illum else img_rgb_raw.copy()

    img_gray = rgb_to_grayscale(img_rgb)
    img_blur = apply_gaussian_blur(img_gray, blur_kernel)
    img_binary = apply_threshold(img_blur, threshold_method)

    results = {
        'rgb': img_rgb,               # ternormalisasi -> dipakai fitur warna & tampilan UI
        'rgb_raw': img_rgb_raw,        # sebelum normalisasi -> disimpan untuk analisis domain gap
        'gray': img_gray,
        'blur': img_blur,
        'binary': img_binary,
        'original_rgb': img_original,
        'target_size': target_size,
    }
    return results
