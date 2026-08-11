"""
STEP 1 (Tahap 2): Ekstraksi Fitur dari Dataset 17 Kelas Cacat + Normal
-> dipetakan ke 7 Kelas Final (SNI 01-2907-2008-aligned).

Membaca semua gambar di DATASET_PATH (struktur: 1 subfolder per subkelas
Kaggle, lihat utils/label_mapping.py untuk daftar nama folder yang valid),
mengekstrak fitur bentuk + warna + tekstur (utils/features_combine.py),
dan menghasilkan satu CSV siap pakai untuk training Random Forest
(train_evaluate_rf.py).

Setiap baris CSV memuat baik label granular (subclass_name, 18 nilai)
maupun label final (class_name, 7 nilai) - supaya analisis kesalahan tetap
bisa ditelusuri sampai ke subkelas asli Kaggle bila diperlukan.
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features_combine import extract_all_features, ALL_FEATURE_NAMES
from utils.label_mapping import resolve_folder, FINAL_CLASS_TO_NUM

# ── Konfigurasi ──
DATASET_PATH = "data/train"   # ganti sesuai lokasi dataset penuh (Aset A)
OUTPUT_CSV = "data/features_dataset_7kelas.csv"

TARGET_SIZE = (224, 224)
BLUR_KERNEL = (5, 5)
INTERPOLATION = "Area-based"
OPEN_KERNEL = (3, 3)
CLOSE_KERNEL = (3, 3)
NORMALIZE_ILLUMINATION = True   # aktifkan white-balance + CLAHE (lihat preprocessing.py)

CSV_COLUMNS = (
    ['filename', 'subclass_name', 'class_name', 'class_num']
    + ALL_FEATURE_NAMES
    + ['is_valid']
)


def load_image(image_path):
    """Load gambar dari path, return RGB array (atau None jika gagal)."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_features_from_image(image_path):
    """Jalankan pipeline penuh: preprocessing -> morfologi -> fitur, untuk satu citra."""
    img_rgb = load_image(image_path)
    if img_rgb is None:
        return None

    pre = preprocess_pipeline(
        img_rgb,
        target_size=TARGET_SIZE,
        blur_kernel=BLUR_KERNEL,
        interpolation_method=INTERPOLATION,
        normalize_illum=NORMALIZE_ILLUMINATION,
    )
    morph = apply_morphology(pre['binary'], open_kernel=OPEN_KERNEL, close_kernel=CLOSE_KERNEL)

    features = extract_all_features(
        morph['closing'],       # binary hasil morfologi -> segmentasi
        pre['gray'],             # grayscale ternormalisasi -> fitur warna/tekstur
        pre['rgb'],               # RGB ternormalisasi -> fitur warna
    )
    return features


def collect_image_list(dataset_path):
    """
    Kumpulkan semua path citra beserta label (subclass & final class).
    Folder yang benar-benar ada di disk yang dipindai (bukan daftar tetap),
    lalu setiap nama folder dicocokkan lewat resolve_folder() -- otomatis
    menerima konvensi "full sour bean" ATAU "Full Sour" (lihat label_mapping.py).
    """
    all_images = []
    unrecognized_folders = []

    subfolders = sorted(
        d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
    )

    for folder_name in subfolders:
        resolved = resolve_folder(folder_name)
        if resolved is None:
            unrecognized_folders.append(folder_name)
            continue
        subclass, final_class = resolved
        folder_path = os.path.join(dataset_path, folder_name)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append({
                    'path': os.path.join(folder_path, filename),
                    'filename': filename,
                    'subclass_name': subclass,
                    'class_name': final_class,
                    'class_num': FINAL_CLASS_TO_NUM[final_class],
                })

    return all_images, unrecognized_folders


def main():
    print("=" * 70)
    print("STEP 1: Ekstraksi Fitur - 17 Kelas Cacat + Normal -> 7 Kelas Final")
    print("=" * 70)
    print(f"Dataset path : {DATASET_PATH}")
    print(f"Output CSV   : {OUTPUT_CSV}")
    print(f"Normalisasi iluminasi: {'AKTIF' if NORMALIZE_ILLUMINATION else 'nonaktif'}")
    print()

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Folder dataset tidak ditemukan: {DATASET_PATH}")
        print("Pastikan path sudah benar dan berisi subfolder per subkelas.")
        return None, None

    all_images, unrecognized_folders = collect_image_list(DATASET_PATH)
    for uf in unrecognized_folders:
        print(f"[WARNING] Folder tidak dikenali (dilewati): {uf}")

    print(f"\nTotal gambar ditemukan: {len(all_images)}")
    print("\nPer kelas final (7 kelas):")
    for class_name in sorted(set(img['class_name'] for img in all_images), key=lambda c: FINAL_CLASS_TO_NUM[c]):
        count = sum(1 for img in all_images if img['class_name'] == class_name)
        print(f"   - {class_name}: {count} gambar")

    if len(all_images) == 0:
        print("[ERROR] Tidak ada gambar ditemukan. Cek DATASET_PATH.")
        return None, None

    results = []
    failed_images = []

    print("\nMemulai ekstraksi fitur...")
    for img_info in tqdm(all_images, desc="Processing images"):
        try:
            features = extract_features_from_image(img_info['path'])
            if features is None or not features.get('is_valid', False):
                failed_images.append({
                    'filename': img_info['filename'],
                    'class': img_info['class_name'],
                    'reason': 'Biji tidak terdeteksi (is_valid=False)',
                })
                continue

            row = {
                'filename': img_info['filename'],
                'subclass_name': img_info['subclass_name'],
                'class_name': img_info['class_name'],
                'class_num': img_info['class_num'],
                'is_valid': True,
            }
            for fname in ALL_FEATURE_NAMES:
                row[fname] = features.get(fname, 0.0)
            results.append(row)

        except Exception as e:
            failed_images.append({
                'filename': img_info['filename'],
                'class': img_info['class_name'],
                'reason': str(e),
            })

    df = pd.DataFrame(results, columns=CSV_COLUMNS)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("HASIL EKSTRAKSI FITUR")
    print("=" * 70)
    print(f"Berhasil diproses : {len(results)} gambar")
    print(f"Gagal             : {len(failed_images)} gambar")
    print(f"CSV disimpan di   : {OUTPUT_CSV}")

    if failed_images:
        print("\nContoh gambar yang gagal (maks. 10):")
        for fail in failed_images[:10]:
            print(f"   - {fail['filename']} ({fail['class']}): {fail['reason']}")
        if len(failed_images) > 10:
            print(f"   ... dan {len(failed_images) - 10} lainnya")

    if len(df) > 0:
        print("\nJumlah sampel valid per kelas final:")
        print(df['class_name'].value_counts().reindex(
            sorted(df['class_name'].unique(), key=lambda c: FINAL_CLASS_TO_NUM[c])))

    return df, failed_images


if __name__ == "__main__":
    df, failed = main()