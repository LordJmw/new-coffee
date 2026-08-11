# utils/dataset_loader.py
"""
Loader untuk dataset SAMPEL yang ditampilkan di sidebar app.py (mode
"📦 Gunakan Dataset Sample"). Berbeda dari extract_features.py: modul ini
TIDAK melakukan ekstraksi fitur, hanya listing file untuk preview UI.

Struktur folder yang diharapkan (default dataset_path: "data/training dataset"):
    data/training dataset/
        <nama folder subkelas>/
            gambar1.jpg
            gambar2.jpg
            ...
Nama folder dicocokkan lewat utils.label_mapping.resolve_folder(), jadi
mendukung nama folder subkelas Kaggle asli ("Full Sour", "Broken", dst.)
maupun variasi penulisan lain yang sudah dikenali di label_mapping.py.
"""

import os

import cv2

from utils.label_mapping import resolve_folder, FINAL_CLASS_TO_NUM

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Grade indikatif untuk tampilan UI - disalin dari CLASS_GRADE_MAP di
# app.py. Didefinisikan ulang di sini (bukan diimpor dari app.py) supaya
# utils/ tidak bergantung balik ke app.py (hindari circular import).
# PENTING: jaga tetap sinkron manual dengan CLASS_GRADE_MAP di app.py.
CLASS_GRADE_MAP = {
    "Normal":                       1,
    "Immature/Discoloration":       2,
    "Sour":                         3,
    "Black":                        3,
    "Physical Damage":              4,
    "Insect Damage":                4,
    "Foreign Material/Processing":  5,
}


def get_dataset_samples(dataset_path):
    """
    Scan dataset_path, kembalikan dict siap pakai untuk dua selectbox
    berantai di sidebar (pilih kelas cacat -> pilih gambar):

        {
            "<nama folder>": {
                "label": "<kelas final 7-kelas>",
                "grade": <int indikatif, 0 jika tidak diketahui>,
                "folder": "<path lengkap folder>",
                "images": ["gambar1.jpg", "gambar2.jpg", ...],
            },
            ...
        }

    Folder yang namanya tidak dikenali resolve_folder() TETAP ditampilkan
    (label = nama foldernya apa adanya, grade = 0) alih-alih disembunyikan,
    karena dataset sample untuk demo UI boleh berisi subset/nama custom di
    luar 18 subkelas Kaggle baku. Folder kosong (tanpa gambar) dilewati.

    Return {} kalau dataset_path tidak ada sama sekali.
    """
    if not os.path.isdir(dataset_path):
        return {}

    samples = {}
    subfolders = sorted(
        d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
    )

    for folder_name in subfolders:
        folder_path = os.path.join(dataset_path, folder_name)
        images = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not images:
            continue

        resolved = resolve_folder(folder_name)
        final_class = resolved[1] if resolved is not None else folder_name

        samples[folder_name] = {
            'label': final_class,
            'grade': CLASS_GRADE_MAP.get(final_class, 0),
            'folder': folder_path,
            'images': images,
        }

    return samples


def get_dataset_statistics(dataset_path):
    """
    Ringkasan jumlah gambar per kelas FINAL (7 kelas) di dataset_path.
    Berguna untuk cek distribusi/imbalance dataset sample lewat dashboard,
    terpisah dari get_dataset_samples() yang fokus untuk keperluan selectbox.

    Return dict {kelas_final: jumlah_gambar}, terurut sesuai urutan baku
    FINAL_CLASS_TO_NUM. Folder yang tidak dikenali label_mapping dihitung
    di bawah key "(Tidak Dikenali)" di posisi terakhir.
    Return {} kalau dataset_path tidak ada.
    """
    if not os.path.isdir(dataset_path):
        return {}

    stats = {}
    subfolders = sorted(
        d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
    )

    for folder_name in subfolders:
        folder_path = os.path.join(dataset_path, folder_name)
        n_images = sum(
            1 for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if n_images == 0:
            continue

        resolved = resolve_folder(folder_name)
        key = resolved[1] if resolved is not None else "(Tidak Dikenali)"
        stats[key] = stats.get(key, 0) + n_images

    def sort_key(item):
        return FINAL_CLASS_TO_NUM.get(item[0], len(FINAL_CLASS_TO_NUM))

    return dict(sorted(stats.items(), key=sort_key))


def load_image_from_dataset(image_path):
    """
    Load satu gambar dari path dataset, return RGB array (atau None jika
    gagal dibaca) - dipakai app.py saat mode "Gunakan Dataset Sample"
    dipilih. Konsisten dengan load_image() di extract_features.py
    (BGR -> RGB, karena cv2.imread selalu membaca BGR).
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)