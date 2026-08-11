# utils/label_mapping.py
"""
Peta kelas Tahap 2: 17 subkelas cacat + normal (Kaggle, Aset A) -> 7 kelas final.

Sumber pemetaan: Bagian 3, "Ringkasan Arah Skripsi Kopi" (tabel Kelas Final).
Nama folder di sini mengikuti nama folder aktual pada dataset
("17 kelas cacat + normal.zip"), bukan nama subkelas asli Kaggle - keduanya
dipetakan eksplisit lewat FOLDER_TO_SUBCLASS agar tetap tertelusuri ke SNI.
"""

# folder dataset -> (nama subkelas asli, kelas final 7-kelas)
FOLDER_TO_SUBCLASS = {
    "Normal":               ("Normal",                "Normal"),
    "Full Sour":             ("Full Sour",             "Sour"),
    "Partial Sour":          ("Partial Sour",          "Sour"),
    "Full Black":            ("Full Black",            "Black"),
    "Partial Black":         ("Partial Black",         "Black"),
    "Severe Insect Damage":  ("Severe Insect Damage",  "Insect Damage"),
    "Slight Insect Damage":  ("Slight Insect Damage",  "Insect Damage"),
    "Broken":                ("Broken",                "Physical Damage"),
    "Cut":                   ("Cut",                   "Physical Damage"),
    "Shell":                 ("Shell",                 "Physical Damage"),
    "Immature":              ("Immature",              "Immature/Discoloration"),
    "Fade":                  ("Fade",                  "Immature/Discoloration"),
    "Withered":              ("Withered",              "Immature/Discoloration"),
    "Husk":                  ("Husk",                  "Foreign Material/Processing"),
    "Parchment":             ("Parchment",             "Foreign Material/Processing"),
    "Dry Cherry":            ("Dry Cherry",             "Foreign Material/Processing"),
    "Floater":               ("Floater",               "Foreign Material/Processing"),
    "Fungus Damage":         ("Fungus Damage",         "Foreign Material/Processing"),
}

# Urutan kelas final tetap (dipakai untuk label encoding & tampilan laporan/CM)
FINAL_CLASSES = [
    "Normal",
    "Sour",
    "Black",
    "Insect Damage",
    "Physical Damage",
    "Immature/Discoloration",
    "Foreign Material/Processing",
]
 
FINAL_CLASS_TO_NUM = {name: i for i, name in enumerate(FINAL_CLASSES)}
NUM_TO_FINAL_CLASS = {i: name for name, i in FINAL_CLASS_TO_NUM.items()}
 
 
# Reverse index: nama subkelas asli (mis. "Full Sour", "Severe Insect Damage")
# -> kelas final. Dibutuhkan karena beberapa arsip dataset memakai nama folder
# = nama subkelas langsung (tanpa suffix " bean"), mis. train.zip.
SUBCLASS_TO_FINAL = {sub: final for sub, final in FOLDER_TO_SUBCLASS.values()}
SUBCLASS_NAMES_LOOKUP = {sub.lower(): sub for sub, _ in FOLDER_TO_SUBCLASS.values()}
 
 
def _normalize_folder_name(folder_name: str) -> str:
    """Bersihkan variasi penulisan nama folder (spasi, suffix ' bean', kapital)."""
    name = folder_name.strip().lower()
    if name.endswith(" bean"):
        name = name[: -len(" bean")]
    return name.strip()
 
 
def resolve_folder(folder_name: str):
    """
    Cocokkan nama folder dataset ke (subclass_name, final_class), menerima
    dua konvensi penamaan:
      1. "full sour bean"  (kunci asli FOLDER_TO_SUBCLASS)
      2. "Full Sour"       (nama subkelas langsung, mis. pada train.zip)
    Return None jika tidak dikenali.
    """
    # Konvensi 1: cocok langsung ke kunci FOLDER_TO_SUBCLASS (case-insensitive)
    key_lower = folder_name.strip().lower()
    for key, (sub, final) in FOLDER_TO_SUBCLASS.items():
        if key.lower() == key_lower:
            return sub, final
 
    # Konvensi 2: nama folder == nama subkelas asli
    normalized = _normalize_folder_name(folder_name)
    if normalized in SUBCLASS_NAMES_LOOKUP:
        sub = SUBCLASS_NAMES_LOOKUP[normalized]
        return sub, SUBCLASS_TO_FINAL[sub]
 
    return None
 
 
def get_final_class(folder_name: str) -> str:
    """Kembalikan kelas final (7 kelas) dari nama folder dataset."""
    resolved = resolve_folder(folder_name)
    if resolved is None:
        raise KeyError(
            f"Folder '{folder_name}' tidak dikenali. Cek ejaan folder atau "
            f"tambahkan mapping baru di FOLDER_TO_SUBCLASS."
        )
    return resolved[1]
 
 
def get_subclass(folder_name: str) -> str:
    """Kembalikan nama subkelas asli (granular, 17+1) dari nama folder dataset."""
    resolved = resolve_folder(folder_name)
    if resolved is None:
        raise KeyError(f"Folder '{folder_name}' tidak dikenali.")
    return resolved[0]