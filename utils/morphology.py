# utils/morphology.py
import cv2
import numpy as np

def morphological_opening(binary_image, kernel_size=(3, 3), iterations=1):
    """
    M4: Opening (Erosi → Dilasi)
    Menghilangkan noise kecil (pixel putih yang bukan bagian biji)
    """
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening

def morphological_closing(binary_image, kernel_size=(3, 3), iterations=1):
    """
    M4: Closing (Dilasi → Erosi)
    Menyambung tepi yang terputus tanpa menghilangkan detail lubang di dalam.
    Iterasi dikurangi ke 1 agar lubang hama tidak tertutup.
    """
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closing

def apply_morphology(binary_image, open_kernel=(3, 3), close_kernel=(3, 3)):
    """
    Menjalankan pipeline morfologi: Opening → Closing
    """
    # 1. Opening: Membersihkan noise di background
    img_opening = morphological_opening(binary_image, open_kernel, iterations=1)
    
    # 2. Closing: Merapikan bentuk biji
    # Gunakan kernel yang lebih kecil atau iterasi sedikit agar lubang hama tetap terjaga
    img_closing = morphological_closing(img_opening, close_kernel, iterations=1)
    
    return {
        'opening': img_opening,
        'closing': img_closing
    }