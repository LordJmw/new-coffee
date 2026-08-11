import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import joblib
import pandas as pd

# Import modul internal
from utils.dataset_loader import get_dataset_samples, get_dataset_statistics, load_image_from_dataset
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features_combine import extract_all_features, ALL_FEATURE_NAMES
from utils.label_mapping import FINAL_CLASSES
from utils.clustering import analyze_color_kmeans
from utils.visualization import create_decision_boundary_plot

# Page config
st.set_page_config(
    page_title="Klasifikasi Biji Kopi",
    page_icon="☕",
    layout="wide"
)

# ============================================
# KONFIGURASI MODEL RF 7-KELAS (BARU)
# ============================================
MODEL_PATH = "outputs/coffee_bean_rf_7class.pkl"   # hasil joblib.dump di train_evaluate_rf.py
FEATURES_CSV_PATH = "data/features_dataset_7kelas.csv"  # hasil extract_features.py

# Grade sederhana untuk tampilan UI saja (BUKAN skema resmi SNI 01-2907-2008,
# yang sebenarnya berbasis akumulasi skor cacat per 100g sampel, bukan
# per-biji tunggal). Sesuaikan urutan/level ini dengan tabel grading final
# di skripsi bila sudah ditentukan.
CLASS_GRADE_MAP = {
    "Normal":                       1,
    "Immature/Discoloration":       2,
    "Sour":                         3,
    "Black":                        3,
    "Physical Damage":              4,
    "Insect Damage":                4,
    "Foreign Material/Processing":  5,
}


@st.cache_resource
def load_ml_model():
    """Load bundle {'model', 'feature_columns', 'classes'} hasil training RF baru."""
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle['model'], bundle['feature_columns']
    except FileNotFoundError:
        return None, None


ml_model, ml_feature_columns = load_ml_model()

# Header
st.title("☕ Klasifikasi Mutu Biji Kopi Mentah")
st.markdown("### Ekstraksi Fitur Morfologi & Geometri dengan Preprocessing Citra Digital")

# Inisialisasi session state
if 'input_image' not in st.session_state:
    st.session_state['input_image'] = None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("📋 Menu")
    input_mode = st.radio("Pilih sumber gambar:", ["📤 Upload Gambar Sendiri", "📦 Gunakan Dataset Sample"])
    st.divider()

    # ========== Mode Klasifikasi ==========
    st.subheader("Metode Klasifikasi")
    if ml_model is not None:
        classification_mode = st.radio(
            "Pilih metode:",
            ["Rule-based", "Machine Learning"],
            help="Rule-based: threshold manual dari makalah | ML: Random Forest 7-kelas (SNI-aligned)"
        )
    else:
        classification_mode = "Rule-based"
        st.warning(f"⚠️ Model ML belum tersedia di `{MODEL_PATH}`. Jalankan "
                   f"extract_features.py lalu train_evaluate_rf.py dulu.")

    st.subheader("⚙️ Parameter Preprocessing")
    resize_option = st.selectbox("Ukuran Resize", ["224x224 (Rekomendasi)", "256x256", "512x512"], index=0)
    t_size = int(resize_option.split('x')[0])
    target_size = (t_size, t_size)

    interpolation_option = st.selectbox("Metode Interpolasi",
        ["Nearest Neighbor", "Bilinear interpolation", "Bicubic interpolation", "Area-based", "Lanczos"], index=3)

    blur_kernel = st.slider("Gaussian Blur Kernel", 3, 11, 5, step=2)
    open_kernel = st.slider("Opening Kernel", 2, 7, 3)
    close_kernel = st.slider("Closing Kernel", 3, 9, 3)
    normalize_illum = st.checkbox("Normalisasi Iluminasi (white-balance + CLAHE)", value=True,
                                   help="Direkomendasikan aktif - ini fitur baru untuk domain adaptation.")

    st.divider()

    if input_mode == "📦 Gunakan Dataset Sample":
        st.subheader("📂 Pilih Sampel Dataset")
        dataset_path = "data/train"
        samples = get_dataset_samples(dataset_path)

        if samples:
            selected_class = st.selectbox("Pilih kelas cacat:", list(samples.keys()))
            class_info = samples[selected_class]
            st.info(f"Label: {class_info['label']} | Grade: {class_info['grade']}")

            selected_image = st.selectbox("Pilih gambar:", class_info['images'])
            if selected_image:
                image_path = os.path.join(class_info['folder'], selected_image)
                st.session_state['sample_image_path'] = image_path
        else:
            st.warning("⚠️ Dataset tidak ditemukan!")


# ============================================
# FUNGSI PREDIKSI ML (RF 7-KELAS, TANPA SCALER)
# ============================================
def predict_with_ml(features, model, feature_columns):
    """
    Prediksi menggunakan Random Forest 7-kelas.
    PENTING: urutan fitur HARUS sama persis dengan feature_columns yang
    disimpan bersama model (= urutan ALL_FEATURE_NAMES saat training).
    RandomForest tidak butuh scaling, jadi tidak ada StandardScaler di sini
    (berbeda dari model 6-kelas lama).
    """
    X = np.array([[features.get(fname, 0.0) for fname in feature_columns]])

    pred_class = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = max(proba) * 100
    grade = CLASS_GRADE_MAP.get(pred_class, 0)

    # urutan proba mengikuti model.classes_, bukan feature_columns
    class_order = list(model.classes_)

    return pred_class, confidence, proba, grade, class_order


# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([1, 1])

with col1:
    if input_mode == "📤 Upload Gambar Sendiri":
        uploaded_file = st.file_uploader("Upload gambar biji kopi", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_container_width=True)
            st.session_state['input_image'] = np.array(image)
    else:
        if 'sample_image_path' in st.session_state:
            image_rgb = load_image_from_dataset(st.session_state['sample_image_path'])
            if image_rgb is not None:
                st.image(image_rgb, caption=f"Sampel: {os.path.basename(st.session_state['sample_image_path'])}", use_container_width=True)
                st.session_state['input_image'] = image_rgb

with col2:
    if st.session_state['input_image'] is not None:
        img = st.session_state['input_image']
        with st.spinner("Memproses analisis..."):
            try:
                # M1-M3: Preprocessing (+ normalisasi iluminasi, fitur baru)
                preprocess_results = preprocess_pipeline(
                    img,
                    target_size=target_size,
                    blur_kernel=(blur_kernel, blur_kernel),
                    interpolation_method=interpolation_option,
                    normalize_illum=normalize_illum,
                )

                # M4: Morphology
                morph_results = apply_morphology(
                    preprocess_results['binary'],
                    open_kernel=(open_kernel, open_kernel),
                    close_kernel=(close_kernel, close_kernel)
                )

                # M6: Feature Extraction - signature BARU: (binary, gray, rgb)
                # bukan (binary, blur, rgb, gray) seperti versi lama.
                geom_features = extract_all_features(
                    morph_results['closing'],
                    preprocess_results['gray'],
                    preprocess_results['rgb'],
                )

                # Fitur baru tidak lagi mengembalikan 'contour' / 'cropped_rgb'
                # langsung sebagai key utama - direkonstruksi di sini.
                main_contour = geom_features.get('_contour')
                hole_contours = geom_features.get('_hole_contours', [])
                bbox = None
                cropped_rgb = None
                if geom_features['is_valid'] and main_contour is not None:
                    bx, by, bw, bh = cv2.boundingRect(main_contour)
                    bbox = (bx, by, bw, bh)
                    cropped_rgb = preprocess_results['rgb'][by:by + bh, bx:bx + bw]

                # Visualisasi overlay geometri
                geo_overlay = preprocess_results['rgb'].copy()
                if geom_features['is_valid'] and main_contour is not None:
                    # Kontur utama (Hijau)
                    cv2.drawContours(geo_overlay, [main_contour], -1, (0, 255, 0), 2)

                    # Lubang terdeteksi (Merah)
                    for hc in hole_contours:
                        cv2.drawContours(geo_overlay, [hc], -1, (255, 50, 50), 2)

                    # Bounding Box
                    bx, by, bw, bh = bbox
                    cv2.rectangle(geo_overlay, (bx, by), (bx + bw, by + bh), (0, 200, 200), 1)

                # M7: K-Means
                kmeans_img, color_centers, cluster_stats = analyze_color_kmeans(
                    cropped_rgb, k=3
                ) if geom_features['is_valid'] and cropped_rgb is not None else (None, None, None)

                if classification_mode == "Machine Learning":
                    tabs = st.tabs(["Proses Citra", "Hasil Ekstraksi", "Analisis Warna", "Penilaian Akhir", "Penilaian Model"])
                else:
                    tabs = st.tabs(["Proses Citra", "Hasil Ekstraksi", "Analisis Warna", "Penilaian Akhir"])

                # ─── TAB 0: Proses Citra ───────────────────────────────────────────
                with tabs[0]:
                    st.subheader("🔍 Deteksi Geometri & Morfologi")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(geo_overlay,
                                 caption=f"Representasi Geometri (Tepi & Lubang) — {interpolation_option}",
                                 use_container_width=True)
                    with c2:
                        st.image(morph_results['closing'],
                                 caption="Binary Mask (Hasil Morfologi)",
                                 use_container_width=True)

                    st.info(
                        "💡 **Keterangan:** Garis **Hijau** = Tepi Luar Biji | "
                        "Garis **Merah** = Lubang/Cacat | "
                        "Kotak **Cyan** = Bounding Box"
                    )

                    if geom_features['is_valid'] and geom_features.get('holes_count', 0) > 0:
                        with st.expander("🔬 Debug: Info Lubang Terdeteksi"):
                            st.caption(f"Jumlah lubang: {int(geom_features['holes_count'])}")
                            st.caption(f"Rasio luas lubang thd biji: {geom_features['holes_area_ratio']*100:.1f}%")
                            st.caption(f"Rata-rata luas lubang: {geom_features['holes_mean_area']:.0f} px²")
                            st.caption(f"Circularity lubang (rata-rata): {geom_features['holes_circularity_mean']:.2f}")

                    st.divider()
                    st.markdown("**Langkah-Langkah Intermediate:**")
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.image(preprocess_results['gray'], caption="1. Grayscale")
                    c_m2.image(preprocess_results['blur'], caption="2. Noise Reduction")
                    if cropped_rgb is not None:
                        c_m3.image(cropped_rgb, caption="3. Hasil Crop (ROI)")

                # ─── TAB 1: Hasil Ekstraksi ────────────────────────────────────────
                with tabs[1]:
                    st.subheader("📊 Parameter Geometri Biji")
                    if geom_features['is_valid']:
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric(label="Luas Biji (Area)", value=f"{geom_features['area']:.0f} px")
                            st.caption("Total piksel tubuh biji.")
                        with m2:
                            sol_pct = geom_features['solidity'] * 100
                            st.metric(label="Kepadatan (Solidity)", value=f"{sol_pct:.1f}%")
                            st.caption("Mendeteksi kerutan/withered.")
                        with m3:
                            st.metric(label="Bentuk (Circularity)", value=f"{geom_features['circularity']:.2f}")
                            st.caption("Nilai 1.0 = Bulat Sempurna.")

                        st.divider()
                        m4, m5, m6 = st.columns(3)
                        with m4:
                            st.metric(label="Rasio Panjang (AR)", value=f"{geom_features['aspect_ratio']:.2f}")
                            st.caption("Panjang vs Lebar (Broken detection).")
                        with m5:
                            ext_pct = geom_features['extent'] * 100
                            st.metric(label="Kepenuhan (Extent)", value=f"{ext_pct:.1f}%")
                            st.caption("Persentase pengisian bounding box.")
                        with m6:
                            hole_count = geom_features.get('holes_count', 0)
                            st.metric(
                                label="Lubang Serangga",
                                value=f"{int(hole_count)} Titik",
                                delta="Terdeteksi" if hole_count > 0 else "Aman",
                                delta_color="inverse" if hole_count > 0 else "normal"
                            )
                            st.caption("Dual-track: rongga besar + titik gelap.")

                        st.divider()
                        st.markdown("**Fitur tekstur & warna tambahan (baru):**")
                        w1, w2, w3 = st.columns(3)
                        with w1:
                            st.metric(label="Edge Density", value=f"{geom_features.get('edge_density', 0):.3f}")
                            st.caption("Proxy retakan permukaan (pengganti center_cut_lines).")
                        with w2:
                            st.metric(label="Hue Mean", value=f"{geom_features.get('hue_mean', 0):.1f}")
                            st.caption("Warna dominan (HSV, tahan-iluminasi).")
                        with w3:
                            st.metric(label="Patchiness (L*)", value=f"{geom_features.get('patch_L_std', 0):.2f}")
                            st.caption("Ketidakmerataan warna permukaan.")
                    else:
                        st.error("⚠️ Objek tidak terdeteksi dengan jelas.")

                # ─── TAB 2: Analisis Warna ─────────────────────────────────────────
                with tabs[2]:
                    if kmeans_img is not None:
                        st.image(kmeans_img, caption="Segmentasi Warna K-Means", use_container_width=True)
                        cols = st.columns(len(cluster_stats))

                        for i, (idx, stat) in enumerate(cluster_stats.items()):
                            with cols[i]:
                                r, g, b = stat['color']

                                if r > 185 and g > 185 and b > 185:
                                    label = "⚪ Glare"
                                    color_theme = "gray"
                                elif r / (g if g > 0 else 1) > 1.28:
                                    label = "🟤 Sour"
                                    color_theme = "red"
                                elif np.std([r, g, b]) < 22:
                                    label = "🔘 Pucat"
                                    color_theme = "orange"
                                else:
                                    label = "🟢 Normal"
                                    color_theme = "green"

                                color_box = np.zeros((50, 100, 3), dtype=np.uint8)
                                color_box[:] = [r, g, b]
                                st.image(color_box, caption=f"Klaster {idx + 1}")
                                st.write(f"**{stat['percentage']:.1f}%** area")
                                st.caption(f"RGB: {list(stat['color'])}")
                                st.markdown(f":{color_theme}[**{label}**]")
                    else:
                        st.warning("⚠️ Klaster warna tidak muncul. Pastikan kontur biji terdeteksi.")

                # ─── TAB 3: Penilaian Akhir ────────────────────────────────────────
                with tabs[3]:
                    if geom_features['is_valid']:

                        # ==========================================
                        # 1. JALANKAN KEDUA METODE (BACKEND)
                        # ==========================================

                        # --- A. Prediksi Machine Learning (RF 7-kelas) ---
                        if ml_model is not None:
                            pred_class, confidence, proba, grade, class_order = predict_with_ml(
                                geom_features, ml_model, ml_feature_columns
                            )

                        # --- B. Prediksi Rule-Based (tetap logika lama, independen dari RF) ---
                        current_score = 100
                        logs = []
                        detected_class = None

                        # ── 1. DRY CHERRY / kegelapan ekstrem ──────────
                        intensity = geom_features.get('mean_intensity', 127)
                        if intensity < 95:
                            penalty = 75
                            current_score -= penalty
                            detected_class = "DRY_CHERRY"
                            logs.append(f"❌ **Intensitas:** Sangat gelap (Inten: {intensity:.0f}) [-{penalty}]")

                        # ── 2. BROKEN / Physical Damage ────────────────
                        if detected_class is None:
                            aspect_ratio = geom_features.get('aspect_ratio', 1.0)
                            extent = geom_features.get('extent', 0.8)
                            if aspect_ratio > 1.7 or aspect_ratio < 0.65 or extent < 0.68:
                                penalty = 75
                                current_score -= penalty
                                detected_class = "PHYSICAL_DAMAGE"
                                logs.append(f"❌ **Fisik:** Broken/Pecah (AR: {aspect_ratio:.2f}) [-{penalty}]")

                        # ── 3. SOUR ─────────────────────────────────────
                        if detected_class is None and cluster_stats is not None:
                            valid_ratios = []
                            for idx, stat in cluster_stats.items():
                                r, g, b = stat['color']
                                if r > 185 and g > 185 and b > 185:
                                    continue
                                valid_ratios.append(r / g if g > 0 else 1.0)
                            if valid_ratios:
                                max_rg = max(valid_ratios)
                                if max_rg > 1.28:
                                    penalty = 50
                                    current_score -= penalty
                                    detected_class = "SOUR"
                                    logs.append(f"⚠️ **Warna:** Sour (R/G: {max_rg:.2f}) [-{penalty}]")

                        # ── 4. IMMATURE/DISCOLORATION (withered-like) ──
                        if detected_class is None:
                            intensity_w = geom_features.get('mean_intensity', 127)
                            circ_w = geom_features.get('circularity', 1.0)
                            sol_w = geom_features.get('solidity', 1.0)
                            ar_w = geom_features.get('aspect_ratio', 1.0)

                            mask_w = np.zeros(preprocess_results['rgb'].shape[:2], dtype=np.uint8)
                            if main_contour is not None:
                                cv2.drawContours(mask_w, [main_contour], -1, 255, -1)
                            rgb_w = preprocess_results['rgb']
                            g_w = float(np.mean(rgb_w[:, :, 1][mask_w == 255])) if np.any(mask_w == 255) else 130
                            b_w = float(np.mean(rgb_w[:, :, 2][mask_w == 255])) if np.any(mask_w == 255) else 90
                            gb_ratio = g_w / b_w if b_w > 0 else 1.5

                            withered_signals = 0
                            withered_detail = []

                            if gb_ratio < 1.40 and intensity_w > 140:
                                withered_signals += 3
                                withered_detail.append(f"G/B={gb_ratio:.3f} & I={intensity_w:.0f}")
                            elif gb_ratio < 1.38 and intensity_w > 132:
                                withered_signals += 2
                                withered_detail.append(f"G/B={gb_ratio:.3f} & I={intensity_w:.0f}(border)")
                            elif intensity_w > 148:
                                withered_signals += 2
                                withered_detail.append(f"I={intensity_w:.0f}>148")

                            if circ_w < 0.80:
                                withered_signals += 1
                                withered_detail.append(f"Circ={circ_w:.3f}")
                            if sol_w < 0.985:
                                withered_signals += 1
                                withered_detail.append(f"Sol={sol_w:.3f}")
                            if ar_w >= 0.88:
                                withered_signals += 1
                                withered_detail.append(f"AR={ar_w:.2f}(port)")

                            if withered_signals >= 3 and current_score > 70:
                                penalty = 30
                                current_score -= penalty
                                detected_class = "IMMATURE_DISCOLORATION"
                                logs.append(f"⚠️ **Fisik+Warna:** Immature/Discoloration ({', '.join(withered_detail)}) [-{penalty}]")

                        # ── 5. INSECT DAMAGE ────────────────────────────
                        holes = geom_features.get('holes_count', 0)
                        if holes > 0:
                            if holes >= 2:
                                penalty = 90
                                logs.append(f"❌ **Hama:** Insect Damage berat ({int(holes)} lubang) [-{penalty}]")
                            else:
                                penalty = 85
                                logs.append(f"❌ **Hama:** Insect Damage (1 lubang terdeteksi) [-{penalty}]")
                            current_score -= penalty

                        # ── Finalisasi ─────────────────────────────────
                        final_score = max(0, current_score)

                        if final_score >= 88:
                            rule_grade_text, rule_grade_num = "NORMAL", 1
                        elif final_score >= 60:
                            rule_grade_text, rule_grade_num = "IMMATURE/DISCOLORATION", 2
                        elif final_score >= 40:
                            rule_grade_text, rule_grade_num = "SOUR", 3
                        elif final_score >= 20:
                            rule_grade_text, rule_grade_num = "PHYSICAL DAMAGE", 4
                        else:
                            rule_grade_text, rule_grade_num = "INSECT DAMAGE", 5

                        # ==========================================
                        # 2. TAMPILAN UTAMA
                        # ==========================================
                        if classification_mode == "Machine Learning" and ml_model is not None:
                            st.subheader(f"Hasil Klasifikasi ML (Random Forest 7-Kelas): **{pred_class}**")

                            col_score, col_grade = st.columns(2)
                            with col_score:
                                st.metric("Confidence", f"{confidence:.1f}%")
                            with col_grade:
                                st.metric("Grade (indikatif)", grade)

                            st.caption("📊 Distribusi Probabilitas per Kelas:")
                            for name, prob in zip(class_order, proba):
                                bar_length = int(prob * 30)
                                bar = "█" * bar_length + "░" * (30 - bar_length)
                                st.caption(f"   {name:<28}: {bar} {prob*100:.1f}%")

                        else:
                            st.subheader(f"Total Skor Akhir: {final_score}")

                            text = f"### HASIL: GRADE {rule_grade_num} ({rule_grade_text})"
                            if rule_grade_num == 1:
                                st.success(text)
                            elif rule_grade_num == 2:
                                st.info(text)
                            elif rule_grade_num == 3:
                                st.warning(text)
                            else:
                                st.error(text)

                            for log in logs:
                                st.write(log)

                        # ==========================================
                        # 3. KOMPARASI (jika ML tersedia)
                        # ==========================================
                        if ml_model is not None:
                            with st.expander("🔍 Analisis: Bandingkan Rule-Based vs Machine Learning"):
                                c_rule, c_ml = st.columns(2)

                                with c_rule:
                                    st.markdown("#### 📐 Rule-Based")
                                    st.metric("Grade", rule_grade_num)
                                    st.write(f"**Kelas:** {rule_grade_text.title()}")
                                    st.write(f"**Skor:** {final_score}/100")
                                    st.markdown("**Jejak Penalti:**")
                                    if len(logs) == 0:
                                        st.caption("- Tidak ada penalti")
                                    else:
                                        for log in logs:
                                            st.caption(f"- {log}")

                                with c_ml:
                                    st.markdown("#### 🤖 Machine Learning")
                                    st.metric("Grade", grade)
                                    st.write(f"**Kelas:** {pred_class}")
                                    st.write(f"**Confidence:** {confidence:.1f}%")
                                    st.markdown("**Top Probabilitas:**")
                                    prob_dict = dict(zip(class_order, proba))
                                    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)[:3])
                                    for name, p in sorted_probs.items():
                                        st.caption(f"- {name}: {p*100:.1f}%")

                    else:
                        st.error("⚠️ Objek tidak terdeteksi dengan jelas.")

                # ─── TAB 4: Penilaian Model ─────────────────────────────────────────
                if classification_mode == "Machine Learning" and ml_model is not None:
                    with tabs[4]:
                        st.subheader("📈 Analisis Model Dinamis")
                        st.write("Confusion matrix & profil fitur dihitung dari dataset fitur 7-kelas.")

                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        from sklearn.metrics import confusion_matrix

                        if os.path.exists(FEATURES_CSV_PATH):
                            df_dataset = pd.read_csv(FEATURES_CSV_PATH)
                            df_dataset = df_dataset[df_dataset['is_valid'] == True]  # noqa: E712

                            # Maks 30 sampel per kelas agar heatmap tetap ringan
                            df_eval = df_dataset.groupby('class_name').head(30)

                            missing_cols = [c for c in ml_feature_columns if c not in df_eval.columns]
                            if missing_cols:
                                st.error(f"CSV fitur tidak cocok dengan model (kolom hilang: {missing_cols[:5]}...). "
                                         f"Jalankan ulang extract_features.py dengan utils versi terbaru.")
                            else:
                                X_eval = df_eval[ml_feature_columns].values
                                y_eval = df_eval['class_name'].values
                                y_pred_eval = ml_model.predict(X_eval)

                                labels_order = [c for c in FINAL_CLASSES if c in set(y_eval) | set(y_pred_eval)]
                                cm = confusion_matrix(y_eval, y_pred_eval, labels=labels_order)

                                st.markdown(f"#### 🧱 Confusion Matrix ({len(df_eval)} sampel dari CSV fitur)")
                                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                                sns.heatmap(
                                    cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=True,
                                    square=True, linewidths=1, linecolor='gray',
                                    xticklabels=labels_order, yticklabels=labels_order
                                )
                                ax_cm.set_title('Confusion Matrix - Random Forest (7 Kelas)\n', fontsize=14)
                                ax_cm.set_xlabel('Predicted Label', fontsize=10)
                                ax_cm.set_ylabel('True Label', fontsize=10)
                                ax_cm.tick_params(axis='x', rotation=45)
                                ax_cm.tick_params(axis='y', rotation=0)
                                st.pyplot(fig_cm)

                                st.divider()
                                st.markdown("#### 🗺️ Decision Boundary (Surrogate, Proyeksi PCA 2D)")
                                st.caption(
                                    "Model asli dilatih di ~90 dimensi fitur, jadi batas keputusan "
                                    "di bawah ini adalah APROKSIMASI dari surrogate RF yang dilatih "
                                    "ulang pada 2 komponen PCA - bukan representasi eksak model "
                                    "produksi. Dipakai untuk eksplorasi visual seberapa terpisah "
                                    "ke-7 kelas dalam ruang fitur, bukan bukti akurasi."
                                )
                                fig_boundary = create_decision_boundary_plot(
                                    X_eval, y_eval, classes=labels_order
                                )
                                st.pyplot(fig_boundary)
                        else:
                            st.error(f"⚠️ `{FEATURES_CSV_PATH}` tidak ditemukan. Jalankan extract_features.py dulu.")

                        st.divider()

                        st.markdown("#### 📊 Profil Fitur Biji Kopi Saat Ini")
                        if geom_features['is_valid']:
                            # Tampilkan subset fitur paling interpretable saja (bukan
                            # ~90 fitur mentah RF) untuk keterbacaan dashboard.
                            display_feats = {
                                'area': geom_features.get('area', 0),
                                'circularity': geom_features.get('circularity', 0),
                                'solidity': geom_features.get('solidity', 0),
                                'extent': geom_features.get('extent', 0),
                                'aspect_ratio': geom_features.get('aspect_ratio', 0),
                                'holes_count': geom_features.get('holes_count', 0),
                                'edge_density': geom_features.get('edge_density', 0),
                                'mean_intensity': geom_features.get('mean_intensity', 0),
                                'red_ratio': geom_features.get('red_ratio', 0),
                                'green_ratio': geom_features.get('green_ratio', 0),
                                'hue_mean': geom_features.get('hue_mean', 0),
                                'patch_L_std': geom_features.get('patch_L_std', 0),
                            }
                            df_feat = pd.DataFrame({
                                'Fitur': list(display_feats.keys()),
                                'Nilai': list(display_feats.values())
                            }).sort_values(by='Nilai', ascending=True)

                            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                            ax_fi.barh(df_feat['Fitur'], df_feat['Nilai'], color='#1f77b4', align='center')
                            ax_fi.set_xlabel('Nilai Kuantitatif (Skala Logaritmik)')
                            ax_fi.set_title('Profil Fitur Biji Kopi Saat Ini (subset interpretable)')
                            for spine in ax_fi.spines.values():
                                spine.set_color('black')
                                spine.set_linewidth(1)
                            ax_fi.set_xscale('symlog')
                            plt.tight_layout()
                            st.pyplot(fig_fi)

            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")

st.divider()