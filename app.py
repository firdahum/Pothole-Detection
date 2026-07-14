import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Deteksi Lubang Jalan",
    page_icon="🛣️",
    layout="wide"
)


# LOKASI MODEL
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# LOAD MODEL
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


try:
    with st.spinner("Memuat model deteksi..."):
        model = load_model()

except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()


# SESSION STATE
if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = []

if "zip_hasil" not in st.session_state:
    st.session_state.zip_hasil = None


# JUDUL
st.title("🛣️ Deteksi Lubang Jalan")

st.write(
    "Unggah satu atau beberapa gambar jalan untuk mendeteksi "
    "keberadaan pothole menggunakan model YOLO11."
)


# SIDEBAR
with st.sidebar:
    st.header("Pengaturan Deteksi")

    confidence_threshold = st.slider(
        "Confidence minimum",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help=(
            "Prediksi dengan keyakinan di bawah nilai ini "
            "tidak akan ditampilkan."
        )
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Digunakan untuk mengurangi bounding box "
            "yang saling bertumpuk."
        )
    )

    image_size = st.selectbox(
        "Ukuran gambar inference",
        options=[640, 800],
        index=0
    )


    label_font_size = st.slider(
        "Ukuran label",
        min_value=8,
        max_value=24,
        value=10,
        step=1
    )

    box_line_width = st.slider(
        "Ketebalan bounding box",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    st.divider()

    st.write(f"Device: **{DEVICE}**")
    st.write("Model: **YOLO11s**")
    st.write("Kelas: **Pothole**")

    if st.button("Hapus seluruh hasil"):
        st.session_state.hasil_deteksi = []
        st.session_state.zip_hasil = None
        st.rerun()


# UPLOAD BANYAK GAMBAR
uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(
        f"{len(uploaded_files)} gambar telah dipilih."
    )


# TOMBOL DETEKSI
detect_button = st.button(
    "🚀 Jalankan Deteksi",
    type="primary",
    disabled=len(uploaded_files) == 0
)


# PROSES DETEKSI
if detect_button:

    all_results = []
    zip_buffer = io.BytesIO()

    progress_bar = st.progress(0)
    status_text = st.empty()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as zip_file:

        total_files = len(uploaded_files)

        for file_index, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):
            status_text.write(
                f"Memproses gambar {file_index} "
                f"dari {total_files}: {uploaded_file.name}"
            )

            try:
                # Membaca gambar
                original_image = Image.open(
                    uploaded_file
                ).convert("RGB")

                # Menjalankan inference
                result = model.predict(
                    source=original_image,
                    imgsz=image_size,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    device=DEVICE,
                    verbose=False
                )[0]

                # Membuat gambar hasil bounding box
                annotated_array = result.plot(
                    labels=True,
                    conf=True,
                    line_width=box_line_width,
                    font_size=label_font_size
                )

                # BGR menjadi RGB
                annotated_array = annotated_array[:, :, ::-1]

                annotated_image = Image.fromarray(
                    annotated_array
                )

                # Ambil detail deteksi
                detection_data = []

                if result.boxes is not None:
                    for box_index, box in enumerate(
                        result.boxes,
                        start=1
                    ):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])

                        coordinates = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        detection_data.append({
                            "No.": box_index,
                            "Kelas": class_name,
                            "Keyakinan": (
                                f"{confidence * 100:.2f}%"
                            ),
                            "X Minimum": round(
                                coordinates[0], 2
                            ),
                            "Y Minimum": round(
                                coordinates[1], 2
                            ),
                            "X Maksimum": round(
                                coordinates[2], 2
                            ),
                            "Y Maksimum": round(
                                coordinates[3], 2
                            )
                        })

                # Simpan hasil menjadi JPG di memori
                image_buffer = io.BytesIO()

                annotated_image.save(
                    image_buffer,
                    format="JPEG",
                    quality=95
                )

                image_bytes = image_buffer.getvalue()

                output_name = (
                    f"{file_index:02d}_hasil_"
                    f"{Path(uploaded_file.name).stem}.jpg"
                )

                # Memasukkan seluruh hasil ke satu ZIP
                zip_file.writestr(
                    output_name,
                    image_bytes
                )

                all_results.append({
                    "filename": uploaded_file.name,
                    "original": original_image.copy(),
                    "annotated": annotated_image.copy(),
                    "detections": detection_data,
                    "download_name": output_name,
                    "image_bytes": image_bytes
                })

            except Exception as error:
                st.warning(
                    f"Gagal memproses {uploaded_file.name}: "
                    f"{error}"
                )

            progress_bar.progress(
                file_index / total_files
            )

    st.session_state.hasil_deteksi = all_results
    st.session_state.zip_hasil = zip_buffer.getvalue()

    status_text.success(
        "Seluruh gambar selesai diproses."
    )


# TAMPILKAN HASIL
if st.session_state.hasil_deteksi:

    st.divider()
    st.header("Hasil Deteksi")

    total_pothole = sum(
        len(item["detections"])
        for item in st.session_state.hasil_deteksi
    )

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Jumlah gambar",
        len(st.session_state.hasil_deteksi)
    )

    metric_col2.metric(
        "Total pothole terdeteksi",
        total_pothole
    )

    # Download seluruh hasil
    st.download_button(
        label="⬇️ Unduh Semua Hasil",
        data=st.session_state.zip_hasil,
        file_name="hasil_deteksi_pothole.zip",
        mime="application/zip",
        use_container_width=True
    )

    for result_index, item in enumerate(
        st.session_state.hasil_deteksi,
        start=1
    ):
        st.divider()

        st.subheader(
            f"{result_index}. {item['filename']}"
        )

        original_column, result_column = st.columns(2)

        with original_column:
            st.write("**Gambar Asli**")
            st.image(
                item["original"],
                use_container_width=True
            )

        with result_column:
            st.write("**Hasil Deteksi**")
            st.image(
                item["annotated"],
                use_container_width=True
            )

        detection_count = len(item["detections"])

        if detection_count > 0:
            st.success(
                f"{detection_count} pothole terdeteksi."
            )

            detection_df = pd.DataFrame(
                item["detections"]
            )

            st.dataframe(
                detection_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "Tidak ada pothole yang terdeteksi "
                f"dengan confidence minimal "
                f"{confidence_threshold:.0%}."
            )

        st.download_button(
            label=f"⬇️ Unduh hasil {item['filename']}",
            data=item["image_bytes"],
            file_name=item["download_name"],
            mime="image/jpeg",
            key=f"download_{result_index}"
        )


# KONDISI AWAL
elif not uploaded_files:
    st.info(
        "Silakan unggah gambar untuk memulai deteksi."
    )

