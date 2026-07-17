import io
import csv
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Deteksi Lubang Jalan",
    page_icon="🛣️",
    layout="wide"
)


# LOKASI MODEL
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "baseline": BASE_DIR / "models" / "baseline.pt",
    "optuna": BASE_DIR / "models" / "optuna.pt",
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "optuna": "Optuna",
}

DISPLAY_COLUMNS = [
    "No.",
    "Kelas",
    "Keyakinan",
]

DETECTION_COLUMNS = [
    "No.",
    "Kelas",
    "Keyakinan",
    "X Minimum",
    "Y Minimum",
    "X Maksimum",
    "Y Maksimum",
]

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# LOAD MODEL
@st.cache_resource(show_spinner=False)
def load_models():
    missing_models = [
        f"{MODEL_LABELS[model_key]}: {model_path}"
        for model_key, model_path in MODEL_PATHS.items()
        if not model_path.exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "Model tidak ditemukan:\n"
            + "\n".join(missing_models)
        )

    return {
        model_key: YOLO(str(model_path))
        for model_key, model_path in MODEL_PATHS.items()
    }


try:
    with st.spinner("Memuat model deteksi..."):
        models = load_models()

except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()


# SESSION STATE
if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = []

if "zip_hasil" not in st.session_state:
    st.session_state.zip_hasil = None


def format_confidence(confidence):
    if confidence is None:
        return "-"

    return f"{confidence * 100:.2f}%"


def get_detection_box(detection, image_size):
    image_width, image_height = image_size

    left = max(0, int(float(detection["X Minimum"])))
    top = max(0, int(float(detection["Y Minimum"])))
    right = min(image_width, int(float(detection["X Maksimum"])))
    bottom = min(image_height, int(float(detection["Y Maksimum"])))

    if right <= left:
        right = min(image_width, left + 1)

    if bottom <= top:
        bottom = min(image_height, top + 1)

    return left, top, right, bottom


def build_selected_detection_images(original_image, detection):
    box = get_detection_box(detection, original_image.size)

    highlighted_image = original_image.copy()
    draw = ImageDraw.Draw(highlighted_image)
    draw.rectangle(box, outline="red", width=5)

    cropped_image = original_image.crop(box)

    return highlighted_image, cropped_image


def build_detection_data(result, current_model):
    detection_data = []
    confidence_values = []

    if result.boxes is not None:
        for box_index, box in enumerate(
            result.boxes,
            start=1
        ):
            class_id = int(box.cls[0])
            class_name = current_model.names[class_id]
            confidence = float(box.conf[0])
            confidence_values.append(confidence)

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

    max_confidence = (
        max(confidence_values)
        if confidence_values
        else None
    )

    return detection_data, max_confidence


def run_detection(
    current_model,
    original_image,
    image_size,
    confidence_threshold,
    iou_threshold,
    label_font_size,
    box_line_width
):
    result = current_model.predict(
        source=original_image,
        imgsz=image_size,
        conf=confidence_threshold,
        iou=iou_threshold,
        device=DEVICE,
        verbose=False
    )[0]

    annotated_array = result.plot(
        labels=True,
        conf=True,
        line_width=box_line_width,
        font_size=label_font_size
    )

    annotated_array = annotated_array[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_array)

    detection_data, max_confidence = build_detection_data(
        result,
        current_model
    )

    image_buffer = io.BytesIO()

    annotated_image.save(
        image_buffer,
        format="JPEG",
        quality=95
    )

    csv_buffer = io.StringIO()
    csv_writer = csv.DictWriter(
        csv_buffer,
        fieldnames=DETECTION_COLUMNS
    )
    csv_writer.writeheader()
    csv_writer.writerows(detection_data)

    return {
        "annotated": annotated_image,
        "detections": detection_data,
        "max_confidence": max_confidence,
        "image_bytes": image_buffer.getvalue(),
        "csv_bytes": csv_buffer.getvalue().encode("utf-8-sig")
    }


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
    st.write("Model baseline: **baseline.pt**")
    st.write("Model Optuna: **optuna.pt**")
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

                model_outputs = {}
                image_stem = Path(uploaded_file.name).stem

                for model_key, current_model in models.items():
                    detection_result = run_detection(
                        current_model=current_model,
                        original_image=original_image,
                        image_size=image_size,
                        confidence_threshold=confidence_threshold,
                        iou_threshold=iou_threshold,
                        label_font_size=label_font_size,
                        box_line_width=box_line_width
                    )

                    output_name = (
                        f"{file_index:02d}_{model_key}_hasil_"
                        f"{image_stem}.jpg"
                    )
                    csv_name = (
                        f"{file_index:02d}_{model_key}_koordinat_"
                        f"{image_stem}.csv"
                    )

                    detection_result["download_name"] = output_name
                    detection_result["csv_download_name"] = csv_name

                    # Memasukkan seluruh hasil ke satu ZIP
                    zip_file.writestr(
                        output_name,
                        detection_result["image_bytes"]
                    )
                    zip_file.writestr(
                        csv_name,
                        detection_result["csv_bytes"]
                    )

                    model_outputs[model_key] = detection_result

                all_results.append({
                    "filename": uploaded_file.name,
                    "original": original_image.copy(),
                    "outputs": model_outputs
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

    total_pothole_by_model = {
        model_key: sum(
            len(item["outputs"][model_key]["detections"])
            for item in st.session_state.hasil_deteksi
            if model_key in item["outputs"]
        )
        for model_key in MODEL_LABELS
    }

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Jumlah gambar",
        len(st.session_state.hasil_deteksi)
    )

    metric_col2.metric(
        "Total pothole Baseline",
        total_pothole_by_model["baseline"]
    )

    metric_col3.metric(
        "Total pothole Optuna",
        total_pothole_by_model["optuna"]
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

        image_columns = st.columns(3)

        with image_columns[0]:
            st.write("**Gambar Asli**")
            st.image(
                item["original"],
                use_container_width=True
            )

        for image_column, model_key in zip(
            image_columns[1:],
            MODEL_LABELS
        ):
            output = item["outputs"].get(model_key)

            with image_column:
                st.write(
                    f"**Hasil {MODEL_LABELS[model_key]}**"
                )

                if output is None:
                    st.warning("Hasil model tidak tersedia.")
                    continue

                st.image(
                    output["annotated"],
                    use_container_width=True
                )

                st.caption(
                    f"{len(output['detections'])} pothole | "
                    "Confidence tertinggi: "
                    f"{format_confidence(output['max_confidence'])}"
                )

        detail_tabs = st.tabs([
            f"Detail {model_label}"
            for model_label in MODEL_LABELS.values()
        ])

        for detail_tab, model_key in zip(
            detail_tabs,
            MODEL_LABELS
        ):
            output = item["outputs"].get(model_key)

            with detail_tab:
                if output is None:
                    st.warning("Hasil model tidak tersedia.")
                    continue

                detection_count = len(output["detections"])

                if detection_count > 0:
                    st.success(
                        f"{detection_count} pothole terdeteksi. "
                        "Confidence tertinggi: "
                        f"{format_confidence(output['max_confidence'])}."
                    )

                    detection_df = pd.DataFrame(
                        output["detections"],
                        columns=DETECTION_COLUMNS
                    )
                    display_detection_df = detection_df[DISPLAY_COLUMNS]

                    st.dataframe(
                        display_detection_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    selected_number = st.selectbox(
                        "Pilih nomor deteksi untuk melihat lokasinya",
                        options=detection_df["No."].tolist(),
                        format_func=lambda number: f"Deteksi {number}",
                        key=f"select_detection_{model_key}_{result_index}"
                    )

                    selected_detection = detection_df[
                        detection_df["No."] == selected_number
                    ].iloc[0].to_dict()

                    highlighted_image, cropped_image = (
                        build_selected_detection_images(
                            item["original"],
                            selected_detection
                        )
                    )

                    st.info(
                        "Lokasi deteksi nomor "
                        f"{selected_detection['No.']} "
                        "ditandai pada gambar."
                    )

                    location_column, crop_column = st.columns([2, 1])

                    with location_column:
                        st.image(
                            highlighted_image,
                            caption="Lokasi pada gambar asli",
                            use_container_width=True
                        )

                    with crop_column:
                        st.image(
                            cropped_image,
                            caption="Crop pothole terpilih",
                            use_container_width=True
                        )

                else:
                    st.warning(
                        "Tidak ada pothole yang terdeteksi "
                        f"dengan confidence minimal "
                        f"{confidence_threshold:.0%}."
                    )

                st.download_button(
                    label=(
                        "Unduh hasil "
                        f"{MODEL_LABELS[model_key]} "
                        f"{item['filename']}"
                    ),
                    data=output["image_bytes"],
                    file_name=output["download_name"],
                    mime="image/jpeg",
                    key=f"download_{model_key}_{result_index}"
                )

                st.download_button(
                    label=(
                        "Unduh data koordinat "
                        f"{MODEL_LABELS[model_key]} "
                        f"{item['filename']}"
                    ),
                    data=output["csv_bytes"],
                    file_name=output["csv_download_name"],
                    mime="text/csv",
                    key=f"download_csv_{model_key}_{result_index}"
                )


# KONDISI AWAL
elif not uploaded_files:
    st.info(
        "Silakan unggah gambar untuk memulai deteksi."
    )

