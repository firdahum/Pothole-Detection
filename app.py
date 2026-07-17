import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


st.set_page_config(
    page_title="Deteksi Lubang Jalan",
    page_icon="🛣️",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent

# model yang dipakai
MODEL_FILES = {
    "baseline": "baseline.pt",
    "optuna": "optuna.pt",
}
MODEL_LABELS = {
    "baseline": "Baseline",
    "optuna": "Optuna",
}

# kolom tabel dan csv
DISPLAY_COLUMNS = ["No.", "Kelas", "Keyakinan"]
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


@st.cache_resource(show_spinner=False)
def load_models():
    model_paths = {
        model_key: BASE_DIR / "models" / filename
        for model_key, filename in MODEL_FILES.items()
    }
    missing = [
        f"{MODEL_LABELS[model_key]}: {model_path}"
        for model_key, model_path in model_paths.items()
        if not model_path.exists()
    ]

    if missing:
        raise FileNotFoundError("Model tidak ditemukan:\n" + "\n".join(missing))

    return {
        model_key: YOLO(str(model_path))
        for model_key, model_path in model_paths.items()
    }


def init_state():
    st.session_state.setdefault("hasil_deteksi", [])
    st.session_state.setdefault("zip_hasil", None)


def confidence_text(confidence):
    return "-" if confidence is None else f"{confidence * 100:.2f}%"


def box_from_detection(detection, image_size):
    # ambil koordinat box
    image_width, image_height = image_size
    left = min(image_width - 1, max(0, int(float(detection["X Minimum"]))))
    top = min(image_height - 1, max(0, int(float(detection["Y Minimum"]))))
    right = min(image_width, int(float(detection["X Maksimum"])))
    bottom = min(image_height, int(float(detection["Y Maksimum"])))

    return (
        left,
        top,
        max(left + 1, right),
        max(top + 1, bottom),
    )


def highlight_and_crop(original_image, detection):
    # kasih tanda merah dan crop box
    box = box_from_detection(detection, original_image.size)
    highlighted_image = original_image.copy()

    ImageDraw.Draw(highlighted_image).rectangle(box, outline="red", width=5)

    return highlighted_image, original_image.crop(box)


def image_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def detections_to_csv(detections):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=DETECTION_COLUMNS)
    writer.writeheader()
    writer.writerows(detections)
    return buffer.getvalue().encode("utf-8-sig")


def detection_rows(result, current_model):
    # rapikan output YOLO ke tabel
    rows = []

    if result.boxes is None:
        return rows, None

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x_min, y_min, x_max, y_max = box.xyxy[0].detach().cpu().tolist()

        rows.append({
            "No.": index,
            "Kelas": current_model.names[class_id],
            "Keyakinan": f"{confidence * 100:.2f}%",
            "X Minimum": round(x_min, 2),
            "Y Minimum": round(y_min, 2),
            "X Maksimum": round(x_max, 2),
            "Y Maksimum": round(y_max, 2),
        })

    confidences = [
        float(box.conf[0])
        for box in result.boxes
    ]

    return rows, max(confidences) if confidences else None


def run_detection(current_model, original_image, settings):
    # jalankan deteksi
    result = current_model.predict(
        source=original_image,
        imgsz=settings["image_size"],
        conf=settings["confidence"],
        iou=settings["iou"],
        device=DEVICE,
        verbose=False,
    )[0]
    annotated = Image.fromarray(
        result.plot(
            labels=True,
            conf=True,
            line_width=settings["box_line_width"],
            font_size=settings["label_font_size"],
        )[:, :, ::-1]
    )
    detections, max_confidence = detection_rows(result, current_model)

    return {
        "annotated": annotated,
        "detections": detections,
        "max_confidence": max_confidence,
        "image_bytes": image_bytes(annotated),
        "csv_bytes": detections_to_csv(detections),
    }


def render_sidebar():
    # pengaturan sidebar
    with st.sidebar:
        st.header("Pengaturan Deteksi")

        settings = {
            "confidence": st.slider(
                "Confidence minimum",
                0.10,
                0.90,
                0.40,
                0.05,
                help="Prediksi dengan keyakinan di bawah nilai ini tidak ditampilkan.",
            ),
            "iou": st.slider(
                "IoU threshold",
                0.10,
                0.90,
                0.50,
                0.05,
                help="Digunakan untuk mengurangi bounding box yang saling bertumpuk.",
            ),
            "image_size": st.selectbox(
                "Ukuran gambar inference",
                options=[640, 800],
                index=0,
            ),
            "label_font_size": st.slider("Ukuran label", 8, 24, 10, 1),
            "box_line_width": st.slider("Ketebalan bounding box", 1, 5, 2, 1),
        }

        st.divider()
        st.write(f"Device: **{DEVICE}**")
        st.write("Model baseline: **baseline.pt**")
        st.write("Model Optuna: **optuna.pt**")
        st.write("Kelas: **Pothole**")

        if st.button("Hapus seluruh hasil"):
            st.session_state.hasil_deteksi = []
            st.session_state.zip_hasil = None
            st.rerun()

    return settings


def process_images(uploaded_files, models, settings):
    # proses semua gambar
    all_results = []
    zip_buffer = io.BytesIO()
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_index, uploaded_file in enumerate(uploaded_files, start=1):
            status_text.write(
                f"Memproses gambar {file_index} dari {total_files}: {uploaded_file.name}"
            )

            try:
                original_image = Image.open(uploaded_file).convert("RGB")
                image_stem = Path(uploaded_file.name).stem
                outputs = {}

                for model_key, current_model in models.items():
                    output = run_detection(current_model, original_image, settings)
                    jpg_name = f"{file_index:02d}_{model_key}_hasil_{image_stem}.jpg"
                    csv_name = f"{file_index:02d}_{model_key}_koordinat_{image_stem}.csv"

                    output["download_name"] = jpg_name
                    output["csv_download_name"] = csv_name
                    zip_file.writestr(jpg_name, output["image_bytes"])
                    zip_file.writestr(csv_name, output["csv_bytes"])
                    outputs[model_key] = output

                all_results.append({
                    "filename": uploaded_file.name,
                    "original": original_image.copy(),
                    "outputs": outputs,
                })

            except Exception as error:
                st.warning(f"Gagal memproses {uploaded_file.name}: {error}")

            progress_bar.progress(file_index / total_files)

    st.session_state.hasil_deteksi = all_results
    st.session_state.zip_hasil = zip_buffer.getvalue()
    status_text.success("Seluruh gambar selesai diproses.")


def render_image_comparison(item):
    # tampilkan asli, baseline, optuna
    columns = st.columns(3)

    with columns[0]:
        st.write("**Gambar Asli**")
        st.image(item["original"], use_container_width=True)

    for column, model_key in zip(columns[1:], MODEL_LABELS):
        output = item["outputs"].get(model_key)

        with column:
            st.write(f"**Hasil {MODEL_LABELS[model_key]}**")

            if output is None:
                st.warning("Hasil model tidak tersedia.")
                continue

            st.image(output["annotated"], use_container_width=True)
            st.caption(
                f"{len(output['detections'])} pothole | "
                f"Confidence tertinggi: {confidence_text(output['max_confidence'])}"
            )


def render_detection_detail(item, output, model_key, result_index, settings):
    # detail hasil per model
    if output is None:
        st.warning("Hasil model tidak tersedia.")
        return

    detection_count = len(output["detections"])

    if detection_count == 0:
        st.warning(
            "Tidak ada pothole yang terdeteksi "
            f"dengan confidence minimal {settings['confidence']:.0%}."
        )
    else:
        st.success(
            f"{detection_count} pothole terdeteksi. "
            f"Confidence tertinggi: {confidence_text(output['max_confidence'])}."
        )
        render_detection_picker(item, output, model_key, result_index)

    render_download_buttons(item, output, model_key, result_index)


def render_detection_picker(item, output, model_key, result_index):
    # pilih box kalau mau lihat lokasi
    detection_df = pd.DataFrame(output["detections"], columns=DETECTION_COLUMNS)
    display_df = detection_df[DISPLAY_COLUMNS]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    show_location = st.checkbox(
        "Tampilkan lokasi deteksi",
        key=f"show_location_{model_key}_{result_index}",
    )

    if not show_location:
        return

    selected_number = st.selectbox(
        "Pilih nomor deteksi untuk melihat lokasinya",
        options=detection_df["No."].tolist(),
        format_func=lambda number: f"Deteksi {number}",
        key=f"select_detection_{model_key}_{result_index}",
    )
    selected_detection = detection_df[
        detection_df["No."] == selected_number
    ].iloc[0].to_dict()
    highlighted_image, cropped_image = highlight_and_crop(
        item["original"],
        selected_detection,
    )

    st.info(f"Lokasi deteksi nomor {selected_detection['No.']} ditandai pada gambar.")
    location_column, crop_column = st.columns([2, 1])

    with location_column:
        st.image(
            highlighted_image,
            caption="Lokasi pada gambar asli",
            use_container_width=True,
        )

    with crop_column:
        st.image(
            cropped_image,
            caption="Crop pothole terpilih",
            use_container_width=True,
        )


def render_download_buttons(item, output, model_key, result_index):
    # tombol download
    label = MODEL_LABELS[model_key]

    st.download_button(
        label=f"Unduh hasil {label} {item['filename']}",
        data=output["image_bytes"],
        file_name=output["download_name"],
        mime="image/jpeg",
        key=f"download_{model_key}_{result_index}",
    )
    st.download_button(
        label=f"Unduh data koordinat {label} {item['filename']}",
        data=output["csv_bytes"],
        file_name=output["csv_download_name"],
        mime="text/csv",
        key=f"download_csv_{model_key}_{result_index}",
    )


def render_results(settings):
    # ringkasan hasil
    st.divider()
    st.header("Hasil Deteksi")

    totals = {
        model_key: sum(
            len(item["outputs"][model_key]["detections"])
            for item in st.session_state.hasil_deteksi
            if model_key in item["outputs"]
        )
        for model_key in MODEL_LABELS
    }
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Jumlah gambar", len(st.session_state.hasil_deteksi))
    metric_2.metric("Total pothole Baseline", totals["baseline"])
    metric_3.metric("Total pothole Optuna", totals["optuna"])

    st.download_button(
        label="Unduh Semua Hasil",
        data=st.session_state.zip_hasil,
        file_name="hasil_deteksi_pothole.zip",
        mime="application/zip",
        use_container_width=True,
    )

    for result_index, item in enumerate(st.session_state.hasil_deteksi, start=1):
        st.divider()
        st.subheader(f"{result_index}. {item['filename']}")
        render_image_comparison(item)

        tabs = st.tabs([f"Detail {label}" for label in MODEL_LABELS.values()])
        for tab, model_key in zip(tabs, MODEL_LABELS):
            with tab:
                render_detection_detail(
                    item,
                    item["outputs"].get(model_key),
                    model_key,
                    result_index,
                    settings,
                )


try:
    with st.spinner("Memuat model deteksi..."):
        models = load_models()
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()


init_state()
settings = render_sidebar()

st.title("🛣️ Deteksi Lubang Jalan")
st.write(
    "Unggah satu atau beberapa gambar jalan untuk mendeteksi "
    "keberadaan pothole menggunakan model YOLO11."
)

uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} gambar telah dipilih.")

if st.button(
    "Jalankan Deteksi",
    type="primary",
    disabled=len(uploaded_files) == 0,
):
    process_images(uploaded_files, models, settings)

if st.session_state.hasil_deteksi:
    render_results(settings)
elif not uploaded_files:
    st.info("Silakan unggah gambar untuk memulai deteksi.")
