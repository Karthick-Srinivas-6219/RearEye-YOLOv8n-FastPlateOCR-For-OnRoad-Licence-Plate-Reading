import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# -----------------------------
# Load models (cached)
# -----------------------------

@st.cache_resource
def load_models():
    plate_finder = YOLO("plate_v2_35_epochs.pt")
    recognizer = LicensePlateRecognizer('cct-xs-v1-global-model')
    return plate_finder, recognizer

plate_finder, m = load_models()

# -----------------------------
# License plate function
# -----------------------------

def read_licence_plates(img):

    plate_results = plate_finder.predict(
        source=img,
        imgsz=640,
        conf=0.72,
        verbose=False
    )

    for r in plate_results:

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, clss):

            x1, y1, x2, y2 = box.astype(int)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            plate_digits = str(m.run(crop))

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                plate_digits,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    return img


# -----------------------------
# UI
# -----------------------------

st.set_page_config(
    page_title="License Plate Reader",
    layout="centered"
)

st.title("🚓📸 RearEye-YOLOv8n-FastPlateOCR-For-OnRoad-Licence-Plate-Reading")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    st.subheader("Input Image")

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    if st.button("Run Detection"):

        with st.spinner("Processing..."):

            output_img = read_licence_plates(img.copy())

            output_rgb = cv2.cvtColor(
                output_img,
                cv2.COLOR_BGR2RGB
            )

        st.success("Done")

        st.subheader("Output Image")

        st.image(
            output_rgb,
            use_container_width=True
        )