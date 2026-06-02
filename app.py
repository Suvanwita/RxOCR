from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import streamlit as st
import torch

from inference import load_checkpoint, predict, preprocess_pil_image
from src.model import CRNN


st.set_page_config(
    page_title="RxOCR Prescription Reader",
    page_icon="Rx",
    layout="wide",
)


@st.cache_resource
def load_model(
    checkpoint_path: str, device_name: str
) -> tuple[CRNN, list[str], int, float, float, torch.device]:
    device = torch.device(device_name)
    checkpoint = load_checkpoint(Path(checkpoint_path), device)

    idx_to_char = checkpoint.get("idx_to_char")
    if idx_to_char is None:
        raise ValueError("Checkpoint must contain idx_to_char to decode predictions")

    saved_args: dict[str, Any] = checkpoint.get("args", {})
    image_height = int(saved_args.get("image_height", 64))
    hidden_size = int(saved_args.get("hidden_size", 256))
    mean = float(saved_args.get("mean", 0.5))
    std = float(saved_args.get("std", 0.5))

    model = CRNN(num_classes=len(idx_to_char), hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_to_char, image_height, mean, std, device


def available_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    st.title("RxOCR Prescription Reader")
    st.caption("Upload a prescription image and run the trained CRNN recognizer.")

    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="checkpoints/best_crnn.pt",
            help="Use the checkpoint produced by train.py.",
        )
        device_name = st.selectbox(
            "Device",
            options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            index=0,
        )
        st.divider()
        st.write("The app uses the same resize and normalization pipeline as inference.py.")

    uploaded_file = st.file_uploader(
        "Upload prescription image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Upload an image to recognize handwritten prescription text.")
        return

    original_image = Image.open(uploaded_file).convert("RGB")

    try:
        model, idx_to_char, image_height, mean, std, device = load_model(
            checkpoint_path, device_name
        )
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    image_tensor, image_width, processed_image = preprocess_pil_image(
        original_image,
        image_height=image_height,
        mean=mean,
        std=std,
    )

    with st.spinner("Recognizing handwriting..."):
        try:
            recognized_text = predict(
                model=model,
                image=image_tensor,
                image_width=image_width,
                idx_to_char=idx_to_char,
                device=device,
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    image_col, processed_col = st.columns(2)
    with image_col:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with processed_col:
        st.subheader("Processed Image")
        st.image(
            processed_image,
            use_container_width=True,
            clamp=True,
            caption=f"Grayscale, resized to height {image_height}px",
        )

    st.subheader("Recognized Text")
    st.text_area(
        "Model output",
        value=recognized_text,
        height=140,
        label_visibility="collapsed",
    )


if __name__ == "__main__":
    main()

