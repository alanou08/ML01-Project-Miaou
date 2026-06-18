"""Streamlit interface for the PetSpeak classifier."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.audio_processing import AudioProcessingError, load_audio_bytes
from src.config import (
    CANONICAL_CLASSES,
    DISPLAY_CLASS_NAMES,
    HOP_LENGTH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    N_FFT,
    N_MELS,
    PROJECT_ROOT,
    SAMPLE_RATE,
    VALID_AUDIO_EXTENSIONS,
)
from src.feature_extraction import FeatureExtractionError, extract_features
from src.predict import load_saved_model, ordered_probabilities
from src.translation_generator import DISCLAIMER, generate_translation

st.set_page_config(
    page_title="PetSpeak",
    page_icon="🐈",
    layout="wide",
)

MEME_IMAGE_PATHS = {
    "waiting_for_food": (
        PROJECT_ROOT / "assets" / "memes" / "waiting_for_food.jpg"
    ),
    "isolation": (
        PROJECT_ROOT / "assets" / "memes" / "isolation.jpg"
    ),
    "brushing": (
        PROJECT_ROOT / "assets" / "memes" / "brushing.jpg"
    ),
}


@st.cache_resource(show_spinner=False)
def cached_model() -> tuple[Any, dict[str, Any]]:
    """Load the model once per Streamlit server process."""
    return load_saved_model(MODEL_PATH, MODEL_METADATA_PATH)


@st.cache_data(show_spinner=False)
def analyze_uploaded_audio(
    file_bytes: bytes,
    suffix: str,
) -> tuple[np.ndarray, dict[str, int | float | bool], np.ndarray]:
    """Decode and featurize uploaded bytes through the shared pipeline."""
    audio, info = load_audio_bytes(file_bytes, suffix=suffix)
    features = extract_features(audio)
    return audio, info.to_dict(), features


def waveform_figure(audio: np.ndarray) -> plt.Figure:
    """Create a readable waveform figure."""
    times = np.arange(audio.size) / SAMPLE_RATE
    figure, axis = plt.subplots(figsize=(9.0, 3.0))
    axis.plot(times, audio, linewidth=0.8)
    axis.set_title("Preprocessed waveform (3 seconds)")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Normalized amplitude")
    axis.set_xlim(0, times[-1] if times.size else 3.0)
    figure.tight_layout()
    return figure


def mel_spectrogram_figure(audio: np.ndarray) -> plt.Figure:
    """Create a mel-spectrogram figure for the preprocessed recording."""
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    figure, axis = plt.subplots(figsize=(9.0, 3.8))
    image = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axis,
    )
    axis.set_title("Mel spectrogram")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (mel scale)")
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    figure.tight_layout()
    return figure


def main() -> None:
    st.title("🐈 PetSpeak")
    st.subheader("Cat Vocalization Context Classifier")
    st.write(
        "Upload a cat vocalization to estimate whether its acoustic pattern is "
        "most consistent with waiting for food, isolation, or brushing."
    )
    st.info(DISCLAIMER)

    if not MODEL_PATH.is_file():
        st.error(
            "No trained model was found. Prepare the dataset and run "
            "`python -m src.build_metadata`, then `python -m src.train`."
        )
        st.stop()

    try:
        model, model_metadata = cached_model()
    except Exception as exc:
        st.error(f"The saved model could not be loaded: {exc}")
        st.stop()

    model_name = model_metadata.get("model_name", "trained classifier")
    st.caption(f"Loaded model: {model_name}")

    accepted_types = [extension.lstrip(".") for extension in VALID_AUDIO_EXTENSIONS]
    uploaded_file = st.file_uploader(
        "Upload an audio recording",
        type=accepted_types,
        help="WAV and FLAC are the most portable. MP3, OGG, and M4A depend on local decoders.",
    )
    if uploaded_file is None:
        st.write("Choose an audio file to begin.")
        return

    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    st.audio(file_bytes, format=uploaded_file.type)

    try:
        with st.spinner("Analyzing the recording..."):
            audio, audio_info, feature_vector = analyze_uploaded_audio(
                file_bytes,
                suffix,
            )
    except (AudioProcessingError, FeatureExtractionError, OSError, ValueError) as exc:
        st.error(f"This audio file could not be processed: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected audio decoder error: {exc}")
        return

    st.markdown("### Audio information")
    info_columns = st.columns(4)
    info_columns[0].metric("Filename", uploaded_file.name)
    info_columns[1].metric(
        "Original duration",
        f"{float(audio_info['original_duration_seconds']):.2f} s",
    )
    info_columns[2].metric(
        "Original sample rate",
        f"{int(audio_info['original_sample_rate']):,} Hz",
    )
    info_columns[3].metric("Channels", str(int(audio_info["channels"])))
    st.caption(
        "Analysis uses mono audio resampled to 16,000 Hz and padded or cropped "
        "to exactly 3 seconds."
    )
    if bool(audio_info["was_silent"]):
        st.warning(
            "The processed recording is silent or nearly silent. A prediction "
            "can still be computed, but it is unlikely to be meaningful."
        )

    visual_left, visual_right = st.columns(2)
    with visual_left:
        waveform = waveform_figure(audio)
        st.pyplot(waveform, clear_figure=True)
    with visual_right:
        spectrogram = mel_spectrogram_figure(audio)
        st.pyplot(spectrogram, clear_figure=True)

    if not st.button("Predict behavioral context", type="primary"):
        return

    try:
        probabilities = ordered_probabilities(model, feature_vector.reshape(1, -1))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]
    digest = hashlib.sha256(file_bytes).hexdigest()
    interpretation = generate_translation(predicted_class, key=digest)

    st.markdown("## Prediction")
    result_left, result_right = st.columns([1, 2])
    with result_left:
        st.metric("Probable context", DISPLAY_CLASS_NAMES[predicted_class])
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    with result_right:
        probability_frame = pd.DataFrame(
            {
                "Context": [
                    DISPLAY_CLASS_NAMES[class_name]
                    for class_name in CANONICAL_CLASSES
                ],
                "Probability (%)": [
                    probabilities[class_name] * 100
                    for class_name in CANONICAL_CLASSES
                ],
            }
        ).set_index("Context")
        st.bar_chart(probability_frame, y="Probability (%)")

    st.markdown("### Playful interpretation")

    interpretation_column, meme_column = st.columns([1, 1])

    with interpretation_column:
        st.success(f"“{interpretation}”")
        st.caption(
            "The sentence above is selected from a predefined rule-based list. "
            "It is separate from the machine-learning prediction."
        )

    with meme_column:
        meme_path = MEME_IMAGE_PATHS.get(predicted_class)

        if meme_path is not None and meme_path.is_file():
            left_space, centered_image, right_space = st.columns([1, 3, 1])

            with centered_image:
                st.image(
                    str(meme_path),
                    caption="Meme selected according to the predicted context.",
                    width=300,
                )
        else:
            st.info("No meme image was found for this prediction.")

    st.warning(DISCLAIMER)

if __name__ == "__main__":
    main()
