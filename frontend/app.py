import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="LeafLens | Plant Health",page_icon="🌿",layout="wide",initial_sidebar_state="collapsed")
# Application styling
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #238636;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #777;
            margin-top: 0;
        }
        .result-card {
            padding: 20px;
            border-radius: 12px;
            background-color: rgba(35, 134, 54, 0.08);
            border: 1px solid rgba(35, 134, 54, 0.3);
        }
    </style>
    """,unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🌿 LeafLens</p>',unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered plant leaf disease detection</p>',unsafe_allow_html=True)
st.divider()
st.markdown("### Analyse your plant")
st.write(
    "Upload a clear photograph of a plant leaf. "
    "LeafLens will analyse the image and identify the most likely disease.")

uploaded_file = st.file_uploader("Choose a leaf image",type=["jpg", "jpeg", "png"],help="Supported formats: JPG, JPEG and PNG. Maximum size: 10 MB.")

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### Uploaded image")
        st.image(image_bytes,caption=uploaded_file.name,width="stretch")
    with col2:
        st.markdown("#### Analysis")
        if st.button("Analyse leaf",type="primary",width="stretch"):
            with st.spinner("Analysing your leaf image..."):
                try:
                    response = requests.post(f"{API_URL}/predict",files={"file": (uploaded_file.name,image_bytes,uploaded_file.type)},timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Analysis completed successfully!")
                        st.markdown('<div class="result-card">',unsafe_allow_html=True)
                        st.markdown("**Predicted condition**")
                        st.subheader(result["predicted_class"].replace("___", " — "))

                        confidence = result["confidence"]
                        st.metric("Prediction confidence",f"{confidence * 100:.2f}%")
                        st.progress(confidence)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("#### Top 5 predictions")
                        for index, prediction in enumerate(result["top_predictions"],start=1):
                            class_name = prediction["class_name"].replace(
                                "___", " — "
                            )
                            score = prediction["confidence"]

                            st.write(f"**{index}. {class_name}**")
                            st.progress(score)
                            st.caption(f"Confidence: {score * 100:.2f}%")

                        st.caption(
                            f"Model: {result['model_name']} | "
                            f"Device: {result['inference_device']}")
                    else:
                        try:
                            error_message = response.json().get("detail","The server could not process the image.")
                        except ValueError:
                            error_message = "Unexpected response from the server."
                        st.error(f"Prediction failed: {error_message}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the LeafLens API. "
                        "Please ensure the backend is running.")
                except requests.exceptions.Timeout:
                    st.error("The prediction request timed out. Please try again.")
                except requests.exceptions.RequestException as error:
                    st.error(f"Communication error: {error}")
else:
    st.info("Upload a leaf image to begin the analysis.")
st.divider()
st.caption(
    "LeafLens | AI-powered plant health analysis | "
    "Research prototype — predictions should be verified by an expert.")