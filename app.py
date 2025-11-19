import streamlit as st
import tensorflow as tf # type: ignore
import numpy as np
from PIL import Image
import cv2 # type: ignore

# --- Configuration ---
MODEL_PATH = 'model/waste_sorter.h5'
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 224

# Simple "database" for recycling recommendations
BIN_RECOMMENDATIONS = {
    'cardboard': {
        'bin': 'Blue Bin (Paper/Cardboard)',
        'color': '#007bff' # Blue
    },
    'glass': {
        'bin': 'Green Bin (Glass)',
        'color': '#28a745' # Green
    },
    'metal': {
        'bin': 'Grey Bin (Metal)',
        'color': '#6c757d' # Grey
    },
    'paper': {
        'bin': 'Blue Bin (Paper/Cardboard)',
        'color': '#007bff' # Blue
    },
    'plastic': {
        'bin': 'Yellow Bin (Plastic)',
        'color': '#ffc107' # Yellow
    },
    'trash': {
        'bin': 'Black Bin (General Waste/Organic)',
        'color': '#343a40' # Black
    }
}

# --- Model Loading ---
@st.cache(allow_output_mutation=True)
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Helper Functions ---
def preprocess_image(image):
    """Preprocesses the input image for the model."""
    # Convert PIL image to numpy array (if needed)
    if isinstance(image, Image.Image):
        image = np.array(image)

    # If the image has an alpha channel, remove it
    if image.shape[-1] == 4:
        image = image[..., :3]
        
    # Resize image to model's expected input size
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values (0-1) and expand dimensions for batch
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0
    return img_array

def make_prediction(img_array):
    """Gets predictions from the model."""
    if model is None:
        return None, 0.0

    predictions = model.predict(img_array)
    score = np.max(predictions)
    class_index = np.argmax(predictions)
    class_name = CLASSES[class_index]
    return class_name, score

def update_stats(predicted_class):
    """Updates the session state statistics."""
    if 'stats' not in st.session_state:
        st.session_state.stats = {c: 0 for c in CLASSES}
    st.session_state.stats[predicted_class] += 1

# --- Streamlit UI ---

st.set_page_config(page_title="AI Waste Sorter", page_icon="♻️", layout="wide")

# Initialize session state for stats
if 'stats' not in st.session_state:
    st.session_state.stats = {c: 0 for c in CLASSES}

# --- Sidebar ---
st.sidebar.title("AI Waste Sorter")
st.sidebar.write("Upload an image or use your camera to classify waste.")

input_method = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Camera"))

st.sidebar.header("Sorting Statistics")
st.sidebar.write("A summary of items sorted during this session.")

# Display stats as a bar chart
stats_data = st.session_state.stats
if sum(stats_data.values()) > 0:
    st.sidebar.bar_chart(stats_data)
else:
    st.sidebar.text("No items sorted yet.")

if st.sidebar.button("Reset Statistics"):
    st.session_state.stats = {c: 0 for c in CLASSES}
    st.experimental_rerun()

# --- Main Page ---
st.title("♻️ AI-Powered Waste Sorting")
st.write("Detect and classify waste to promote proper recycling.")

# Create two columns for input and output
col1, col2 = st.columns(2)

image_to_process = None

with col1:
    st.subheader("Input Image")
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, caption='Uploaded Image', use_column_width=True)
            
    elif input_method == "Use Camera":
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            image_to_process = Image.open(img_file_buffer)
            st.image(image_to_process, caption='Captured Image', use_column_width=True)

with col2:
    st.subheader("Classification Results")
    
    if image_to_process is None:
        st.info("Please provide an image using the options on the left.")
    else:
        # Process the image
        processed_image_array = preprocess_image(image_to_process)
        
        # Make prediction
        with st.spinner('Classifying...'):
            class_name, confidence = make_prediction(processed_image_array)
        
        if class_name:
            # Update stats
            update_stats(class_name)
            
            # Get recommendation
            recommendation = BIN_RECOMMENDATIONS[class_name]
            bin_color = recommendation['color']
            
            # Display results
            st.success(f"**Predicted Class:** {class_name.capitalize()}")
            
            st.write("**Confidence:**")
            st.progress(float(confidence))
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {bin_color}; color: white; text-align: center;">
                <h3 style="color: white;">Recommended Bin:</h3>
                <h2>{recommendation['bin']}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Could not classify the image. Is the model loaded correctly?")

st.markdown("---")
st.write("Powered by TensorFlow, Keras, and Streamlit. Model trained on the Kaggle Waste Classification dataset.")