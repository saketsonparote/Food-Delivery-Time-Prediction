import streamlit as st
import numpy as np
import pickle

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍕 Food Delivery Time Predictor",
    page_icon="🛵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Card container */
    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    /* Section headers */
    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #a78bfa;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* Prediction result box */
    .result-box {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.45);
        margin-top: 20px;
    }
    .result-label {
        font-size: 14px;
        color: rgba(255,255,255,0.7);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-time {
        font-size: 72px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }
    .result-unit {
        font-size: 20px;
        color: rgba(255,255,255,0.75);
        margin-top: 6px;
    }

    /* Insight chips */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
        justify-content: center;
    }
    .chip {
        background: rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 13px;
        color: #e0d7ff;
    }

    /* Slider & select overrides */
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #f0f0f0 !important;
        border-radius: 10px !important;
    }
    .stSlider > div > div > div > div {
        background: #7c3aed !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 1px;
        border: none;
        border-radius: 12px;
        padding: 14px 0;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(124, 58, 237, 0.6);
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .hero h1 {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero p {
        color: rgba(255,255,255,0.55);
        font-size: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.25);
        font-size: 12px;
        margin-top: 40px;
        padding-bottom: 20px;
    }

    /* Hide default streamlit branding */
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load Model & Encoders ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("optimized_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, label_encoders = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Helper: encode input ──────────────────────────────────────────────────────
def encode_input(weather, traffic, time_of_day, vehicle):
    """Encode categorical values using the saved label encoders."""
    encoded = {}
    mapping = {
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
    }
    for col, val in mapping.items():
        if col in label_encoders:
            le = label_encoders[col]
            if val in le.classes_:
                encoded[col] = le.transform([val])[0]
            else:
                # fallback: use first class
                encoded[col] = le.transform([le.classes_[0]])[0]
        else:
            # If encoder not found, use a simple manual fallback
            fallbacks = {
                "Weather":       {"Clear": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3, "Windy": 4},
                "Traffic_Level": {"High": 0, "Low": 1, "Medium": 2},
                "Time_of_Day":   {"Afternoon": 0, "Evening": 1, "Morning": 2, "Night": 3},
                "Vehicle_Type":  {"Bike": 0, "Car": 1, "Scooter": 2},
            }
            encoded[col] = fallbacks.get(col, {}).get(val, 0)
    return encoded


# ── Weather & Traffic emoji helpers ──────────────────────────────────────────
WEATHER_EMOJI = {
    "Clear": "☀️", "Rainy": "🌧️", "Foggy": "🌫️",
    "Snowy": "❄️", "Windy": "💨"
}
TRAFFIC_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
VEHICLE_EMOJI = {"Bike": "🚲", "Scooter": "🛵", "Car": "🚗"}
TIME_EMOJI = {
    "Morning": "🌅", "Afternoon": "🌞",
    "Evening": "🌇", "Night": "🌙"
}


# ── UI ────────────────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero">
    <h1>🛵 Delivery Time Predictor</h1>
    <p>Fill in the order details below and get an estimated delivery time instantly.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found! Make sure `optimized_rf_model.pkl` and `label_encoders.pkl` are in the same folder as `main.py`.")
    st.stop()

# ── Section 1: Distance & Prep ────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📦 Order Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    distance = st.slider(
        "📍 Distance (km)",
        min_value=0.5, max_value=20.0,
        value=8.0, step=0.1,
        help="Distance from restaurant to delivery location"
    )
with col2:
    prep_time = st.slider(
        "🍳 Preparation Time (min)",
        min_value=1, max_value=30,
        value=15, step=1,
        help="Time restaurant takes to prepare the order"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Conditions ─────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌦️ Delivery Conditions</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    weather_options = ["Clear", "Rainy", "Foggy", "Snowy", "Windy"]
    weather = st.selectbox(
        "Weather",
        options=weather_options,
        format_func=lambda x: f"{WEATHER_EMOJI[x]} {x}",
        index=0
    )
with col4:
    traffic_options = ["Low", "Medium", "High"]
    traffic = st.selectbox(
        "Traffic Level",
        options=traffic_options,
        format_func=lambda x: f"{TRAFFIC_EMOJI[x]} {x}",
        index=1
    )
with col5:
    time_options = ["Morning", "Afternoon", "Evening", "Night"]
    time_of_day = st.selectbox(
        "Time of Day",
        options=time_options,
        format_func=lambda x: f"{TIME_EMOJI[x]} {x}",
        index=1
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Courier ────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧑‍💼 Courier Details</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    vehicle_options = ["Bike", "Scooter", "Car"]
    vehicle = st.selectbox(
        "Vehicle Type",
        options=vehicle_options,
        format_func=lambda x: f"{VEHICLE_EMOJI[x]} {x}",
        index=1
    )
with col7:
    experience = st.slider(
        "👨‍💼 Courier Experience (years)",
        min_value=0.0, max_value=10.0,
        value=3.0, step=0.5,
        help="Years of delivery experience"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🚀 Predict Delivery Time")

if predict_clicked:
    # Encode categoricals
    encoded = encode_input(weather, traffic, time_of_day, vehicle)

    # Build feature array — order must match training
    # Training columns (excluding Order_ID and target):
    # Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type,
    # Preparation_Time_min, Courier_Experience_yrs
    features = np.array([[
        distance,
        encoded["Weather"],
        encoded["Traffic_Level"],
        encoded["Time_of_Day"],
        encoded["Vehicle_Type"],
        prep_time,
        experience,
    ]])

    prediction = model.predict(features)[0]
    predicted_minutes = round(float(prediction))

    # Delivery window
    lower = max(1, predicted_minutes - 5)
    upper = predicted_minutes + 5

    # Result card
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Delivery Time</div>
        <div class="result-time">{predicted_minutes}</div>
        <div class="result-unit">minutes</div>
        <div class="chip-row">
            <div class="chip">⏱️ Window: {lower}–{upper} min</div>
            <div class="chip">{WEATHER_EMOJI[weather]} {weather}</div>
            <div class="chip">{TRAFFIC_EMOJI[traffic]} {traffic} Traffic</div>
            <div class="chip">{VEHICLE_EMOJI[vehicle]} {vehicle}</div>
            <div class="chip">{TIME_EMOJI[time_of_day]} {time_of_day}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Friendly message
    st.markdown("<br>", unsafe_allow_html=True)
    if predicted_minutes <= 30:
        msg = "⚡ Super fast delivery! Barely enough time to set the table."
    elif predicted_minutes <= 50:
        msg = "✅ Decent delivery time. Go wash your hands."
    elif predicted_minutes <= 70:
        msg = "🕐 Moderate wait. Maybe watch a YouTube short or two."
    else:
        msg = "😅 Long wait ahead. Perfect time for a snack... wait, that's what you're ordering."

    st.info(msg)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using Random Forest · Streamlit · Python
</div>
""", unsafe_allow_html=True)