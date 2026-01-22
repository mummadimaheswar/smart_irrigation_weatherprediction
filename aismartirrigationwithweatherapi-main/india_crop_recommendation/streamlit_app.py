"""
India Crop Recommendation System - Streamlit Frontend
Easy-to-use web interface with Grok AI chatbot integration

Run with: streamlit run streamlit_app.py
"""
import os
import sys
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🌾 India Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .crop-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .sensor-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
    }
    
    /* Chat UI - Dark Theme */
    .chat-container {
        background: #1a1a2e;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .chat-user {
        background: #0066cc;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-ai {
        background: #16213e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        border-left: 3px solid #0066cc;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-ai strong, .chat-ai b {
        color: #4da6ff;
    }
    .chat-ai ul, .chat-ai ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .chat-ai li {
        margin: 0.3rem 0;
    }
    .chat-ai code {
        background: #0d1b2a;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #4da6ff;
    }
    .chat-wrapper {
        background: #0d1b2a;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #1e3a5f;
    }
    .chat-header {
        background: #0066cc;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════════════════

INDIAN_STATES = [
    "Andhra Pradesh", "Gujarat", "Himachal Pradesh", "Maharashtra",
    "Punjab", "Rajasthan", "Tamil Nadu", "Telangana",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

SAMPLE_DATA = {
    "Maharashtra": [21.43, 11.54, 12.98, 19.20, 15.28, 19.04, 10.16, 9.01, 16.17, 18.23,
                   22.15, 14.67, 17.89, 20.45, 13.22, 18.90, 15.67, 21.34, 16.78, 19.56],
    "Gujarat": [18.5, 15.2, 22.1, 19.8, 16.4, 20.3, 17.9, 14.6, 21.7, 18.2,
               15.8, 19.4, 16.9, 20.8, 17.3, 15.9, 21.2, 18.7, 16.1, 19.9],
    "Punjab": [25.3, 28.1, 24.7, 26.9, 23.5, 27.4, 25.8, 29.2, 24.1, 26.3,
              28.7, 25.0, 27.1, 24.9, 26.5, 28.3, 25.6, 27.8, 24.4, 26.8],
    "Rajasthan": [12.5, 10.2, 14.1, 11.8, 13.4, 9.3, 15.9, 11.6, 12.7, 10.2,
                13.8, 11.4, 14.9, 10.8, 12.3, 15.9, 11.2, 13.7, 10.1, 14.9],
    "Tamil Nadu": [19.5, 22.2, 18.1, 21.8, 17.4, 23.3, 19.9, 20.6, 18.7, 22.2,
                 17.8, 21.4, 18.9, 22.8, 19.3, 20.9, 18.2, 21.7, 19.1, 20.9],
    "Andhra Pradesh": [20.5, 18.2, 22.1, 19.8, 21.4, 17.3, 23.9, 18.6, 20.7, 19.2,
                      22.8, 17.4, 21.9, 18.8, 20.3, 22.9, 18.2, 21.7, 19.1, 20.9],
    "Telangana": [18.5, 21.2, 17.1, 20.8, 16.4, 22.3, 18.9, 19.6, 17.7, 21.2,
                16.8, 20.4, 17.9, 21.8, 18.3, 19.9, 17.2, 20.7, 18.1, 19.9],
    "Uttar Pradesh": [22.5, 25.2, 21.1, 24.8, 20.4, 26.3, 22.9, 23.6, 21.7, 25.2,
                     20.8, 24.4, 21.9, 25.8, 22.3, 23.9, 21.2, 24.7, 22.1, 23.9],
    "West Bengal": [28.5, 31.2, 27.1, 30.8, 26.4, 32.3, 28.9, 29.6, 27.7, 31.2,
                   26.8, 30.4, 27.9, 31.8, 28.3, 29.9, 27.2, 30.7, 28.1, 29.9],
    "Himachal Pradesh": [15.5, 18.2, 14.1, 17.8, 13.4, 19.3, 15.9, 16.6, 14.7, 18.2,
                        13.8, 17.4, 14.9, 18.8, 15.3, 16.9, 14.2, 17.7, 15.1, 16.9],
    "Uttarakhand": [16.5, 19.2, 15.1, 18.8, 14.4, 20.3, 16.9, 17.6, 15.7, 19.2,
                  14.8, 18.4, 15.9, 19.8, 16.3, 17.9, 15.2, 18.7, 16.1, 17.9]
}

CROP_RULES = {
    "rice": {"sm_min": 30, "sm_max": 80, "temp_min": 20, "temp_max": 35, "season": "kharif", "emoji": "🌾"},
    "wheat": {"sm_min": 20, "sm_max": 50, "temp_min": 10, "temp_max": 25, "season": "rabi", "emoji": "🌾"},
    "maize": {"sm_min": 25, "sm_max": 60, "temp_min": 18, "temp_max": 32, "season": "kharif", "emoji": "🌽"},
    "cotton": {"sm_min": 20, "sm_max": 50, "temp_min": 20, "temp_max": 40, "season": "kharif", "emoji": "☁️"},
    "sugarcane": {"sm_min": 40, "sm_max": 70, "temp_min": 20, "temp_max": 35, "season": "perennial", "emoji": "🎋"},
    "groundnut": {"sm_min": 20, "sm_max": 45, "temp_min": 25, "temp_max": 35, "season": "kharif", "emoji": "🥜"},
    "soybean": {"sm_min": 30, "sm_max": 60, "temp_min": 20, "temp_max": 30, "season": "kharif", "emoji": "🫘"},
    "mustard": {"sm_min": 15, "sm_max": 40, "temp_min": 10, "temp_max": 25, "season": "rabi", "emoji": "🌻"},
    "chickpea": {"sm_min": 15, "sm_max": 35, "temp_min": 15, "temp_max": 30, "season": "rabi", "emoji": "🫘"},
    "potato": {"sm_min": 25, "sm_max": 50, "temp_min": 15, "temp_max": 25, "season": "rabi", "emoji": "🥔"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'sensor_values' not in st.session_state:
    st.session_state.sensor_values = [0.0] * 20

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'grok_api_key' not in st.session_state:
    st.session_state.grok_api_key = os.getenv("GROK_API_KEY", "")

if 'weather_api_key' not in st.session_state:
    st.session_state.weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_season(month: int) -> str:
    """Determine agricultural season from month."""
    if 6 <= month <= 10:
        return "kharif"
    elif month >= 10 or month <= 3:
        return "rabi"
    return "zaid"


def calculate_crop_scores(sm: float, temp: float, month: int) -> List[Dict]:
    """Calculate crop suitability scores."""
    season = get_season(month)
    scores = []
    
    for crop, rules in CROP_RULES.items():
        score = 0
        
        # Soil moisture score (40%)
        if rules["sm_min"] <= sm <= rules["sm_max"]:
            optimal_sm = (rules["sm_min"] + rules["sm_max"]) / 2
            sm_score = 1 - abs(sm - optimal_sm) / (rules["sm_max"] - rules["sm_min"])
            score += sm_score * 0.4
        
        # Temperature score (30%)
        if rules["temp_min"] <= temp <= rules["temp_max"]:
            optimal_temp = (rules["temp_min"] + rules["temp_max"]) / 2
            temp_score = 1 - abs(temp - optimal_temp) / (rules["temp_max"] - rules["temp_min"])
            score += temp_score * 0.3
        
        # Season match (30%)
        if rules["season"] == season or rules["season"] == "perennial":
            score += 0.3
        elif rules["season"] in ["kharif", "rabi"] and season == "zaid":
            score += 0.1
        
        if score > 0.3:
            scores.append({
                "crop": crop.title(),
                "confidence": min(score, 1.0),
                "season": rules["season"],
                "emoji": rules["emoji"],
                "notes": f"Optimal moisture: {rules['sm_min']}-{rules['sm_max']}%"
            })
    
    return sorted(scores, key=lambda x: x["confidence"], reverse=True)[:5]


def fetch_weather(city: str, api_key: str) -> Optional[Dict]:
    """Fetch weather data from OpenWeatherMap."""
    if not api_key:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "city": data["name"]
            }
    except Exception as e:
        st.warning(f"Weather fetch error: {e}")
    return None


def call_grok_api(message: str, context: Dict, api_key: str) -> Optional[str]:
    """Call Groq Cloud API for chat responses."""
    if not api_key:
        return "⚠️ No API key provided. Please enter your Groq API key in the sidebar."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an expert Indian agricultural advisor. Help farmers with:
- Crop recommendations based on location, season, soil moisture
- Soil & water management advice
- Weather-based farming decisions
Be concise, practical, and use simple language."""
        
        context_str = ""
        if context:
            context_str = f"\n\nCurrent context: {json.dumps(context)}"
        
        # Groq Cloud models
        models_to_try = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"]
        
        last_error = None
        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt + context_str},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                return "❌ **Invalid API Key**: Please check your Groq API key is correct."
            elif response.status_code == 404 or response.status_code == 400:
                # Model not found, try next
                last_error = f"Model {model_name} not available"
                continue
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
        
        return f"❌ **API Error**: {last_error}"
        
    except requests.exceptions.Timeout:
        return "❌ **Timeout**: The API request took too long. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error**: Cannot reach the Groq API. Check your internet connection."
    except Exception as e:
        return f"❌ **Error**: {str(e)}"


def get_offline_response(message: str, context: Dict) -> str:
    """Generate offline response when API is not available."""
    msg = message.lower()
    state = context.get("state", "your region")
    moisture = context.get("soil_moisture")
    month = context.get("month", datetime.now().month)
    season = get_season(month)
    
    if any(word in msg for word in ["recommend", "crop", "grow", "plant"]):
        if season == "kharif":
            crops = ["Rice", "Cotton", "Maize", "Soybean", "Groundnut"]
        elif season == "rabi":
            crops = ["Wheat", "Mustard", "Chickpea", "Potato", "Barley"]
        else:
            crops = ["Watermelon", "Muskmelon", "Cucumber", "Vegetables"]
        
        response = f"**{season.title()} Season Recommendations for {state}:**\n\n"
        for i, crop in enumerate(crops, 1):
            response += f"{i}. {crop}\n"
        
        if moisture:
            response += f"\n💧 With soil moisture at {moisture:.1f}%, "
            if moisture < 25:
                response += "consider drought-resistant crops."
            elif moisture > 50:
                response += "water-loving crops would thrive."
            else:
                response += "most crops should do well."
        
        return response
    
    if any(word in msg for word in ["weather", "rain", "monsoon"]):
        return f"""🌧️ **Weather Guidance for {state}:**

• Current season: **{season.title()}**
• If expecting heavy rain, ensure proper drainage
• Light rain is ideal for sowing operations
• Monitor forecasts for timely irrigation"""
    
    if any(word in msg for word in ["irrigation", "water"]):
        return f"""💧 **Irrigation Tips:**

• Morning irrigation (6-9 AM) reduces evaporation
• Drip irrigation saves 30-50% water
• Mulching helps retain soil moisture
• Consider rainwater harvesting"""
    
    return """I'm your AI farming assistant! I can help with:

• **"What crops should I grow?"** - Get recommendations
• **"Irrigation tips"** - Water management advice  
• **"Weather guidance"** - Seasonal advice

*Note: Enter your Grok API key in the sidebar for AI-powered responses.*"""

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - API KEYS & SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🔑 API Keys")
    
    # Groq API Key Section
    st.markdown("##### 🤖 Groq Cloud API")
    st.caption("Get FREE key: [console.groq.com](https://console.groq.com)")
    grok_key = st.text_input(
        "Groq API Key",
        value=st.session_state.grok_api_key,
        type="password",
        placeholder="gsk_xxxxxxxxxxxx",
        help="Get your FREE key from console.groq.com"
    )
    if grok_key != st.session_state.grok_api_key:
        st.session_state.grok_api_key = grok_key
    
    if st.button("🧪 Test Groq", use_container_width=True):
        if grok_key:
            if not grok_key.startswith("gsk_"):
                st.error("❌ Keys start with 'gsk_'")
            else:
                with st.spinner("Testing..."):
                    result = call_grok_api("Say hello", {}, grok_key)
                    if result and "❌" not in result:
                        st.success("✅ Connected!")
                    else:
                        st.error(result)
        else:
            st.warning("Enter key first")
    
    st.markdown("---")
    
    # Weather API Key Section
    st.markdown("##### 🌤️ OpenWeatherMap API")
    st.caption("Get key: [openweathermap.org](https://openweathermap.org/api)")
    weather_key = st.text_input(
        "Weather API Key",
        value=st.session_state.weather_api_key,
        type="password",
        placeholder="Enter API key",
        help="Get your key from openweathermap.org"
    )
    if weather_key != st.session_state.weather_api_key:
        st.session_state.weather_api_key = weather_key
    
    if st.button("🧪 Test Weather", use_container_width=True):
        if weather_key:
            with st.spinner("Testing..."):
                result = fetch_weather("Delhi", weather_key)
                if result:
                    st.success(f"✅ {result['temp']}°C")
                else:
                    st.error("❌ Invalid key")
        else:
            st.warning("Enter key first")
    
    st.markdown("---")
    
    # Quick Stats
    st.header("📊 Sensor Stats")
    sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
    if sensor_vals:
        avg = sum(sensor_vals) / len(sensor_vals)
        st.metric("Average Moisture", f"{avg:.1f}%")
        col1, col2 = st.columns(2)
        col1.metric("Min", f"{min(sensor_vals):.1f}%")
        col2.metric("Max", f"{max(sensor_vals):.1f}%")
        st.metric("Active Sensors", f"{len(sensor_vals)}/20")
    else:
        st.info("Enter sensor readings")
    
    st.markdown("---")
    st.markdown("**India Crop Recommendation v2.0**")
    st.markdown("🌾 Streamlit Edition")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🌾 India Crop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered crop recommendations with 20 sensor inputs, RAG Knowledge Base & Grok chatbot</p>', unsafe_allow_html=True)

# Create tabs - 6 tabs total
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 IoT Sensor Data",
    "🚀 Decision Engine", 
    "🎯 Crop Recommendations", 
    "🤖 AI Chatbot", 
    "📚 RAG Knowledge Base",
    "📊 Data Analysis"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: IoT SENSOR DATA
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📡 20-Point IoT Soil Moisture Sensor Network")
    st.caption("Configure soil moisture readings from your IoT sensor network at 15cm depth")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### Quick Actions")
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        with action_col1:
            state_for_sample = st.selectbox("Load State Data", ["Select..."] + INDIAN_STATES, key="sample_state_iot")
        
        with action_col2:
            if st.button("📊 Load Sample", use_container_width=True, key="load_sample_iot"):
                if state_for_sample != "Select..." and state_for_sample in SAMPLE_DATA:
                    st.session_state.sensor_values = SAMPLE_DATA[state_for_sample].copy()
                    st.rerun()
        
        with action_col3:
            if st.button("🎲 Random Generate", use_container_width=True, key="random_iot"):
                import random
                st.session_state.sensor_values = [round(random.uniform(10, 35), 1) for _ in range(20)]
                st.rerun()
        
        with action_col4:
            if st.button("🗑️ Clear All", use_container_width=True, key="clear_iot"):
                st.session_state.sensor_values = [0.0] * 20
                st.rerun()
        
        st.markdown("---")
        st.markdown("##### Manual Sensor Entry (20 Sensors)")
        st.caption("Enter soil moisture % for each sensor location")
        
        for row in range(5):
            cols = st.columns(4)
            for col_idx in range(4):
                sensor_idx = row * 4 + col_idx
                with cols[col_idx]:
                    st.session_state.sensor_values[sensor_idx] = st.number_input(
                        f"Sensor {sensor_idx + 1}",
                        value=float(st.session_state.sensor_values[sensor_idx]),
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key=f"sensor_iot_{sensor_idx}"
                    )
    
    with col2:
        st.markdown("##### 🗺️ Sensor Field Layout (4x5 Grid)")
        st.caption("🟢 ≥20% Optimal  |  🟠 10-19% Low  |  🔴 <10% Critical  |  ⚪ No Data")
        
        for row in range(5):
            cols = st.columns(4)
            for col_idx in range(4):
                sensor_idx = row * 4 + col_idx
                val = st.session_state.sensor_values[sensor_idx]
                
                with cols[col_idx]:
                    if val > 0:
                        if val >= 20:
                            st.success(f"**S{sensor_idx+1}** 💧\n\n**{val:.1f}%**")
                        elif val >= 10:
                            st.warning(f"**S{sensor_idx+1}** ⚠️\n\n**{val:.1f}%**")
                        else:
                            st.error(f"**S{sensor_idx+1}** 🔴\n\n**{val:.1f}%**")
                    else:
                        st.info(f"**S{sensor_idx+1}** ○\n\n**{val:.1f}%**")
        
        st.caption(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        if sensor_vals:
            st.markdown("##### Field Statistics")
            st.metric("Average VWC", f"{sum(sensor_vals)/len(sensor_vals):.1f}%")
            st.metric("Field Coverage", f"{len(sensor_vals)}/20 sensors")
            
            avg = sum(sensor_vals) / len(sensor_vals)
            if avg < 15:
                st.error("🔴 Critical: Immediate irrigation needed")
            elif avg < 22:
                st.warning("🟡 Low: Consider irrigation")
            elif avg < 35:
                st.success("🟢 Optimal: Good moisture level")
            else:
                st.info("🔵 High: Monitor drainage")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIVE WEATHER & IRRIGATION PREDICTION SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("🌦️ Live Weather & Irrigation Prediction")
    
    # Location selector
    loc_col1, loc_col2, loc_col3 = st.columns([1, 1, 1])
    
    with loc_col1:
        weather_state = st.selectbox("🏛️ State", INDIAN_STATES, index=3, key="weather_state_iot")
    
    with loc_col2:
        weather_city = st.text_input("🏘️ City/District", placeholder="e.g., Pune, Mumbai", key="weather_city_iot")
    
    with loc_col3:
        crop_for_prediction = st.selectbox("🌾 Crop Type", list(CROP_RULES.keys()), key="crop_iot")
    
    # Weather and Prediction columns
    weather_col1, weather_col2, weather_col3 = st.columns([1, 1, 1])
    
    with weather_col1:
        if st.button("🌦️ Fetch Weather & Predict", type="primary", use_container_width=True, key="fetch_weather_iot"):
            api_key = st.session_state.weather_api_key
            if not api_key:
                st.error("⚠️ Please set Weather API Key in the sidebar!")
            else:
                with st.spinner(f"Fetching weather..."):
                    city = weather_city if weather_city else weather_state
                    weather_data = fetch_weather(city, api_key)
                    
                    if weather_data:
                        st.session_state.iot_live_weather = weather_data
                        st.session_state.iot_weather_fetched = True
                        st.success(f"✅ Weather fetched for {weather_data.get('city', city)}")
                    else:
                        st.error("❌ Failed to fetch weather. Check API key.")
    
    with weather_col2:
        st.markdown("##### 🌤️ Current Weather")
        
        if st.session_state.get('iot_weather_fetched') and 'iot_live_weather' in st.session_state:
            weather = st.session_state.iot_live_weather
            
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.metric("🌡️ Temperature", f"{weather.get('temp', 'N/A')}°C")
                st.metric("💨 Condition", weather.get('description', 'N/A').title())
            with wcol2:
                st.metric("💧 Humidity", f"{weather.get('humidity', 'N/A')}%")
                st.metric("📍 Location", weather.get('city', 'N/A'))
        else:
            st.info("👆 Click 'Fetch Weather & Predict' to load live weather data")
    
    with weather_col3:
        st.markdown("##### 🚿 Irrigation Prediction")
        
        sensor_vals_pred = [v for v in st.session_state.sensor_values if v > 0]
        
        if st.session_state.get('iot_weather_fetched') and 'iot_live_weather' in st.session_state and len(sensor_vals_pred) >= 5:
            weather = st.session_state.iot_live_weather
            avg_moisture = sum(sensor_vals_pred) / len(sensor_vals_pred)
            temp = weather.get('temp', 25)
            humidity = weather.get('humidity', 50)
            
            # Get crop rules
            crop_rules = CROP_RULES.get(crop_for_prediction, {"sm_min": 20, "sm_max": 50})
            optimal_sm = (crop_rules["sm_min"] + crop_rules["sm_max"]) / 2
            
            # Calculate irrigation need score
            moisture_deficit = optimal_sm - avg_moisture
            evapotranspiration_factor = (temp / 30) * (1 - humidity / 100)
            irrigation_score = moisture_deficit + (evapotranspiration_factor * 10)
            
            # Determine action
            if avg_moisture < crop_rules["sm_min"] * 0.7:
                action = "🔴 URGENT: Irrigate Immediately"
                action_type = "error"
                volume = max(20, min(50, moisture_deficit * 2))
            elif avg_moisture < crop_rules["sm_min"]:
                action = "🟠 EXECUTE: Irrigation Recommended"
                action_type = "warning"
                volume = max(10, min(30, moisture_deficit * 1.5))
            elif irrigation_score > 10:
                action = "🟡 DEFER: High evaporation, monitor"
                action_type = "warning"
                volume = max(5, min(20, moisture_deficit))
            elif avg_moisture > crop_rules["sm_max"]:
                action = "🔵 SKIP: Soil saturated, no irrigation"
                action_type = "info"
                volume = 0
            else:
                action = "🟢 OPTIMAL: No irrigation needed"
                action_type = "success"
                volume = 0
            
            # Display prediction
            if action_type == "error":
                st.error(action)
            elif action_type == "warning":
                st.warning(action)
            elif action_type == "info":
                st.info(action)
            else:
                st.success(action)
            
            st.metric("📊 Avg Soil Moisture", f"{avg_moisture:.1f}%")
            st.metric(f"🎯 Optimal for {crop_for_prediction.title()}", f"{crop_rules['sm_min']}-{crop_rules['sm_max']}%")
            
            if volume > 0:
                st.metric("💧 Recommended Volume", f"{volume:.0f} mm")
            
            # Detailed reasoning
            with st.expander("📋 Prediction Details"):
                st.write(f"**Crop:** {crop_for_prediction.title()}")
                st.write(f"**Soil Moisture:** {avg_moisture:.1f}% (Need: {crop_rules['sm_min']}-{crop_rules['sm_max']}%)")
                st.write(f"**Temperature:** {temp}°C")
                st.write(f"**Humidity:** {humidity}%")
                st.write(f"**Moisture Deficit:** {moisture_deficit:.1f}%")
                st.write(f"**ET Factor:** {evapotranspiration_factor:.2f}")
                st.write(f"**Decision Score:** {irrigation_score:.1f}")
        
        elif len(sensor_vals_pred) < 5:
            st.warning("⚠️ Need at least 5 sensor readings for prediction")
        else:
            st.info("👆 Fetch weather first to get irrigation prediction")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🚀 Smart Irrigation Decision Engine")
    st.caption("AI-powered decision system with EXECUTE/DEFER/SKIP logic")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("##### 📍 Location & Crop")
        decision_state = st.selectbox("State", INDIAN_STATES, index=3, key="decision_state")
        crop_type = st.selectbox("Crop Type", list(CROP_RULES.keys()), key="decision_crop")
        days_after_sowing = st.slider("Days After Sowing", 1, 180, 60)
        
        st.markdown("##### 🌦️ Weather Conditions")
        ambient_temp = st.slider("Ambient Temperature (°C)", 10, 50, 32)
        humidity = st.slider("Relative Humidity (%)", 20, 100, 65)
        forecast_rain = st.slider("Forecast Rain 24h (mm)", 0.0, 100.0, 2.5, 0.5)
        rain_probability = st.slider("Rain Probability (%)", 0, 100, 25)
        
        st.markdown("##### ⚙️ Constraints")
        equipment_available = st.checkbox("Equipment Available", value=True)
        water_quota = st.number_input("Water Quota (mm)", value=100.0, min_value=0.0)
    
    with col2:
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        
        if len(sensor_vals) < 5:
            st.warning("⚠️ Please enter at least 5 sensor readings in the IoT Sensor Data tab")
            avg_moisture = 22.0
        else:
            avg_moisture = sum(sensor_vals) / len(sensor_vals)
            st.info(f"📡 Using {len(sensor_vals)} sensor readings | Average: {avg_moisture:.1f}%")
        
        if st.button("🚀 Run Irrigation Decision", type="primary", use_container_width=True):
            with st.spinner("Processing environmental data and generating decision..."):
                # Get crop rules
                crop_rules = CROP_RULES.get(crop_type, {"sm_min": 20, "sm_max": 50})
                optimal_sm = (crop_rules["sm_min"] + crop_rules["sm_max"]) / 2
                
                # Calculate irrigation need
                moisture_deficit = optimal_sm - avg_moisture
                evapotranspiration_factor = (ambient_temp / 30) * (1 - humidity / 100)
                irrigation_score = moisture_deficit + (evapotranspiration_factor * 10)
                
                # Decision logic
                if not equipment_available:
                    action = "BLOCKED"
                    reason = "Equipment not available"
                    action_color = "error"
                    volume = 0
                elif rain_probability > 70 and forecast_rain > 10:
                    action = "DEFER"
                    reason = f"High rain probability ({rain_probability}%) with {forecast_rain}mm expected"
                    action_color = "warning"
                    volume = 0
                elif avg_moisture < crop_rules["sm_min"] * 0.7:
                    action = "EXECUTE"
                    reason = f"Critical moisture deficit. Soil at {avg_moisture:.1f}%, need >{crop_rules['sm_min']}%"
                    action_color = "error"
                    volume = max(20, min(50, moisture_deficit * 2))
                elif avg_moisture < crop_rules["sm_min"]:
                    action = "EXECUTE"
                    reason = f"Below optimal range. Current: {avg_moisture:.1f}%, Target: {crop_rules['sm_min']}-{crop_rules['sm_max']}%"
                    action_color = "warning"
                    volume = max(10, min(30, moisture_deficit * 1.5))
                elif avg_moisture > crop_rules["sm_max"]:
                    action = "SKIP"
                    reason = f"Soil moisture above optimal ({avg_moisture:.1f}% > {crop_rules['sm_max']}%)"
                    action_color = "info"
                    volume = 0
                else:
                    action = "SKIP"
                    reason = f"Soil moisture optimal ({avg_moisture:.1f}% in {crop_rules['sm_min']}-{crop_rules['sm_max']}%)"
                    action_color = "success"
                    volume = 0
                
                # Display Decision
                st.markdown("### 📋 Irrigation Decision Report")
                
                if action_color == "error":
                    st.error(f"🚨 **Action: {action}**")
                elif action_color == "warning":
                    st.warning(f"⚠️ **Action: {action}**")
                elif action_color == "info":
                    st.info(f"ℹ️ **Action: {action}**")
                else:
                    st.success(f"✅ **Action: {action}**")
                
                st.markdown(f"**Reason:** {reason}")
                
                # Metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Soil Moisture", f"{avg_moisture:.1f}%")
                with metric_cols[1]:
                    st.metric("Target Range", f"{crop_rules['sm_min']}-{crop_rules['sm_max']}%")
                with metric_cols[2]:
                    st.metric("Irrigation Volume", f"{volume:.0f} mm" if volume > 0 else "None")
                with metric_cols[3]:
                    st.metric("Decision Score", f"{irrigation_score:.1f}")
                
                # Detailed Report
                with st.expander("📊 Detailed Analysis", expanded=True):
                    st.markdown(f"""
                    **Environmental Conditions:**
                    - 🌡️ Temperature: {ambient_temp}°C
                    - 💧 Humidity: {humidity}%
                    - 🌧️ Forecast Rain: {forecast_rain}mm ({rain_probability}% probability)
                    - 🌱 Crop: {crop_type.title()} (Day {days_after_sowing})
                    
                    **Soil Analysis:**
                    - 📊 Average Moisture: {avg_moisture:.1f}%
                    - 📈 Optimal Range: {crop_rules['sm_min']}-{crop_rules['sm_max']}%
                    - 📉 Moisture Deficit: {moisture_deficit:.1f}%
                    - 🔥 ET Factor: {evapotranspiration_factor:.2f}
                    
                    **Constraints:**
                    - ⚙️ Equipment: {'Available ✅' if equipment_available else 'Unavailable ❌'}
                    - 💧 Water Quota: {water_quota}mm remaining
                    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: CROP RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Location & Weather")
        
        state = st.selectbox("State", INDIAN_STATES, index=3, key="rec_state")  # Maharashtra default
        district = st.text_input("District (optional)", placeholder="e.g., Pune", key="rec_district")
        
        # Weather fetch
        weather_col1, weather_col2 = st.columns(2)
        with weather_col1:
            temperature = st.number_input("Temperature (°C)", value=28.0, min_value=0.0, max_value=50.0, key="rec_temp")
        with weather_col2:
            if st.button("🌤️ Auto-fetch Weather", key="rec_weather_btn"):
                if st.session_state.weather_api_key:
                    city = district if district else state
                    weather = fetch_weather(city, st.session_state.weather_api_key)
                    if weather:
                        st.success(f"{weather['city']}: {weather['temp']}°C, {weather['description']}")
                        temperature = weather['temp']
                else:
                    st.warning("Add Weather API key in sidebar")
        
        rainfall = st.number_input("Rainfall (mm)", value=50.0, min_value=0.0, max_value=500.0)
        humidity = st.number_input("Humidity (%)", value=65.0, min_value=0.0, max_value=100.0)
        irrigation = st.checkbox("Irrigation Available", value=True)
        planting_date = st.date_input("Planting Date", value=date.today())
    
    with col2:
        st.subheader("💧 20 Sensor Readings")
        st.caption("Soil moisture % at 15cm depth")
        
        # Quick actions
        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("📊 Load Sample", use_container_width=True):
                if state in SAMPLE_DATA:
                    st.session_state.sensor_values = SAMPLE_DATA[state].copy()
                    st.rerun()
        with action_col2:
            if st.button("🎲 Random", use_container_width=True):
                import random
                st.session_state.sensor_values = [round(random.uniform(10, 40), 2) for _ in range(20)]
                st.rerun()
        with action_col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.sensor_values = [0.0] * 20
                st.rerun()
        
        # Sensor input grid (4 columns x 5 rows)
        for row in range(5):
            cols = st.columns(4)
            for col_idx in range(4):
                sensor_idx = row * 4 + col_idx
                with cols[col_idx]:
                    st.session_state.sensor_values[sensor_idx] = st.number_input(
                        f"S{sensor_idx + 1}",
                        value=st.session_state.sensor_values[sensor_idx],
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key=f"sensor_{sensor_idx}"
                    )
    
    st.markdown("---")
    
    # Get Recommendations Button
    if st.button("🌾 Get Crop Recommendations", type="primary", use_container_width=True):
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        
        if len(sensor_vals) < 5:
            st.error("Please enter at least 5 sensor readings")
        else:
            avg_moisture = sum(sensor_vals) / len(sensor_vals)
            month = planting_date.month
            
            with st.spinner("Analyzing conditions..."):
                recommendations = calculate_crop_scores(avg_moisture, temperature, month)
            
            if recommendations:
                st.success(f"Found {len(recommendations)} suitable crops!")
                
                # Display recommendations
                for i, rec in enumerate(recommendations):
                    confidence_pct = int(rec["confidence"] * 100)
                    
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    with col_a:
                        st.markdown(f"### {rec['emoji']}")
                    with col_b:
                        st.markdown(f"**{rec['crop']}** - {rec['season'].title()} season")
                        st.caption(rec['notes'])
                    with col_c:
                        st.metric("Confidence", f"{confidence_pct}%")
                    
                    st.progress(rec["confidence"])
                    st.markdown("---")
                
                # Summary
                st.info(f"""
                **Summary:**
                - State: {state} | District: {district or 'N/A'}
                - Season: {get_season(month).title()}
                - Avg Soil Moisture: {avg_moisture:.1f}%
                - Temperature: {temperature}°C
                - Active Sensors: {len(sensor_vals)}/20
                """)
            else:
                st.warning("No suitable crops found for current conditions")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    # Dark themed chat wrapper
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🤖 AI Farming Assistant | Powered by Groq</div>', unsafe_allow_html=True)
    
    # Chat container with dark background
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        if not st.session_state.chat_history:
            st.markdown('''
            <div class="chat-ai">
                👋 <strong>Hello! I'm your AI farming assistant.</strong><br><br>
                I can help you with:<br>
                • <strong>Crop recommendations</strong> for your region<br>
                • <strong>Soil & water management</strong> advice<br>
                • <strong>Weather-based</strong> farming decisions<br><br>
                Just ask me anything about Indian agriculture!
            </div>
            ''', unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    # Convert markdown to HTML for better display
                    content = msg["content"].replace('\n', '<br>').replace('**', '<strong>').replace('*', '<em>')
                    content = content.replace('<strong><strong>', '<strong>').replace('</strong></strong>', '</strong>')
                    st.markdown(f'<div class="chat-ai">{content}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🌾 Recommend Crops"):
            st.session_state.pending_message = "What crops should I grow based on my current conditions?"
    with quick_col2:
        if st.button("💧 Soil Analysis"):
            st.session_state.pending_message = "Analyze my soil moisture readings and give advice"
    with quick_col3:
        if st.button("🚿 Irrigation Tips"):
            st.session_state.pending_message = "Give me irrigation tips for my region"
    with quick_col4:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Check for pending message
    if 'pending_message' in st.session_state:
        user_input = st.session_state.pending_message
        del st.session_state.pending_message
    else:
        user_input = None
    
    # Chat input
    user_message = st.chat_input("Ask about crops, soil, weather...")
    
    if user_message or user_input:
        message = user_message or user_input
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": message})
        
        # Build context
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        context = {
            "state": state if 'state' in dir() else "Maharashtra",
            "month": datetime.now().month,
        }
        if sensor_vals:
            context["soil_moisture"] = sum(sensor_vals) / len(sensor_vals)
            context["sensor_count"] = len(sensor_vals)
        
        # Get response
        with st.spinner("🤖 Thinking..."):
            if st.session_state.grok_api_key:
                response = call_grok_api(message, context, st.session_state.grok_api_key)
                # Response is always a string now, check for error markers
            else:
                response = get_offline_response(message, context)
        
        # Add AI response
        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("📊 Sensor Data Analysis")
    
    sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
    
    if sensor_vals:
        # Visualization
        df = pd.DataFrame({
            "Sensor": [f"S{i+1}" for i, v in enumerate(st.session_state.sensor_values) if v > 0],
            "Moisture (%)": sensor_vals
        })
        
        st.bar_chart(df.set_index("Sensor"))
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{sum(sensor_vals)/len(sensor_vals):.2f}%")
        with col2:
            st.metric("Minimum", f"{min(sensor_vals):.2f}%")
        with col3:
            st.metric("Maximum", f"{max(sensor_vals):.2f}%")
        with col4:
            variance = sum((x - sum(sensor_vals)/len(sensor_vals))**2 for x in sensor_vals) / len(sensor_vals)
            st.metric("Std Dev", f"{variance**0.5:.2f}")
        
        # Data table
        st.markdown("### Raw Data")
        st.dataframe(df, use_container_width=True)
        
        # Moisture interpretation
        avg = sum(sensor_vals) / len(sensor_vals)
        st.markdown("### Interpretation")
        if avg < 20:
            st.error("⚠️ **Low Moisture**: Consider immediate irrigation. Drought-resistant crops recommended.")
        elif avg < 35:
            st.warning("💧 **Moderate Moisture**: Good for most crops. Monitor regularly.")
        elif avg < 55:
            st.success("✅ **Optimal Moisture**: Excellent conditions for crop growth.")
        else:
            st.info("🌊 **High Moisture**: Good for water-loving crops. Ensure proper drainage.")
    else:
        st.info("📝 Enter sensor readings in the Recommendations tab to see analysis")
        
        # Show sample data preview
        st.markdown("### Available Sample Data")
        sample_df = pd.DataFrame(SAMPLE_DATA).T
        sample_df.columns = [f"S{i+1}" for i in range(20)]
        sample_df["Average"] = sample_df.mean(axis=1)
        st.dataframe(sample_df[["Average"]], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: RAG KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("📚 RAG Knowledge Base")
    st.markdown("""
    Upload agricultural documents to create a knowledge base. The system uses:
    - **Semantic Search**: Find contextually similar content
    - **Keyword Search (BM25)**: Match exact terms
    - **Metadata Filtering**: Filter by state, crop, category
    - **Hybrid Retrieval**: Combine methods for best results
    """)
    
    # Initialize RAG session state
    if 'rag_documents' not in st.session_state:
        st.session_state.rag_documents = []
    if 'rag_query_results' not in st.session_state:
        st.session_state.rag_query_results = []
    
    rag_col1, rag_col2 = st.columns([1, 1])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # DOCUMENT UPLOAD
    # ─────────────────────────────────────────────────────────────────────────────
    
    with rag_col1:
        st.markdown("### 📤 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'csv', 'json', 'md'],
            help="Supported: TXT, PDF, CSV, JSON, Markdown"
        )
        
        with st.expander("📋 Document Metadata (optional)", expanded=False):
            doc_state = st.selectbox("State", [""] + INDIAN_STATES, key="doc_state")
            doc_crop = st.text_input("Crop", placeholder="e.g., cotton, wheat, rice", key="doc_crop")
            doc_category = st.selectbox(
                "Category",
                ["", "farming_guide", "irrigation", "pest_management", "fertilizer", "weather", "soil", "market", "other"],
                key="doc_category"
            )
            doc_source = st.text_input("Source", placeholder="e.g., ICAR, KVK, University", key="doc_source")
            doc_description = st.text_area("Description", placeholder="Brief description of the document", key="doc_desc")
        
        if st.button("📥 Upload & Process", use_container_width=True, disabled=not uploaded_file):
            if uploaded_file:
                with st.spinner("Processing document..."):
                    try:
                        # Try API first
                        api_url = os.getenv("API_URL", "http://localhost:8000")
                        
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {}
                        if doc_state:
                            data["state"] = doc_state
                        if doc_crop:
                            data["crop"] = doc_crop
                        if doc_category:
                            data["category"] = doc_category
                        if doc_source:
                            data["source"] = doc_source
                        if doc_description:
                            data["description"] = doc_description
                        
                        response = requests.post(
                            f"{api_url}/rag/upload",
                            files=files,
                            data=data,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success(f"✅ Uploaded: {uploaded_file.name} ({result.get('num_chunks', 0)} chunks)")
                                # Refresh document list
                                st.session_state.rag_refresh = True
                            else:
                                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                    
                    except requests.exceptions.ConnectionError:
                        st.warning("⚠️ API not available. Processing locally...")
                        # Local processing fallback
                        try:
                            from api.rag_system import get_rag_system
                            rag = get_rag_system()
                            
                            metadata = {}
                            if doc_state:
                                metadata["state"] = doc_state
                            if doc_crop:
                                metadata["crop"] = doc_crop
                            if doc_category:
                                metadata["category"] = doc_category
                            if doc_source:
                                metadata["source"] = doc_source
                            if doc_description:
                                metadata["description"] = doc_description
                            
                            doc = rag.add_document(
                                filename=uploaded_file.name,
                                content=uploaded_file.getvalue(),
                                metadata=metadata
                            )
                            st.success(f"✅ Processed locally: {uploaded_file.name} ({len(doc.chunks)} chunks)")
                            st.session_state.rag_refresh = True
                        except Exception as e:
                            st.error(f"❌ Local processing error: {e}")
                    
                    except Exception as e:
                        st.error(f"❌ Upload error: {e}")
        
        # ─────────────────────────────────────────────────────────────────────────────
        # DOCUMENT LIST
        # ─────────────────────────────────────────────────────────────────────────────
        
        st.markdown("### 📑 Documents in Knowledge Base")
        
        # Fetch documents
        try:
            api_url = os.getenv("API_URL", "http://localhost:8000")
            response = requests.get(f"{api_url}/rag/documents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.rag_documents = data.get("documents", [])
        except:
            # Try local
            try:
                from api.rag_system import get_rag_system
                rag = get_rag_system()
                st.session_state.rag_documents = rag.list_documents()
            except:
                pass
        
        if st.session_state.rag_documents:
            for doc in st.session_state.rag_documents:
                with st.container():
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(f"**📄 {doc.get('filename', 'Unknown')}**")
                        st.caption(f"ID: {doc.get('id', 'N/A')} | Chunks: {doc.get('num_chunks', 0)} | Type: {doc.get('file_type', 'N/A')}")
                        meta = doc.get('metadata', {})
                        if meta:
                            meta_str = " | ".join([f"{k}: {v}" for k, v in meta.items() if k != 'upload_date'])
                            if meta_str:
                                st.caption(f"📋 {meta_str}")
                    with col_b:
                        if st.button("🗑️", key=f"del_{doc.get('id')}", help="Delete document"):
                            try:
                                api_url = os.getenv("API_URL", "http://localhost:8000")
                                requests.delete(f"{api_url}/rag/documents/{doc.get('id')}", timeout=5)
                                st.rerun()
                            except:
                                try:
                                    from api.rag_system import get_rag_system
                                    rag = get_rag_system()
                                    rag.delete_document(doc.get('id'))
                                    st.rerun()
                                except:
                                    pass
                    st.divider()
        else:
            st.info("📭 No documents uploaded yet. Upload your first document above!")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # RAG QUERY & CHAT
    # ─────────────────────────────────────────────────────────────────────────────
    
    with rag_col2:
        st.markdown("### 🔍 Query Knowledge Base")
        
        # Query options
        with st.expander("⚙️ Query Settings", expanded=False):
            retrieval_method = st.selectbox(
                "Retrieval Method",
                ["hybrid", "semantic", "keyword"],
                help="hybrid combines semantic + keyword for best results"
            )
            
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.1)
            with col_w2:
                keyword_weight = st.slider("Keyword Weight", 0.0, 1.0, 0.5, 0.1)
            
            top_k = st.slider("Number of Results", 1, 10, 5)
            
            # Filters
            st.markdown("**Filters:**")
            filter_state = st.selectbox("Filter by State", [""] + INDIAN_STATES, key="filter_state")
            filter_crop = st.text_input("Filter by Crop", placeholder="e.g., cotton", key="filter_crop")
            filter_category = st.selectbox(
                "Filter by Category",
                ["", "farming_guide", "irrigation", "pest_management", "fertilizer", "weather", "soil", "market"],
                key="filter_category"
            )
        
        # Query input
        rag_query = st.text_input("🔎 Enter your search query", placeholder="How to manage irrigation for cotton?")
        
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            search_btn = st.button("🔍 Search Knowledge Base", use_container_width=True, disabled=not rag_query)
        with col_q2:
            chat_btn = st.button("💬 Ask with RAG Context", use_container_width=True, disabled=not rag_query)
        
        # Handle Search
        if search_btn and rag_query:
            with st.spinner("Searching..."):
                try:
                    api_url = os.getenv("API_URL", "http://localhost:8000")
                    
                    payload = {
                        "query": rag_query,
                        "top_k": top_k,
                        "retrieval_method": retrieval_method,
                        "semantic_weight": semantic_weight,
                        "keyword_weight": keyword_weight
                    }
                    if filter_state:
                        payload["state"] = filter_state
                    if filter_crop:
                        payload["crop"] = filter_crop
                    if filter_category:
                        payload["category"] = filter_category
                    
                    response = requests.post(f"{api_url}/rag/query", json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.rag_query_results = data.get("results", [])
                    else:
                        st.error(f"Search failed: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    # Local fallback
                    try:
                        from api.rag_system import get_rag_system
                        rag = get_rag_system()
                        
                        filters = {}
                        if filter_state:
                            filters["state"] = filter_state
                        if filter_crop:
                            filters["crop"] = filter_crop
                        if filter_category:
                            filters["category"] = filter_category
                        
                        results = rag.query(
                            query=rag_query,
                            top_k=top_k,
                            retrieval_method=retrieval_method,
                            metadata_filters=filters if filters else None,
                            semantic_weight=semantic_weight,
                            keyword_weight=keyword_weight
                        )
                        st.session_state.rag_query_results = [r.to_dict() for r in results]
                    except Exception as e:
                        st.error(f"Local search error: {e}")
                
                except Exception as e:
                    st.error(f"Search error: {e}")
        
        # Handle Chat with RAG
        if chat_btn and rag_query:
            with st.spinner("🤖 Getting RAG-enhanced response..."):
                try:
                    api_url = os.getenv("API_URL", "http://localhost:8000")
                    
                    payload = {
                        "message": rag_query,
                        "use_rag": True,
                        "top_k": top_k,
                        "api_key": st.session_state.get("grok_api_key")
                    }
                    if filter_state:
                        payload["state"] = filter_state
                    if filter_crop:
                        payload["crop"] = filter_crop
                    
                    response = requests.post(f"{api_url}/rag/chat", json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            st.markdown("### 💬 RAG-Enhanced Response")
                            st.markdown(data.get("response", "No response"))
                            
                            if data.get("sources"):
                                with st.expander("📚 Sources Used", expanded=False):
                                    for src in data["sources"]:
                                        st.markdown(f"- **{src.get('filename')}** (score: {src.get('score', 0):.3f})")
                        else:
                            st.error(f"Error: {data.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Chat failed: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ API not available. Using offline mode...")
                    # Fallback to regular chat with context note
                    if st.session_state.get("grok_api_key"):
                        context = {"state": filter_state or "India", "rag_query": rag_query}
                        response = call_grok_api(
                            f"Based on agricultural knowledge, please answer: {rag_query}",
                            context,
                            st.session_state.grok_api_key
                        )
                        st.markdown("### 💬 Response (without RAG)")
                        st.markdown(response)
                    else:
                        st.error("No API key available for chat")
                
                except Exception as e:
                    st.error(f"Chat error: {e}")
        
        # Display Search Results
        if st.session_state.rag_query_results:
            st.markdown("### 📋 Search Results")
            
            for i, result in enumerate(st.session_state.rag_query_results):
                with st.container():
                    score = result.get("score", 0)
                    score_color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
                    
                    st.markdown(f"**Result {i+1}** - Score: :{score_color}[{score:.3f}]")
                    st.markdown(f"*Source: {result.get('metadata', {}).get('filename', 'Unknown')}*")
                    
                    # Truncate content for display
                    content = result.get("content", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    
                    st.markdown(f"```\n{content}\n```")
                    
                    meta = result.get("metadata", {})
                    if meta:
                        meta_items = [f"{k}: {v}" for k, v in meta.items() if k not in ['filename', 'upload_date'] and v]
                        if meta_items:
                            st.caption(" | ".join(meta_items[:3]))
                    
                    st.divider()
        
        # RAG Stats
        st.markdown("### 📊 Knowledge Base Stats")
        try:
            api_url = os.getenv("API_URL", "http://localhost:8000")
            response = requests.get(f"{api_url}/rag/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Documents", stats.get("total_documents", 0))
                with stat_col2:
                    st.metric("Chunks", stats.get("total_chunks", 0))
                with stat_col3:
                    st.metric("Vocab Size", stats.get("vocabulary_size", 0))
        except:
            try:
                from api.rag_system import get_rag_system
                rag = get_rag_system()
                stats = rag.get_stats()
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Documents", stats.get("total_documents", 0))
                with stat_col2:
                    st.metric("Chunks", stats.get("total_chunks", 0))
                with stat_col3:
                    st.metric("Vocab Size", stats.get("vocabulary_size", 0))
            except:
                st.info("📊 Stats unavailable")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌾 India Crop Recommendation System v2.0 | Streamlit Edition</p>
    <p>Built for Indian Agriculture 🇮🇳 | 20 Sensor Inputs | RAG Knowledge Base | Grok AI Chatbot</p>
</div>
""", unsafe_allow_html=True)
