"""
Smart Irrigation & Crop Guidance System - Complete Interactive Web Interface
Combines: 20 Sensor IoT Data, AI Chat, Decision Engine, Crop Recommendations

Run with: streamlit run smart_irrigation_app.py
"""
import os
import sys
import json
import random
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, List, Any

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_irrigation.environmental_state import (
    SensorData, WeatherData, EnvironmentalFusionEngine
)
from smart_irrigation.decision_arbitration import (
    DecisionArbitrator, SafetyConstraints, IrrigationAction
)
from smart_irrigation.decision_report import DecisionReportGenerator
from smart_irrigation.config import CROP_PARAMS

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🌱 Smart Irrigation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Black, White, Blue theme
st.markdown("""
<style>
    /* Global theme */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2196F3;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .decision-execute {
        background: #1565C0;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }
    .decision-defer {
        background: #1976D2;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }
    .decision-skip {
        background: #2196F3;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }
    .decision-override {
        background: #0D47A1;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }
    .metric-card {
        background: #1565C0;
        padding: 1rem;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
    }
    .crop-card {
        background: #111111;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .guidance-card {
        background: #111111;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #2196F3;
        color: #ffffff;
    }
    .trigger-card {
        background: #111111;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .risk-card {
        background: #111111;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #1565C0;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .chat-wrapper {
        background: #111111;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #2196F3;
    }
    .chat-header {
        background: #1565C0;
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .chat-user {
        background: #1976D2;
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        text-align: left;
        font-size: 0.95rem;
    }
    .chat-ai {
        background: #111111;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border-left: 3px solid #2196F3;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-ai strong, .chat-ai b {
        color: #2196F3;
        font-weight: 600;
    }
    .sensor-box {
        background: #111111;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2196F3;
        color: #ffffff;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1565C0, #2196F3);
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2196F3 !important;
    }
    /* Labels and text */
    label, .stMarkdown, p, span {
        color: #ffffff !important;
    }
    /* Selectbox and input styling */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #2196F3 !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #1565C0 !important;
        color: #ffffff !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #2196F3 !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111111 !important;
        color: #2196F3 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1565C0 !important;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #2196F3 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
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

SAMPLE_IOT_DATA = {
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

STATE_COORDS = {
    "Maharashtra": (19.75, 75.71),
    "Gujarat": (22.26, 71.19),
    "Punjab": (31.15, 75.34),
    "Rajasthan": (27.02, 74.22),
    "Tamil Nadu": (11.13, 78.66),
    "Andhra Pradesh": (15.91, 79.74),
    "Telangana": (18.11, 79.02),
    "Uttar Pradesh": (26.85, 80.91),
    "West Bengal": (22.99, 87.85),
    "Himachal Pradesh": (31.10, 77.17),
    "Uttarakhand": (30.07, 79.49)
}

# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL LOCATION DATA: State → District → Mandal/Taluk → Village
# ═══════════════════════════════════════════════════════════════════════════════

LOCATION_HIERARCHY = {
    "Maharashtra": {
        "districts": {
            "Pune": {
                "mandals": {
                    "Haveli": ["Wagholi", "Lohegaon", "Kharadi", "Hadapsar", "Manjri"],
                    "Mulshi": ["Pirangut", "Paud", "Lavasa", "Hinjewadi", "Maan"],
                    "Junnar": ["Narayangaon", "Otur", "Ale", "Kukadi", "Nimgiri"],
                    "Baramati": ["Baramati Town", "Supe", "Morgaon", "Pandare", "Katewadi"],
                }
            },
            "Nashik": {
                "mandals": {
                    "Nashik": ["Panchavati", "Satpur", "Cidco", "Gangapur", "Makhmalabad"],
                    "Sinnar": ["Sinnar Town", "Vadner", "Ghoti", "Takli", "Pathare"],
                    "Igatpuri": ["Igatpuri Town", "Ghoti Budruk", "Talegaon", "Kasara", "Trimbak"],
                }
            },
            "Nagpur": {
                "mandals": {
                    "Nagpur Urban": ["Dharampeth", "Sitabuldi", "Sadar", "Gandhibagh", "Lakadganj"],
                    "Hingna": ["Hingna Town", "Wadi", "Butibori", "Kalmeshwar", "Fetri"],
                    "Kamptee": ["Kamptee Town", "Kanhan", "Mauda", "Parseoni", "Saoner"],
                }
            },
            "Ahmednagar": {
                "mandals": {
                    "Ahmednagar": ["Ahmednagar City", "Savedi", "Kedgaon", "Burudgaon", "Walki"],
                    "Shrirampur": ["Shrirampur Town", "Belapur", "Loni", "Rahata", "Kopargaon"],
                    "Sangamner": ["Sangamner Town", "Akole", "Ashvi", "Ozar", "Nimon"],
                }
            },
        }
    },
    "Gujarat": {
        "districts": {
            "Ahmedabad": {
                "mandals": {
                    "Ahmedabad City": ["Maninagar", "Vastrapur", "Bopal", "Gota", "Chandkheda"],
                    "Daskroi": ["Bavla", "Sanand", "Sarkhej", "Dholka", "Bagodara"],
                    "Dholka": ["Dholka Town", "Koth", "Ranpur", "Bhadaj", "Aslali"],
                }
            },
            "Surat": {
                "mandals": {
                    "Surat City": ["Adajan", "Vesu", "Piplod", "Athwa", "Katargam"],
                    "Choryasi": ["Sachin", "Hazira", "Ichhapore", "Mora", "Dumas"],
                    "Kamrej": ["Kamrej Town", "Kim", "Kosamba", "Kadodara", "Palsana"],
                }
            },
            "Rajkot": {
                "mandals": {
                    "Rajkot": ["Rajkot City", "Mavdi", "Kalawad", "Kothariya", "Aji"],
                    "Gondal": ["Gondal Town", "Jetpur", "Dhoraji", "Upleta", "Jamkandorna"],
                }
            },
        }
    },
    "Punjab": {
        "districts": {
            "Ludhiana": {
                "mandals": {
                    "Ludhiana East": ["Ludhiana City", "Dugri", "BRS Nagar", "Model Town", "Sarabha Nagar"],
                    "Khanna": ["Khanna Town", "Samrala", "Doraha", "Payal", "Machhiwara"],
                    "Jagraon": ["Jagraon Town", "Raikot", "Mullanpur", "Sudhar", "Sidhwan"],
                }
            },
            "Amritsar": {
                "mandals": {
                    "Amritsar": ["Amritsar City", "Majitha", "Ajnala", "Rayya", "Attari"],
                    "Tarn Taran": ["Tarn Taran Town", "Patti", "Khadur Sahib", "Bhikhiwind", "Naushera"],
                }
            },
            "Patiala": {
                "mandals": {
                    "Patiala": ["Patiala City", "Rajpura", "Nabha", "Samana", "Patran"],
                    "Sangrur": ["Sangrur Town", "Malerkotla", "Dhuri", "Sunam", "Moonak"],
                }
            },
        }
    },
    "Rajasthan": {
        "districts": {
            "Jaipur": {
                "mandals": {
                    "Jaipur": ["Jaipur City", "Sanganer", "Amer", "Jamwa Ramgarh", "Chaksu"],
                    "Chomu": ["Chomu Town", "Shahpura", "Phulera", "Sambhar", "Jobner"],
                }
            },
            "Jodhpur": {
                "mandals": {
                    "Jodhpur": ["Jodhpur City", "Mandore", "Osian", "Phalodi", "Shergarh"],
                    "Bilara": ["Bilara Town", "Pipar", "Bhopalgarh", "Luni", "Balesar"],
                }
            },
            "Udaipur": {
                "mandals": {
                    "Udaipur": ["Udaipur City", "Girwa", "Mavli", "Salumbar", "Sarada"],
                    "Rajsamand": ["Rajsamand Town", "Kumbhalgarh", "Nathdwara", "Amet", "Bhim"],
                }
            },
        }
    },
    "Tamil Nadu": {
        "districts": {
            "Chennai": {
                "mandals": {
                    "Chennai North": ["Tondiarpet", "Royapuram", "Madhavaram", "Manali", "Tiruvottiyur"],
                    "Chennai South": ["Mylapore", "Adyar", "Velachery", "Guindy", "Alandur"],
                }
            },
            "Coimbatore": {
                "mandals": {
                    "Coimbatore North": ["Coimbatore City", "Singanallur", "Vadavalli", "Thondamuthur", "Perur"],
                    "Coimbatore South": ["Pollachi", "Valparai", "Anamalai", "Sulur", "Mettupalayam"],
                }
            },
            "Madurai": {
                "mandals": {
                    "Madurai East": ["Madurai City", "Thiruparankundram", "Melur", "Kottampatti", "Vadipatti"],
                    "Madurai West": ["Usilampatti", "Peraiyur", "Tirumangalam", "Kallikudi", "Sedapatti"],
                }
            },
        }
    },
    "Andhra Pradesh": {
        "districts": {
            "Guntur": {
                "mandals": {
                    "Guntur": ["Guntur City", "Gorantla", "Prathipadu", "Chebrolu", "Pedakakani"],
                    "Tenali": ["Tenali Town", "Duggirala", "Kollipara", "Kakumanu", "Amruthalur"],
                    "Mangalagiri": ["Mangalagiri Town", "Tadepalli", "Namburu", "Kaza", "Lingamguntla"],
                }
            },
            "Krishna": {
                "mandals": {
                    "Vijayawada": ["Vijayawada City", "Gollapudi", "Nunna", "Kanuru", "Penamaluru"],
                    "Machilipatnam": ["Machilipatnam Town", "Bantumilli", "Guduru", "Pedana", "Challapalli"],
                }
            },
            "Visakhapatnam": {
                "mandals": {
                    "Visakhapatnam Urban": ["Visakhapatnam City", "Gajuwaka", "Pedagantyada", "Pendurthi", "Gopalapatnam"],
                    "Anakapalli": ["Anakapalli Town", "Yelamanchili", "Parawada", "Chodavaram", "Narsipatnam"],
                }
            },
        }
    },
    "Telangana": {
        "districts": {
            "Hyderabad": {
                "mandals": {
                    "Hyderabad": ["Secunderabad", "Begumpet", "Ameerpet", "Kukatpally", "Madhapur"],
                    "Charminar": ["Charminar", "Falaknuma", "Chandrayangutta", "Santosh Nagar", "Yakutpura"],
                }
            },
            "Rangareddy": {
                "mandals": {
                    "Rajendranagar": ["Rajendranagar", "Narsingi", "Puppalaguda", "Kokapet", "Gandipet"],
                    "Shamshabad": ["Shamshabad Town", "Kandukur", "Keshampet", "Amangal", "Chevella"],
                    "Ibrahimpatnam": ["Ibrahimpatnam", "Hayathnagar", "Peddemul", "Manchal", "Abdullapurmet"],
                }
            },
            "Warangal": {
                "mandals": {
                    "Warangal Urban": ["Warangal City", "Kazipet", "Hanamkonda", "Subedari", "Waddepally"],
                    "Jangaon": ["Jangaon Town", "Zaffergadh", "Devaruppula", "Raghunathpally", "Bachannapeta"],
                }
            },
        }
    },
    "Uttar Pradesh": {
        "districts": {
            "Lucknow": {
                "mandals": {
                    "Lucknow": ["Lucknow City", "Gomti Nagar", "Aliganj", "Indira Nagar", "Hazratganj"],
                    "Mohanlalganj": ["Mohanlalganj Town", "Bakshi Ka Talab", "Sarojini Nagar", "Malihabad", "Kakori"],
                }
            },
            "Varanasi": {
                "mandals": {
                    "Varanasi": ["Varanasi City", "Ramnagar", "Pindra", "Cholapur", "Kashi Vidyapeeth"],
                    "Chandauli": ["Chandauli Town", "Mughalsarai", "Sakaldiha", "Chakia", "Naugarh"],
                }
            },
            "Agra": {
                "mandals": {
                    "Agra": ["Agra City", "Etmadpur", "Fatehabad", "Kiraoli", "Bah"],
                    "Firozabad": ["Firozabad Town", "Shikohabad", "Jasrana", "Tundla", "Sirsaganj"],
                }
            },
        }
    },
    "West Bengal": {
        "districts": {
            "Kolkata": {
                "mandals": {
                    "Kolkata": ["Kolkata City", "Salt Lake", "New Town", "Howrah", "Dum Dum"],
                }
            },
            "North 24 Parganas": {
                "mandals": {
                    "Barasat": ["Barasat Town", "Madhyamgram", "Barrackpore", "Titagarh", "Khardaha"],
                    "Basirhat": ["Basirhat Town", "Taki", "Haroa", "Hasnabad", "Sandeshkhali"],
                }
            },
            "Hooghly": {
                "mandals": {
                    "Hooghly": ["Chinsurah", "Bandel", "Chandannagar", "Serampore", "Rishra"],
                    "Arambagh": ["Arambagh Town", "Goghat", "Pursurah", "Khanakul", "Tarakeswar"],
                }
            },
        }
    },
    "Himachal Pradesh": {
        "districts": {
            "Shimla": {
                "mandals": {
                    "Shimla": ["Shimla City", "Mashobra", "Kufri", "Fagu", "Theog"],
                    "Rampur": ["Rampur Town", "Narkanda", "Kumarsain", "Rohru", "Jubbal"],
                }
            },
            "Kangra": {
                "mandals": {
                    "Dharamshala": ["Dharamshala", "McLeodganj", "Kangra Town", "Palampur", "Baijnath"],
                    "Nurpur": ["Nurpur Town", "Jawali", "Fatehpur", "Indora", "Dehra"],
                }
            },
        }
    },
    "Uttarakhand": {
        "districts": {
            "Dehradun": {
                "mandals": {
                    "Dehradun": ["Dehradun City", "Mussoorie", "Rishikesh", "Doiwala", "Vikasnagar"],
                    "Chakrata": ["Chakrata Town", "Kalsi", "Tyuni", "Lakhamandal", "Sahiya"],
                }
            },
            "Haridwar": {
                "mandals": {
                    "Haridwar": ["Haridwar City", "Roorkee", "Laksar", "Bhagwanpur", "Narsan"],
                }
            },
            "Nainital": {
                "mandals": {
                    "Nainital": ["Nainital Town", "Bhimtal", "Haldwani", "Ramnagar", "Kaladhungi"],
                    "Kashipur": ["Kashipur Town", "Bajpur", "Gadarpur", "Jaspur", "Sitarganj"],
                }
            },
        }
    },
}

# Location coordinates for weather API (District/Mandal level)
LOCATION_COORDS = {
    # Maharashtra
    "Pune": (18.52, 73.86), "Nashik": (19.99, 73.79), "Nagpur": (21.15, 79.09), "Ahmednagar": (19.09, 74.74),
    "Haveli": (18.54, 73.95), "Mulshi": (18.51, 73.51), "Junnar": (19.21, 73.88), "Baramati": (18.15, 74.58),
    # Gujarat  
    "Ahmedabad": (23.02, 72.57), "Surat": (21.17, 72.83), "Rajkot": (22.30, 70.80),
    "Daskroi": (22.99, 72.41), "Choryasi": (21.12, 72.79), "Kamrej": (21.27, 72.95),
    # Punjab
    "Ludhiana": (30.90, 75.85), "Amritsar": (31.63, 74.87), "Patiala": (30.34, 76.39),
    "Khanna": (30.70, 76.22), "Jagraon": (30.79, 75.47),
    # Rajasthan
    "Jaipur": (26.91, 75.79), "Jodhpur": (26.29, 73.02), "Udaipur": (24.58, 73.71),
    "Chomu": (27.17, 75.72), "Bilara": (26.18, 73.70),
    # Tamil Nadu
    "Chennai": (13.08, 80.27), "Coimbatore": (11.01, 76.97), "Madurai": (9.92, 78.12),
    # Andhra Pradesh
    "Guntur": (16.31, 80.44), "Krishna": (16.18, 81.13), "Visakhapatnam": (17.69, 83.22),
    "Tenali": (16.24, 80.64), "Mangalagiri": (16.43, 80.57),
    # Telangana
    "Hyderabad": (17.39, 78.49), "Rangareddy": (17.24, 78.07), "Warangal": (17.98, 79.53),
    "Shamshabad": (17.24, 78.42), "Ibrahimpatnam": (17.15, 78.67),
    # Uttar Pradesh
    "Lucknow": (26.85, 80.95), "Varanasi": (25.32, 82.99), "Agra": (27.18, 78.02),
    # West Bengal
    "Kolkata": (22.57, 88.36), "Hooghly": (22.91, 88.39), "Barasat": (22.72, 88.48),
    # Himachal Pradesh
    "Shimla": (31.10, 77.17), "Kangra": (32.10, 76.27), "Dharamshala": (32.22, 76.32),
    # Uttarakhand
    "Dehradun": (30.32, 78.03), "Haridwar": (29.95, 78.16), "Nainital": (29.38, 79.45),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'sensor_values' not in st.session_state:
    # Initialize with sample data from Maharashtra by default
    st.session_state.sensor_values = SAMPLE_IOT_DATA["Maharashtra"].copy()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'decision_history' not in st.session_state:
    st.session_state.decision_history = []

if 'last_report' not in st.session_state:
    st.session_state.last_report = None

if 'grok_api_key' not in st.session_state:
    st.session_state.grok_api_key = os.getenv("GROK_API_KEY", "")

if 'weather_api_key' not in st.session_state:
    st.session_state.weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_season(month: int) -> str:
    if 6 <= month <= 10:
        return "kharif"
    elif month >= 10 or month <= 3:
        return "rabi"
    return "zaid"


def generate_random_sensors(base_moisture: float = 20.0, variation: float = 10.0) -> List[float]:
    return [round(max(5, min(50, base_moisture + random.uniform(-variation, variation))), 2) for _ in range(20)]


def calculate_crop_scores(sm: float, temp: float, month: int) -> List[Dict]:
    season = get_season(month)
    scores = []
    
    for crop, rules in CROP_RULES.items():
        score = 0
        if rules["sm_min"] <= sm <= rules["sm_max"]:
            optimal_sm = (rules["sm_min"] + rules["sm_max"]) / 2
            sm_score = 1 - abs(sm - optimal_sm) / (rules["sm_max"] - rules["sm_min"])
            score += sm_score * 0.4
        if rules["temp_min"] <= temp <= rules["temp_max"]:
            optimal_temp = (rules["temp_min"] + rules["temp_max"]) / 2
            temp_score = 1 - abs(temp - optimal_temp) / (rules["temp_max"] - rules["temp_min"])
            score += temp_score * 0.3
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
    if not api_key:
        return "⚠️ No API key provided. Please enter your Groq API key in the sidebar."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an expert Indian agricultural advisor and smart irrigation specialist. Help farmers with:
- Crop recommendations based on location, season, soil moisture
- Irrigation scheduling and water management
- Weather-based farming decisions
Be concise, practical, and use simple language."""
        
        context_str = ""
        if context:
            context_str = f"\n\nCurrent context: {json.dumps(context)}"
        
        models_to_try = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
        
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
                return "❌ Invalid API Key"
        
        return "❌ API Error"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_offline_response(message: str, context: Dict) -> str:
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
    
    if any(word in msg for word in ["irrigation", "water", "irrigate"]):
        return f"""💧 **Smart Irrigation Guidance:**

• Season: **{season.title()}**

**Recommendations:**
1. Morning irrigation (6-9 AM) reduces evaporation
2. Use drip irrigation to save 30-50% water
3. Apply mulching to retain soil moisture
4. Monitor weather forecast before irrigating
5. Use Decision Engine tab for AI-powered scheduling"""
    
    return """🌱 I'm your Smart Irrigation AI Assistant!

**I can help with:**
• **"What crops should I grow?"** - Get recommendations
• **"Irrigation tips"** - Water management advice  
• **"When to irrigate?"** - Smart scheduling

*Use the Decision Engine tab for AI-powered irrigation control!*"""

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🔑 API Configuration")
    
    st.markdown("##### 🤖 Groq Cloud API")
    st.caption("Get FREE key: [console.groq.com](https://console.groq.com)")
    grok_key = st.text_input(
        "Groq API Key",
        value=st.session_state.grok_api_key,
        type="password",
        placeholder="gsk_xxxxxxxxxxxx"
    )
    if grok_key != st.session_state.grok_api_key:
        st.session_state.grok_api_key = grok_key
    
    # Test Groq API Connection
    if st.button("🧪 Test Groq Connection", use_container_width=True):
        if grok_key:
            if not grok_key.startswith("gsk_"):
                st.error("❌ Invalid format. Key should start with 'gsk_'")
            else:
                with st.spinner("Testing connection..."):
                    try:
                        headers = {
                            "Authorization": f"Bearer {grok_key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 10
                        }
                        response = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=10
                        )
                        if response.status_code == 200:
                            st.success("✅ Groq API Connected Successfully!")
                        elif response.status_code == 401:
                            st.error("❌ Invalid API Key")
                        else:
                            st.error(f"❌ Error: {response.status_code}")
                    except requests.exceptions.Timeout:
                        st.error("❌ Connection Timeout")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter API key first")
    
    st.markdown("---")
    
    st.markdown("##### 🌤️ OpenWeatherMap API")
    weather_key = st.text_input(
        "Weather API Key",
        value=st.session_state.weather_api_key,
        type="password",
        placeholder="Enter API key"
    )
    if weather_key != st.session_state.weather_api_key:
        st.session_state.weather_api_key = weather_key
    
    # Test Weather API Connection
    if st.button("🧪 Test Weather Connection", use_container_width=True):
        if weather_key:
            with st.spinner("Testing connection..."):
                try:
                    url = f"https://api.openweathermap.org/data/2.5/weather?q=Delhi,IN&appid={weather_key}&units=metric"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Connected! Delhi: {data['main']['temp']}°C")
                    elif response.status_code == 401:
                        st.error("❌ Invalid API Key")
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except requests.exceptions.Timeout:
                    st.error("❌ Connection Timeout")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter API key first")
    
    st.markdown("---")
    
    st.header("📊 Sensor Statistics")
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
    st.markdown("**Smart Irrigation v2.0**")
    st.markdown("🌱 AI-Powered Decision System")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🌱 Smart Irrigation & Crop Guidance System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Decision System with 20 IoT Sensors, Weather Integration & Explainable Decisions</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 IoT Sensor Data", 
    "🚀 Decision Engine", 
    "🌾 Crop Recommendations",
    "🤖 AI Chatbot", 
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
            state_for_sample = st.selectbox("Load State Data", ["Select..."] + INDIAN_STATES, key="sample_state")
        
        with action_col2:
            if st.button("📊 Load Sample", use_container_width=True):
                if state_for_sample != "Select..." and state_for_sample in SAMPLE_IOT_DATA:
                    st.session_state.sensor_values = SAMPLE_IOT_DATA[state_for_sample].copy()
                    st.rerun()
        
        with action_col3:
            if st.button("🎲 Random Generate", use_container_width=True):
                st.session_state.sensor_values = generate_random_sensors(base_moisture=22.0, variation=8.0)
                st.rerun()
        
        with action_col4:
            if st.button("🗑️ Clear All", use_container_width=True):
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
                        key=f"sensor_{sensor_idx}"
                    )
    
    with col2:
        st.markdown("##### 🗺️ Sensor Field Layout (4x5 Grid)")
        
        # Legend using native Streamlit
        st.caption("🟢 ≥20% Optimal  |  🟠 10-19% Low  |  🔴 <10% Critical  |  ⚪ No Data")
        
        # Grid using native Streamlit columns (4x5 = 20 sensors)
        for row in range(5):
            cols = st.columns(4)
            for col_idx in range(4):
                sensor_idx = row * 4 + col_idx
                val = st.session_state.sensor_values[sensor_idx]
                
                with cols[col_idx]:
                    if val > 0:
                        if val >= 20:
                            icon = "💧"
                            status = "optimal"
                        elif val >= 10:
                            icon = "⚠️"
                            status = "low"
                        else:
                            icon = "🔴"
                            status = "critical"
                    else:
                        icon = "○"
                        status = "no-data"
                    
                    # Use st.container with colored background via metric
                    if status == "optimal":
                        st.success(f"**S{sensor_idx+1}** {icon}\n\n**{val:.1f}%**")
                    elif status == "low":
                        st.warning(f"**S{sensor_idx+1}** {icon}\n\n**{val:.1f}%**")
                    elif status == "critical":
                        st.error(f"**S{sensor_idx+1}** {icon}\n\n**{val:.1f}%**")
                    else:
                        st.info(f"**S{sensor_idx+1}** {icon}\n\n**{val:.1f}%**")
        
        # Timestamp
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
    # WEATHER + IRRIGATION PREDICTION SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("🌦️ Live Weather & Irrigation Prediction")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HIERARCHICAL LOCATION SELECTOR: State → District → Mandal → Village
    # ═══════════════════════════════════════════════════════════════════════════
    
    loc_col1, loc_col2, loc_col3, loc_col4 = st.columns(4)
    
    with loc_col1:
        selected_state = st.selectbox("🏛️ State", INDIAN_STATES, index=3, key="weather_state_tab1")
    
    # Get districts for selected state
    state_data = LOCATION_HIERARCHY.get(selected_state, {})
    districts = list(state_data.get("districts", {}).keys())
    
    with loc_col2:
        if districts:
            selected_district = st.selectbox("🏘️ District", ["Select..."] + districts, key="weather_district")
        else:
            selected_district = st.selectbox("🏘️ District", ["No data available"], key="weather_district_empty")
            selected_district = None
    
    # Get mandals for selected district
    mandals = []
    if selected_district and selected_district != "Select...":
        district_data = state_data.get("districts", {}).get(selected_district, {})
        mandals = list(district_data.get("mandals", {}).keys())
    
    with loc_col3:
        if mandals:
            selected_mandal = st.selectbox("📍 Mandal/Taluk", ["Select..."] + mandals, key="weather_mandal")
        else:
            selected_mandal = st.selectbox("📍 Mandal/Taluk", ["Select district first..."], key="weather_mandal_empty")
            selected_mandal = None
    
    # Get villages for selected mandal
    villages = []
    if selected_mandal and selected_mandal != "Select...":
        villages = state_data.get("districts", {}).get(selected_district, {}).get("mandals", {}).get(selected_mandal, [])
    
    with loc_col4:
        if villages:
            selected_village = st.selectbox("🏡 Village/Town", ["Select..."] + villages, key="weather_village")
        else:
            selected_village = st.selectbox("🏡 Village/Town", ["Select mandal first..."], key="weather_village_empty")
            selected_village = None
    
    # Display selected location path
    location_parts = [selected_state]
    if selected_district and selected_district != "Select...":
        location_parts.append(selected_district)
    if selected_mandal and selected_mandal != "Select...":
        location_parts.append(selected_mandal)
    if selected_village and selected_village != "Select...":
        location_parts.append(selected_village)
    
    location_path = " → ".join(location_parts)
    
    # Get coordinates for the most specific location
    weather_location = selected_village if (selected_village and selected_village != "Select...") else \
                       selected_mandal if (selected_mandal and selected_mandal != "Select...") else \
                       selected_district if (selected_district and selected_district != "Select...") else \
                       selected_state
    
    # Try to get coords from most specific to least specific
    lat, lon = LOCATION_COORDS.get(selected_mandal, 
               LOCATION_COORDS.get(selected_district, 
               STATE_COORDS.get(selected_state, (20.0, 78.0))))
    
    st.info(f"📍 **Selected Location:** {location_path}  |  📌 Coordinates: {lat}°N, {lon}°E")
    
    # Weather and Prediction columns
    weather_col1, weather_col2, weather_col3 = st.columns([1, 2, 2])
    
    with weather_col1:
        st.markdown("##### 🌾 Crop & Action")
        
        crop_for_prediction = st.selectbox("Crop Type", list(CROP_RULES.keys()), key="crop_tab1")
        
        if st.button("🌦️ Fetch Weather & Predict", type="primary", use_container_width=True):
            api_key = st.session_state.weather_api_key
            if not api_key:
                st.error("⚠️ Please set Weather API Key in the sidebar!")
            else:
                with st.spinner(f"Fetching weather for {weather_location}..."):
                    # Fetch weather from API using most specific location
                    weather_data = fetch_weather(weather_location, api_key)
                    
                    # If village/mandal fails, try district, then state
                    if not weather_data and selected_district and selected_district != "Select...":
                        weather_data = fetch_weather(selected_district, api_key)
                    if not weather_data:
                        weather_data = fetch_weather(selected_state, api_key)
                    
                    if weather_data:
                        st.session_state.live_weather = weather_data
                        st.session_state.weather_fetched = True
                        st.session_state.weather_location = location_path
                        st.success(f"✅ Weather fetched for {weather_data.get('city', weather_location)}")
                    else:
                        st.error("❌ Failed to fetch weather. Check API key.")
    
    with weather_col2:
        st.markdown("##### 🌤️ Current Weather")
        
        if 'live_weather' in st.session_state and st.session_state.get('weather_fetched'):
            weather = st.session_state.live_weather
            
            # Show fetched location
            if 'weather_location' in st.session_state:
                st.caption(f"📍 {st.session_state.weather_location}")
            
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.metric("🌡️ Temperature", f"{weather.get('temp', 'N/A')}°C")
                st.metric("💨 Description", weather.get('description', 'N/A').title())
            with wcol2:
                st.metric("💧 Humidity", f"{weather.get('humidity', 'N/A')}%")
                st.metric("📍 City", weather.get('city', 'N/A'))
        else:
            st.info("👆 Click 'Fetch Weather & Predict' to load live weather data")
    
    with weather_col3:
        st.markdown("##### 🚿 Irrigation Prediction")
        
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        
        if 'live_weather' in st.session_state and st.session_state.get('weather_fetched') and len(sensor_vals) >= 5:
            weather = st.session_state.live_weather
            avg_moisture = sum(sensor_vals) / len(sensor_vals)
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
        
        elif len(sensor_vals) < 5:
            st.warning("⚠️ Need at least 5 sensor readings for prediction")
        else:
            st.info("👆 Fetch weather first to get irrigation prediction")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🚀 Smart Irrigation Decision Engine")
    st.caption("AI-powered decision arbitration with EXECUTE/DEFER/OVERRIDE logic")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("##### 📍 Location & Crop")
        state = st.selectbox("State", INDIAN_STATES, index=3, key="decision_state")
        
        lat, lon = STATE_COORDS.get(state, (20.0, 78.0))
        st.caption(f"Coordinates: {lat}°N, {lon}°E")
        
        crop_type = st.selectbox("Crop Type", ["wheat", "rice", "maize", "cotton", "sugarcane", "soybean"], key="decision_crop")
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
                try:
                    sensor = SensorData(
                        soil_moisture_vwc=avg_moisture / 100,
                        soil_temperature_c=ambient_temp - 5,
                        ambient_temperature_c=ambient_temp,
                        ambient_humidity_pct=humidity,
                        timestamp=datetime.now(timezone.utc),
                        reliability_score=0.9
                    )
                    
                    weather = WeatherData(
                        forecast_rain_24h_mm=forecast_rain,
                        forecast_rain_48h_mm=forecast_rain * 1.5,
                        forecast_temp_max_c=ambient_temp + 5,
                        forecast_temp_min_c=ambient_temp - 8,
                        forecast_humidity_pct=humidity,
                        forecast_wind_speed_ms=3.0,
                        forecast_cloud_cover_pct=50.0,
                        rain_probability_pct=rain_probability,
                        weather_description="forecast",
                        forecast_timestamp=datetime.now(timezone.utc),
                        confidence_score=0.85
                    )
                    
                    constraints = SafetyConstraints(
                        equipment_available=equipment_available,
                        water_quota_remaining_mm=water_quota
                    )
                    
                    fusion = EnvironmentalFusionEngine(CROP_PARAMS)
                    env_state = fusion.fuse(
                        sensor_data=sensor,
                        weather_data=weather,
                        crop_type=crop_type,
                        days_after_sowing=days_after_sowing,
                        latitude=lat
                    )
                    
                    arbitrator = DecisionArbitrator(constraints)
                    result = arbitrator.arbitrate(env_state)
                    
                    from smart_irrigation.decision_arbitration import ActuationController
                    controller = ActuationController()
                    control_signal = controller.generate_control_signal(result)
                    
                    report_gen = DecisionReportGenerator()
                    report = report_gen.generate(
                        state=env_state,
                        result=result,
                        control_signal=control_signal,
                        location={"state": state, "lat": lat, "lon": lon}
                    )
                    
                    st.session_state.last_report = report
                    st.session_state.decision_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "decision": result.action.value,
                        "confidence": result.confidence_score,
                        "moisture": avg_moisture
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating decision: {str(e)}")
        
        if st.session_state.last_report:
            report = st.session_state.last_report
            
            st.markdown("---")
            
            decision = report.irrigation_decision
            
            if decision == "EXECUTE":
                st.markdown('<div class="decision-execute">✅ EXECUTE IRRIGATION</div>', unsafe_allow_html=True)
            elif decision == "DEFER":
                st.markdown('<div class="decision-defer">⏸️ DEFER IRRIGATION</div>', unsafe_allow_html=True)
            elif decision == "SKIP":
                st.markdown('<div class="decision-skip">⏭️ SKIP - No Action Needed</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="decision-override">⚠️ OVERRIDE - Safety Check</div>', unsafe_allow_html=True)
            
            st.markdown("")
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Confidence", f"{report.decision_confidence:.0%}")
            with mcol2:
                env = report.environmental_summary
                st.metric("Current VWC", f"{env.get('current_vwc', 0):.0%}")
            with mcol3:
                st.metric("Predicted 24h", f"{env.get('predicted_vwc_24h', 0):.0%}")
            with mcol4:
                st.metric("ET Demand", f"{env.get('etc_mm_day', 0):.1f} mm/day")
            
            st.markdown("##### 🌾 Crop Guidance")
            guidance = report.crop_guidance
            st.markdown(f"""
            <div class="guidance-card">
                <strong>💧 Water Demand:</strong> {guidance.water_demand_explanation}<br><br>
                <strong>📝 Action Rationale:</strong> {guidance.action_rationale}
            </div>
            """, unsafe_allow_html=True)
            
            triggers = report.arbitration_details.get('triggered_conditions', [])
            if triggers:
                st.markdown("##### ⚡ Decision Triggers")
                for trigger in triggers:
                    st.markdown(f'<div class="trigger-card"><strong>{trigger.get("type", "").replace("_", " ").title()}</strong>: {trigger.get("description", "")}</div>', unsafe_allow_html=True)
            
            if decision == "EXECUTE":
                st.markdown("##### 💧 Irrigation Prescription")
                arb = report.arbitration_details
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.metric("Water Amount", f"{arb.get('recommended_water_mm', 0):.1f} mm")
                with pcol2:
                    st.metric("Duration", f"{arb.get('recommended_duration_minutes', 0):.0f} min")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: CROP RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🌾 Crop Recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### 📍 Location & Conditions")
        rec_state = st.selectbox("State", INDIAN_STATES, index=3, key="rec_state")
        rec_temp = st.number_input("Temperature (°C)", value=28.0, min_value=0.0, max_value=50.0, key="rec_temp")
        rec_date = st.date_input("Planting Date", value=date.today(), key="rec_date")
    
    with col2:
        st.markdown("##### 💧 Sensor Summary")
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        if sensor_vals:
            avg = sum(sensor_vals) / len(sensor_vals)
            st.metric("Average Soil Moisture", f"{avg:.1f}%")
            st.caption(f"From {len(sensor_vals)} active sensors")
        else:
            avg = 25.0
            st.info("Using default moisture. Add sensors in Tab 1.")
    
    if st.button("🌾 Get Recommendations", type="primary", use_container_width=True):
        recommendations = calculate_crop_scores(avg, rec_temp, rec_date.month)
        
        if recommendations:
            st.success(f"Found {len(recommendations)} suitable crops for {get_season(rec_date.month).title()} season!")
            
            for rec in recommendations:
                col_a, col_b, col_c = st.columns([1, 4, 1])
                with col_a:
                    st.markdown(f"### {rec['emoji']}")
                with col_b:
                    st.markdown(f"**{rec['crop']}** - {rec['season'].title()} season")
                    st.caption(rec['notes'])
                with col_c:
                    st.metric("Match", f"{int(rec['confidence']*100)}%")
                st.progress(rec["confidence"])
                st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🤖 AI Farming & Irrigation Assistant | Powered by Groq</div>', unsafe_allow_html=True)
    
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown('''
            <div class="chat-ai">
                👋 <strong>Hello! I'm your Smart Irrigation AI Assistant.</strong><br><br>
                I can help you with:<br>
                • <strong>Irrigation scheduling</strong> and water management<br>
                • <strong>Crop recommendations</strong> for your region<br>
                • <strong>Sensor data analysis</strong> and insights<br>
                • <strong>Weather-based</strong> farming decisions<br><br>
                Try asking: "When should I irrigate my wheat field?"
            </div>
            ''', unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    content = msg["content"].replace('\n', '<br>').replace('**', '<strong>').replace('*', '<em>')
                    st.markdown(f'<div class="chat-ai">{content}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("**Quick Actions:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    
    with qcol1:
        if st.button("🌾 Crop Advice"):
            st.session_state.pending_message = "What crops should I grow based on my current soil moisture?"
    with qcol2:
        if st.button("💧 Irrigation Tips"):
            st.session_state.pending_message = "When should I irrigate my field based on current sensor readings?"
    with qcol3:
        if st.button("📊 Analyze Sensors"):
            st.session_state.pending_message = "Analyze my soil moisture sensor data and provide insights"
    with qcol4:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    if 'pending_message' in st.session_state:
        user_input = st.session_state.pending_message
        del st.session_state.pending_message
    else:
        user_input = None
    
    user_message = st.chat_input("Ask about irrigation, crops, sensors...")
    
    if user_message or user_input:
        message = user_message or user_input
        
        st.session_state.chat_history.append({"role": "user", "content": message})
        
        sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
        context = {
            "state": "Maharashtra",
            "month": datetime.now().month,
            "season": get_season(datetime.now().month)
        }
        if sensor_vals:
            context["soil_moisture"] = sum(sensor_vals) / len(sensor_vals)
            context["sensor_count"] = len(sensor_vals)
            context["min_moisture"] = min(sensor_vals)
            context["max_moisture"] = max(sensor_vals)
        
        with st.spinner("🤖 Thinking..."):
            if st.session_state.grok_api_key:
                response = call_grok_api(message, context, st.session_state.grok_api_key)
            else:
                response = get_offline_response(message, context)
        
        st.session_state.chat_history.append({"role": "ai", "content": response})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("📊 Sensor Data Analysis & Decision History")
    
    sensor_vals = [v for v in st.session_state.sensor_values if v > 0]
    
    if sensor_vals:
        df = pd.DataFrame({
            "Sensor": [f"S{i+1}" for i, v in enumerate(st.session_state.sensor_values) if v > 0],
            "Moisture (%)": sensor_vals
        })
        
        st.markdown("##### Soil Moisture Distribution")
        st.bar_chart(df.set_index("Sensor"))
        
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
        
        avg = sum(sensor_vals) / len(sensor_vals)
        st.markdown("##### Field Moisture Interpretation")
        if avg < 15:
            st.error("⚠️ **Critical**: Immediate irrigation required. Crop stress likely.")
        elif avg < 22:
            st.warning("💧 **Low**: Schedule irrigation soon. Monitor closely.")
        elif avg < 35:
            st.success("✅ **Optimal**: Good conditions for most crops.")
        else:
            st.info("🌊 **High**: Adequate moisture. Check drainage if persistent.")
    else:
        st.info("📝 Enter sensor readings in the IoT Sensor Data tab")
    
    if st.session_state.decision_history:
        st.markdown("---")
        st.markdown("##### Decision History")
        
        history_df = pd.DataFrame(st.session_state.decision_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.decision_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌱 <strong>Smart Irrigation & Crop Guidance System v2.0</strong></p>
    <p>20 IoT Sensors | AI Decision Engine | Weather Integration | Explainable Outputs</p>
    <p><small>Built for Indian Agriculture 🇮🇳</small></p>
</div>
""", unsafe_allow_html=True)
