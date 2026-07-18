# ============================================================
#  AquaSense API — FastAPI Backend
#  Simulated Annealing দিয়ে Water Quality Analysis
# ============================================================
import os
import json          # ← নতুন লাইন যোগ করো
import re  
import base64
import google.generativeai as genai
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import random
import math
from typing import Optional

app = FastAPI(title="AquaSense API", version="1.0.0")

# ── CORS — Flutter app থেকে access করতে দাও ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sensor Data Model ──
class SensorData(BaseModel):
    temperature: float      # পানির তাপমাত্রা (°C)
    ammonia: float          # NH₃ (ppm)
    tds: float              # TDS (ppm)
    humidity: float         # আর্দ্রতা (%)
    air_temp: float         # বাতাসের তাপমাত্রা (°C)
    pond_id: str            # কোন পুকুরের data
    readings_count: Optional[int] = 1  # কতটা reading পাঠাচ্ছে

# ── Weekly Analysis Model (QuantumCycle এর জন্য — ভবিষ্যতে) ──
class WeeklyData(BaseModel):
    pond_id: str
    readings: list[dict]   # ৭ দিনের সব readings

# ══════════════════════════════════════════════════════
#  COST FUNCTION
#  পানির quality কতটা ভালো বা খারাপ সেটা একটা
#  single number এ প্রকাশ করে (কম হলে ভালো)
# ══════════════════════════════════════════════════════

def calculate_cost(weights: dict, data: SensorData) -> float:
    """
    Cost Function = weighted sum of parameter deviations
    প্রতিটা parameter কতটা safe range থেকে দূরে সেটা measure করে
    """

    # Safe ranges (Bangladesh aquaculture standards)
    safe_ranges = {
        'temperature': (24, 30),   # °C
        'ammonia':     (0, 0.25),  # ppm
        'tds':         (200, 500), # ppm
        'humidity':    (60, 85),   # %
    }

    cost = 0.0

    # Temperature cost
    t = data.temperature
    t_min, t_max = safe_ranges['temperature']
    if t < t_min:
        t_deviation = (t_min - t) / t_min
    elif t > t_max:
        t_deviation = (t - t_max) / t_max
    else:
        t_deviation = 0.0
    cost += weights['temperature'] * t_deviation

    # Ammonia cost (সবচেয়ে গুরুত্বপূর্ণ)
    a = data.ammonia
    a_max = safe_ranges['ammonia'][1]
    if a > a_max:
        a_deviation = (a - a_max) / a_max
    else:
        a_deviation = 0.0
    cost += weights['ammonia'] * a_deviation

    # TDS cost
    td = data.tds
    td_min, td_max = safe_ranges['tds']
    if td < td_min:
        td_deviation = (td_min - td) / td_min
    elif td > td_max:
        td_deviation = (td - td_max) / td_max
    else:
        td_deviation = 0.0
    cost += weights['tds'] * td_deviation

    # Humidity cost
    h = data.humidity
    h_min, h_max = safe_ranges['humidity']
    if h < h_min:
        h_deviation = (h_min - h) / h_min
    elif h > h_max:
        h_deviation = (h - h_max) / h_max
    else:
        h_deviation = 0.0
    cost += weights['humidity'] * h_deviation

    return cost

# ══════════════════════════════════════════════════════
#  SIMULATED ANNEALING
#  সবচেয়ে ভালো parameter weights খুঁজে বের করে
#  যেটা দিয়ে পানির quality সবচেয়ে ভালোভাবে measure হয়
# ══════════════════════════════════════════════════════

def simulated_annealing(data: SensorData) -> dict:
    """
    Simulated Annealing Algorithm:
    
    ধাপ ১: Random weights দিয়ে শুরু করো
    ধাপ ২: একটু weights পরিবর্তন করো
    ধাপ ৩: নতুন weights ভালো হলে রাখো
    ধাপ ৪: তাপমাত্রা কমাও (ধীরে ধীরে explore কম করো)
    ধাপ ৫: ৫০০ বার repeat করো
    ধাপ ৬: সবচেয়ে ভালো weights return করো
    """

    # Initial weights — সব parameter সমান গুরুত্ব দিয়ে শুরু
    current_weights = {
        'temperature': 0.25,
        'ammonia':     0.25,
        'tds':         0.25,
        'humidity':    0.25,
    }

    current_cost = calculate_cost(current_weights, data)
    best_weights = current_weights.copy()
    best_cost = current_cost

    # Annealing parameters
    temperature = 1.0      # Initial "temperature" (high = more random exploration)
    cooling_rate = 0.995   # প্রতি iteration এ কতটা ঠান্ডা হবে
    min_temperature = 0.01 # কতটা ঠান্ডা হলে থামবে
    iterations = 500       # কতবার try করবে

    for i in range(iterations):
        if temperature < min_temperature:
            break

        # একটা random parameter এর weight একটু পরিবর্তন করো
        param = random.choice(list(current_weights.keys()))
        delta = random.uniform(-0.1, 0.1)

        new_weights = current_weights.copy()
        new_weights[param] = max(0.05, min(0.95, new_weights[param] + delta))

        # Normalize — সব weights এর যোগফল = 1
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}

        new_cost = calculate_cost(new_weights, data)
        cost_diff = new_cost - current_cost

        # ভালো solution হলে সবসময় accept করো
        # খারাপ solution হলে কিছুটা সম্ভাবনায় accept করো (explore করার জন্য)
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_weights = new_weights
            current_cost = new_cost

            if current_cost < best_cost:
                best_weights = current_weights.copy()
                best_cost = current_cost

        # ধীরে ধীরে ঠান্ডা হচ্ছে
        temperature *= cooling_rate

    return {
        'weights': best_weights,
        'final_cost': best_cost,
        'iterations_run': i + 1,
    }

# ══════════════════════════════════════════════════════
#  PREDICTION ENGINE
#  Optimized weights ব্যবহার করে ভবিষ্যৎ অবস্থা predict করে
# ══════════════════════════════════════════════════════

def generate_predictions(weights: dict, data: SensorData) -> list:
    predictions = []

    # NH₃ prediction — সবচেয়ে গুরুত্বপূর্ণ parameter
    nh3_weight = weights['ammonia']
    if data.ammonia > 0.20:
        trend = "বাড়তে পারে ⚠️"
        warn = True
        detail = f"বর্তমান {data.ammonia:.2f} ppm — safe limit (0.25 ppm) এর কাছাকাছি"
    elif data.ammonia > 0.15:
        trend = "সামান্য বাড়তে পারে"
        warn = False
        detail = f"বর্তমান {data.ammonia:.2f} ppm — এখনো safe কিন্তু নজর রাখুন"
    else:
        trend = "স্বাভাবিক থাকবে ✅"
        warn = False
        detail = f"বর্তমান {data.ammonia:.2f} ppm — ভালো অবস্থায় আছে"

    predictions.append({
        'parameter': 'অ্যামোনিয়া (NH₃)',
        'icon': '☠️',
        'trend': trend,
        'warn': warn,
        'detail': detail,
        'weight': round(nh3_weight, 3),
        'current_value': data.ammonia,
        'unit': 'ppm',
    })

    # Temperature prediction
    temp_weight = weights['temperature']
    if data.temperature > 31:
        trend = "আরও বাড়তে পারে ⚠️"
        warn = True
        detail = f"বর্তমান {data.temperature:.1f}°C — মাছের জন্য অস্বস্তিকর হতে পারে"
    elif data.temperature < 22:
        trend = "আরও কমতে পারে ⚠️"
        warn = True
        detail = f"বর্তমান {data.temperature:.1f}°C — মাছের বৃদ্ধি ধীর হতে পারে"
    else:
        trend = "স্বাভাবিক থাকবে ✅"
        warn = False
        detail = f"বর্তমান {data.temperature:.1f}°C — আদর্শ তাপমাত্রা"

    predictions.append({
        'parameter': 'পানির তাপমাত্রা',
        'icon': '🌡️',
        'trend': trend,
        'warn': warn,
        'detail': detail,
        'weight': round(temp_weight, 3),
        'current_value': data.temperature,
        'unit': '°C',
    })

    # TDS prediction
    tds_weight = weights['tds']
    if data.tds > 450:
        trend = "বাড়তে পারে ⚠️"
        warn = True
        detail = f"বর্তমান {data.tds:.0f} ppm — পানি পরিবর্তন প্রয়োজন হতে পারে"
    elif data.tds < 200:
        trend = "কম আছে ⚠️"
        warn = True
        detail = f"বর্তমান {data.tds:.0f} ppm — পানিতে mineral কম"
    else:
        trend = "স্বাভাবিক থাকবে ✅"
        warn = False
        detail = f"বর্তমান {data.tds:.0f} ppm — ভালো অবস্থায় আছে"

    predictions.append({
        'parameter': 'পানির লবণ (TDS)',
        'icon': '🧪',
        'trend': trend,
        'warn': warn,
        'detail': detail,
        'weight': round(tds_weight, 3),
        'current_value': data.tds,
        'unit': 'ppm',
    })

    # Humidity prediction
    hum_weight = weights['humidity']
    if data.humidity > 88:
        trend = "বেশি থাকবে ⚠️"
        warn = True
        detail = f"বর্তমান {data.humidity:.0f}% — ছত্রাকের ঝুঁকি বাড়তে পারে"
    else:
        trend = "স্বাভাবিক থাকবে ✅"
        warn = False
        detail = f"বর্তমান {data.humidity:.0f}% — স্বাভাবিক অবস্থায় আছে"

    predictions.append({
        'parameter': 'বায়ু আর্দ্রতা',
        'icon': '💧',
        'trend': trend,
        'warn': warn,
        'detail': detail,
        'weight': round(hum_weight, 3),
        'current_value': data.humidity,
        'unit': '%',
    })

    return predictions

# ══════════════════════════════════════════════════════
#  ADVICE GENERATOR
#  Predictions দেখে বাংলায় পরামর্শ দেয়
# ══════════════════════════════════════════════════════

def generate_advice(predictions: list, data: SensorData, cost: float) -> list:
    advice = []
    has_warning = any(p['warn'] for p in predictions)

    if data.ammonia > 0.25:
        advice.append({
            'icon': '💧',
            'title': 'জরুরি: পানি পরিবর্তন করুন',
            'body': 'পুকুরের ৩০% পানি এখনই পরিবর্তন করুন। NH₃ এর মাত্রা বিপজ্জনক পর্যায়ে।',
            'priority': 'high',
        })
    elif data.ammonia > 0.20:
        advice.append({
            'icon': '⚠️',
            'title': 'সতর্কতা: NH₃ বাড়ছে',
            'body': 'আজকের মধ্যে ২০% পানি পরিবর্তন করুন এবং খাবারের পরিমাণ কমান।',
            'priority': 'medium',
        })

    if data.temperature > 32:
        advice.append({
            'icon': '🌡️',
            'title': 'তাপমাত্রা বেশি',
            'body': 'পুকুরে ছায়ার ব্যবস্থা করুন। Aerator চালু রাখুন।',
            'priority': 'medium',
        })

    if data.tds > 500:
        advice.append({
            'icon': '🧪',
            'title': 'TDS বেশি',
            'body': 'বিশুদ্ধ পানি যুক্ত করুন। পানির লবণ কমানো দরকার।',
            'priority': 'medium',
        })

    if not has_warning:
        advice.append({
            'icon': '😊',
            'title': 'পুকুরের অবস্থা ভালো!',
            'body': 'সব কিছু স্বাভাবিক আছে। নিয়মিত পর্যবেক্ষণ চালিয়ে যান।',
            'priority': 'low',
        })

    return advice

# ══════════════════════════════════════════════════════
#  HEALTH SCORE
# ══════════════════════════════════════════════════════

def calculate_health_score(cost: float) -> float:
    # Cost যত কম, health score তত বেশি
    score = max(0, min(100, 100 - (cost * 200)))
    return round(score, 1)

# ══════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "AquaSense API চলছে! 🐟",
        "version": "1.0.0",
        "algorithm": "Simulated Annealing",
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "algorithm": "Simulated Annealing"}

@app.post("/api/analyze")
def analyze(data: SensorData):
    """
    Main endpoint — Flutter app এখানে sensor data পাঠাবে
    Simulated Annealing চালিয়ে result return করবে
    """

    # ── Simulated Annealing চালাও ──
    sa_result = simulated_annealing(data)
    optimized_weights = sa_result['weights']
    final_cost = sa_result['final_cost']

    # ── Predictions বানাও ──
    predictions = generate_predictions(optimized_weights, data)

    # ── Advice বানাও ──
    advice = generate_advice(predictions, data, final_cost)

    # ── Health Score ──
    health_score = calculate_health_score(final_cost)

    # ── Overall Status ──
    if health_score >= 85:
        overall_status = "চমৎকার ✅"
        overall_color = "green"
    elif health_score >= 70:
        overall_status = "ভালো ✅"
        overall_color = "green"
    elif health_score >= 50:
        overall_status = "মোটামুটি ⚠️"
        overall_color = "yellow"
    else:
        overall_status = "সতর্কতা প্রয়োজন 🔴"
        overall_color = "red"

    return {
        "pond_id": data.pond_id,
        "algorithm": "Simulated Annealing",
        "iterations": sa_result['iterations_run'],
        "optimized_weights": {k: round(v, 4) for k, v in optimized_weights.items()},
        "final_cost": round(final_cost, 6),
        "health_score": health_score,
        "overall_status": overall_status,
        "overall_color": overall_color,
        "predictions": predictions,
        "advice": advice,
        "fish_health": {
            "growth_rate": "স্বাভাবিক ✅" if data.temperature >= 24 and data.ammonia < 0.25 else "ধীর হতে পারে ⚠️",
            "disease_risk": "মাঝারি ⚠️" if data.ammonia > 0.20 or data.temperature > 32 else "কম 🟢",
            "feeding_advice": "খাবার কমিয়ে দিন ⚠️" if data.ammonia > 0.25 else "স্বাভাবিক দিন ✅",
        }
    }

@app.post("/api/analyze/weekly")
def analyze_weekly(data: WeeklyData):
    """
    Weekly analysis — QuantumCycle এর জন্য (ভবিষ্যতে IBM Quantum এ যাবে)
    এখন: ৭ দিনের average দিয়ে Simulated Annealing চালাবে
    """
    if not data.readings:
        return {"error": "কোনো reading নেই"}

    # ৭ দিনের average calculate করো
    avg_temp = sum(r['temperature'] for r in data.readings) / len(data.readings)
    avg_nh3  = sum(r['ammonia'] for r in data.readings) / len(data.readings)
    avg_tds  = sum(r['tds'] for r in data.readings) / len(data.readings)
    avg_hum  = sum(r['humidity'] for r in data.readings) / len(data.readings)
    avg_air  = sum(r.get('air_temp', 30) for r in data.readings) / len(data.readings)

    weekly_sensor = SensorData(
        temperature=avg_temp,
        ammonia=avg_nh3,
        tds=avg_tds,
        humidity=avg_hum,
        air_temp=avg_air,
        pond_id=data.pond_id,
        readings_count=len(data.readings),
    )

    result = analyze(weekly_sensor)
    result['analysis_type'] = 'weekly'
    result['readings_analyzed'] = len(data.readings)
    result['note'] = 'QuantumCycle: ভবিষ্যতে এটা IBM Quantum এ চলবে'

    return result







# ============================================================
#  এই code টা তোমার main.py এর একদম শেষে যুক্ত করবে
#  (if __name__ == "__main__": এর আগে)
# ============================================================



# ── Gemini Setup ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Firebase Admin Setup (quota tracking এর জন্য) ──
# Firestore এ quota save করবো
def get_firestore_client():
    try:
        if not firebase_admin._apps:
            # Service account JSON Render env variable থেকে নেবে
            import json
            service_account_info = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")
            if service_account_info:
                cred = credentials.Certificate(json.loads(service_account_info))
                initialize_app(cred)
        return firestore.client()
    except Exception:
        return None

# ── Fish Symptom Analysis Endpoint ──
@app.post("/api/fish-symptom")
async def analyze_fish_symptom(
    image: UploadFile = File(...),
    pond_id: str = Form(default="unknown"),
    user_id: str = Form(default="unknown"),
):
    """
    Flutter app থেকে মাছের ছবি আসবে
    Gemini AI দিয়ে disease detect করবে
    Firestore এ quota update করবে
    """

    if not GEMINI_API_KEY:
        return {
            "success": False,
            "error": "Gemini API key নেই। Admin প্যানেল থেকে key set করুন।"
        }

    try:
        # ── ছবি পড়ো ──
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # ── Gemini Model ──
        model = genai.GenerativeModel('gemini-2.0-flash')

        # ── Prompt — বাংলায় result চাই ──
        prompt = """
        তুমি একজন বিশেষজ্ঞ মৎস্য রোগ বিশেষজ্ঞ। এই মাছের ছবি দেখে বিশ্লেষণ করো।

        নিচের JSON format এ উত্তর দাও (শুধু JSON, অন্য কিছু না):
        {
            "disease_detected": true/false,
            "disease_name": "রোগের নাম ইংরেজিতে",
            "disease_name_bangla": "রোগের নাম বাংলায়",
            "confidence": 85,
            "symptoms_found": ["লক্ষণ ১", "লক্ষণ ২"],
            "cause": "কারণ বাংলায়",
            "treatment": ["চিকিৎসা ১", "চিকিৎসা ২", "চিকিৎসা ৩"],
            "prevention": ["প্রতিরোধ ১", "প্রতিরোধ ২"],
            "urgency": "high/medium/low",
            "recommendation": "সংক্ষিপ্ত পরামর্শ বাংলায়",
            "is_healthy": true/false
        }

        যদি মাছ সুস্থ থাকে:
        - disease_detected: false
        - is_healthy: true
        - disease_name_bangla: "মাছ সুস্থ আছে"

        যদি ছবিতে মাছ না থাকে:
        - disease_detected: false  
        - disease_name_bangla: "মাছের ছবি পাওয়া যায়নি"
        - recommendation: "মাছের স্পষ্ট ছবি তুলুন"
        """

        # ── Gemini API Call ──
        response = model.generate_content([
            prompt,
            {
                "mime_type": image.content_type or "image/jpeg",
                "data": image_base64
            }
        ])

        response_text = response.text.strip()

        # JSON extract করো (markdown code block থাকলে সরাও)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
        else:
            result_data = json.loads(response_text)

        # ── Firestore এ Quota Update করো ──
        try:
            db = get_firestore_client()
            if db:
                quota_ref = db.collection('ai_usage').document('fish_symptom_quota')
                quota_ref.set({
                    'total_calls': firestore.Increment(1),
                    'last_used': firestore.SERVER_TIMESTAMP,
                    'last_pond_id': pond_id,
                    'last_user_id': user_id,
                }, merge=True)

                # Per-user tracking
                user_ref = db.collection('ai_usage').document(f'user_{user_id}')
                user_ref.set({
                    'fish_symptom_calls': firestore.Increment(1),
                    'last_used': firestore.SERVER_TIMESTAMP,
                }, merge=True)
        except Exception as quota_err:
            # Quota update fail হলেও result return করো
            print(f"Quota update error: {quota_err}")

        return {
            "success": True,
            "analysis": result_data,
            "model": "gemini-2.0-flash",
        }

    except json.JSONDecodeError:
        # Gemini valid JSON না দিলে raw text parse করার চেষ্টা
        return {
            "success": True,
            "analysis": {
                "disease_detected": False,
                "disease_name_bangla": "বিশ্লেষণ সম্পন্ন",
                "confidence": 70,
                "symptoms_found": [],
                "treatment": ["মৎস্য বিশেষজ্ঞের পরামর্শ নিন"],
                "prevention": ["নিয়মিত পর্যবেক্ষণ করুন"],
                "urgency": "low",
                "recommendation": response.text[:200] if response.text else "ছবি আরও স্পষ্ট করে তুলুন",
                "is_healthy": True,
            },
            "model": "gemini-2.0-flash",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"বিশ্লেষণে সমস্যা হয়েছে: {str(e)}"
        }


# ── Quota Status Endpoint (Admin Panel এর জন্য) ──
@app.get("/api/quota-status")
def quota_status():
    """Admin Panel এ quota দেখানোর জন্য"""
    try:
        db = get_firestore_client()
        if db:
            doc = db.collection('ai_usage').document('fish_symptom_quota').get()
            if doc.exists:
                return {"success": True, "quota": doc.to_dict()}
        return {"success": True, "quota": {"total_calls": 0}}
    except Exception as e:
        return {"success": False, "error": str(e)}












if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)