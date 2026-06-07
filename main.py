import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import os

# Load environment variables
load_dotenv()

# Create FastAPI app once
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create OpenAI client once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return {"status": "ok"}

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "SlotSanity backend is running",
        "openai_key_loaded": os.getenv("OPENAI_API_KEY") is not None
    }

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Request model for AI Coach
class CoachRequest(BaseModel):
    bankroll: float
    session_loss: float
    volatility: str
    time_played_minutes: int

# AI Coach endpoint
@app.post("/ai/coach")
async def ai_coach(request: CoachRequest):
    prompt = f"""
You are Jax, the SlotSanity AI assistant.

Your personality:
- Confident, modern, and grounded
- Direct but never harsh
- Supportive without being hypey or preachy
- Focused on clarity, risk awareness, and smart decision-making

Your job:
Analyze the player's current gambling situation using the provided data:
- Bankroll: {request.bankroll}
- Session loss: {request.session_loss}
- Volatility: {request.volatility}
- Time played: {request.time_played_minutes} minutes

Based on this information, evaluate:
- The player’s current risk level
- Whether their behavior suggests tilt, fatigue, or overextension
- Whether continuing play is reasonable or risky

You must return ONLY valid JSON with the following structure:

{{
  "coach_message": "string — short, direct guidance from Jax",
  "risk_score": "number between 1 and 10",
  "recommended_action": "continue | pause | cash_out",
  "personality": "jax"
}}


Rules:
- Do not include markdown.
- Do not include labels or explanations outside the JSON.
- Do not add commentary before or after the JSON.
- The JSON must be valid and parseable.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
        return data
    except Exception as e:
        return {
            "error": "Invalid JSON returned by Jax",
            "raw_response": raw,
            "exception": str(e)
        }
