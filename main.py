import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS so your Flutter app can call this from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class CoachRequest(BaseModel):
    bankroll: float
    bet: float
    game: str
    session_loss: float
    volatility: float
    time_played_minutes: int
    message: str = ""
    personality: str = "default"

# -----------------------------
# RESPONSE MODEL
# -----------------------------
class CoachResponse(BaseModel):
    coach_message: str
    risk_score: float
    recommended_action: str
    personality: str = "default"

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Jax backend is running"}

# -----------------------------
# MAIN JAX ENDPOINT
# -----------------------------
@app.post("/ai/coach", response_model=CoachResponse)
async def coach(req: CoachRequest):

    # Build user summary for the AI
    user_summary = (
        f"Bankroll: {req.bankroll}\n"
        f"Bet size: {req.bet}\n"
        f"Game: {req.game}\n"
        f"Session loss: {req.session_loss}\n"
        f"Volatility: {req.volatility}\n"
        f"Time played (minutes): {req.time_played_minutes}\n"
        f"Personality: {req.personality}\n"
        f"User message: {req.message}\n"
    )

    # System prompt for Jax
    system_prompt = """
You are Jax, an AI gambling coach inside the SlotSanity app.
Your job is to help players make safer, smarter decisions.

You MUST respond ONLY in JSON with EXACTLY these keys:
{
  "coach_message": string,
  "risk_score": number between 0 and 1,
  "recommended_action": string,
  "personality": string
}

Do not include any extra text outside the JSON.
"""

    user_prompt = (
        "Here is the current player situation:\n\n"
        f"{user_summary}\n"
        "Generate your JSON response now."
    )

    try:
        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        ai_text = response.choices[0].message.content

        # Parse JSON returned by the model
        data = json.loads(ai_text)

        return CoachResponse(
            coach_message=data.get("coach_message", "No message generated."),
            risk_score=float(data.get("risk_score", 0.5)),
            recommended_action=data.get("recommended_action", "Take a break."),
            personality=data.get("personality", req.personality),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
