from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

# ---------------------------------------------------------
# APP + CORS
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Flutter device → Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------------
# REQUEST MODEL — matches Flutter EXACTLY
# ---------------------------------------------------------
class CoachingRequest(BaseModel):
    bankroll: float
    bet: float
    game: str
    session_loss: float
    volatility: float
    time_played_minutes: int

# ---------------------------------------------------------
# AI COACH ENDPOINT
# ---------------------------------------------------------
@app.post("/ai/coach")
async def coach(request: CoachingRequest):
    data = request.dict()

    bankroll = data["bankroll"]
    bet = data["bet"]
    game = data["game"]
    session_loss = data["session_loss"]
    volatility = data["volatility"]
    time_played_minutes = data["time_played_minutes"]

    # Build the coaching prompt
    prompt = f"""
    You are Jax, a gambling coach.

    Bankroll: {bankroll}
    Bet size: {bet}
    Game: {game}
    Session loss: {session_loss}
    Volatility: {volatility}
    Time played: {time_played_minutes} minutes

    Provide:
    - A coaching message
    - A risk score (0–100)
    - A recommended action
    """

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_text = response.choices[0].message["content"]

    # Return structured JSON for Flutter
    return {
        "coach_message": ai_text,
        "risk_score": 42,  # placeholder until you compute real values
        "recommended_action": "Take a break"
    }
