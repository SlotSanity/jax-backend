import os
import json
from fastapi import FastAPI
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


class CoachRequest(BaseModel):
    bankroll: float
    bet: float
    game: str
    session_loss: float
    volatility: float
    time_played_minutes: int


class CoachResponse(BaseModel):
    coach_message: str
    risk_score: float
    recommended_action: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "Jax backend is running"}


@app.post("/ai/coach", response_model=CoachResponse)
async def coach(req: CoachRequest):
    """
    Main Jax coaching endpoint used by your Flutter app.
    """

    user_summary = (
        f"Bankroll: {req.bankroll}\n"
        f"Bet size: {req.bet}\n"
        f"Game: {req.game}\n"
        f"Session loss: {req.session_loss}\n"
        f"Volatility: {req.volatility}\n"
        f"Time played (minutes): {req.time_played_minutes}\n"
    )

    system_prompt = (
        "You are Jax, a friendly but firm gambling coach. "
        "You help slot players make safer, smarter decisions. "
        "Always respond in **JSON** with exactly these keys:\n"
        '{\n'
        '  "coach_message": string,\n'
        '  "risk_score": number between 0 and 1,\n'
        '  "recommended_action": string\n'
        '}\n'
        "Do not include any extra keys or text outside the JSON."
    )

    user_prompt = (
        "Here is the current player situation:\n\n"
        f"{user_summary}\n"
        "Based on this, generate your JSON response."
    )

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    # ✅ FIXED: access content as an attribute, not like a dict
    ai_text = response.choices[0].message.content

    # Parse JSON returned by the model
    try:
        data = json.loads(ai_text)
    except json.JSONDecodeError:
        # Fallback if the model misbehaves
        return CoachResponse(
            coach_message="I had trouble understanding the AI response. Let's take a short break and try again.",
            risk_score=0.7,
            recommended_action="Take a break and reassess your bankroll.",
        )

    # Ensure all keys exist with safe defaults
    coach_message = data.get(
        "coach_message",
        "I couldn't generate a detailed message, but it's a good idea to play within your limits.",
    )
    risk_score = float(data.get("risk_score", 0.5))
    recommended_action = data.get(
        "recommended_action",
        "Consider pausing for a bit and reviewing your results.",
    )

    return CoachResponse(
        coach_message=coach_message,
        risk_score=risk_score,
        recommended_action=recommended_action,
    )
