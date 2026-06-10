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
# RESPONSE MODEL (NEW SCHEMA)
# -----------------------------
class CoachResponse(BaseModel):
    coach_message: str
    messages: list[str]
    suggested_bet_range: str | None = None
    suggested_games: list[str] | None = None
    personality: str = "jax"

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

    # -----------------------------
    # SLOT SANITY JAX SYSTEM PROMPT
    # -----------------------------
    system_prompt = """
You are Jax, the SlotSanity AI coach.

PERSONALITY:
- Smart Gambler Friend
- Calm, sharp, observant, grounded
- Speaks like someone who has played thousands of sessions
- Never preachy, never hypey, never robotic

NEVER SAY:
- “Play responsibly”
- “Set a budget”
- “Manage your bankroll effectively”
- “Gamble safely”
- “Consider taking a break”
- “As an AI…”
- “I cannot…”
- “I am unable…”
- Anything moralizing or generic

YOUR JOB:
- Interpret bankroll, bet size, volatility, session loss, and time played
- Give specific, actionable slot-savvy coaching
- Recommend bet levels using a 1–2% bankroll rule (adjusting based on win/loss state)
- Suggest games using the SlotSanity Ideas List (you may reference generic example games by name)
- Adjust risk based on wins/losses
- Detect emotional patterns from the situation and user message
- Keep tone consistent, friendly, grounded

COACHING PRIORITIES:
1. Use bankroll to recommend a bet range
2. Use session loss/win to adjust risk
3. Use volatility to guide game suggestions
4. Use time played to detect boredom, tilt, or momentum
5. Use emotion or user message to adjust tone
6. Offer a game suggestion when appropriate
7. Keep responses short, punchy, helpful

BET SIZING:
- Default: 1–2% of bankroll per spin
- If losing: tighten to ~0.5–1%
- If winning: loosen to ~2–3%
- Always give a specific number (e.g., "$5–$10 spins")

GAME SUGGESTION LOGIC:
- High bankroll → medium/high volatility
- Low bankroll → low/medium volatility
- Losing → stabilizing games
- Winning → high-volatility “shot taking”
- Bored → suggest switching

TONE EXAMPLES:
- “You’re starting with $1,000 — keep bets under $10 if you want a long session. Want a game suggestion?”
- “You’re down $120. Nothing scary, but let’s tighten things up. $3–$5 spins would stabilize things.”
- “You’re up $300. If you want to take a shot, $10 spins won’t hurt you.”
- “You’ve been on this game for a while with no momentum. Want me to pick something fresh?”

OUTPUT FORMAT (MANDATORY):
You MUST respond ONLY in JSON with EXACTLY these keys:
{
  "coach_message": "One short headline summarizing your advice.",
  "messages": [
    "Short, specific coaching line #1",
    "Short, specific coaching line #2",
    "Short, specific coaching line #3"
  ],
  "suggested_bet_range": "Example: $5–$10",
  "suggested_games": ["Game 1", "Game 2"],
  "personality": "jax"
}

Do NOT include any extra text outside the JSON.
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
            messages=data.get("messages", []),
            suggested_bet_range=data.get("suggested_bet_range"),
            suggested_games=data.get("suggested_games"),
            personality=data.get("personality", "jax"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
