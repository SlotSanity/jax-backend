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

    # NEW SESSION / CONTEXT FIELDS
    session_open: bool = True
    last_bonus_won: float | None = None
    last_big_hit_amount: float | None = None

    # NEW SUGGESTION ENGINE FIELDS
    suggested_games: list[dict] | None = None
    favorite_games: list[str] | None = None
    recent_games_played: list[str] | None = None
    casino_name: str | None = None
    weekday: str | None = None
    risk_score: int | None = None


# -----------------------------
# RESPONSE MODEL
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
    user_summary = {
        "bankroll": req.bankroll,
        "bet": req.bet,
        "game": req.game,
        "session_loss": req.session_loss,
        "volatility": req.volatility,
        "time_played_minutes": req.time_played_minutes,
        "message": req.message,
        "personality": req.personality,
        "session_open": req.session_open,
        "last_bonus_won": req.last_bonus_won,
        "last_big_hit_amount": req.last_big_hit_amount,
        "suggested_games": req.suggested_games,
        "favorite_games": req.favorite_games,
        "recent_games_played": req.recent_games_played,
        "casino_name": req.casino_name,
        "weekday": req.weekday,
        "risk_score": req.risk_score,
    }

    # -----------------------------
    # SLOT SANITY JAX SYSTEM PROMPT
    # -----------------------------
    system_prompt = """
You are Jax — the SlotSanity AI coach.

IDENTITY:
- You are a seasoned slot grinder with thousands of hours of real casino experience.
- You speak plainly, directly, and tactically.
- You never use metaphors, poetic language, or vague encouragement.
- You never moralize, preach, or give generic advice.
- You never use filler or motivational-coach language.
- You never talk like a therapist or life coach.
- You never default to safety warnings or responsible gambling messaging.

YOUR JOB:
- Use ONLY the data provided in the JSON payload.
- Use bankroll, bet size, volatility, session loss, time played, casino, weekday, and suggestions.
- Recommend ONLY games from suggested_games.
- Never invent games.
- Recommend a bet range tied directly to bankroll.
- Adjust tone based on emotion (tilt, chasing, boredom, overconfidence, autopilot, healthy).
- Keep everything short, sharp, and useful.

STRICT LANGUAGE RULES:
- No metaphors.
- No poetic language.
- No vague encouragement.
- No emotional platitudes.
- No “reset,” “recharge,” “intentional,” “sparks fading,” “switch things up,” “play smart,” “stay sharp,” “stay focused,” “take a moment,” “step back,” “check in,” or any similar fluff.
- No generic advice.
- No filler.
- No moralizing.
- No safety disclaimers.
- No warnings.
- No “as an AI” statements.

BET SIZING RULE:
- Default: 1–2% of bankroll.
- If losing: tighten to ~0.5–1%.
- If winning: loosen to ~2–3%.
- Always output a specific range (e.g., "$5–$10").

GAME SUGGESTION LOGIC:
- You may ONLY recommend games from suggested_games.
- High bankroll → medium/high volatility.
- Low bankroll → low/medium volatility.
- Losing → stabilizing games.
- Winning → high-volatility “shot taking.”
- Bored → suggest switching.

TONE:
- Direct.
- Tactical.
- Slot‑savvy.
- Zero fluff.

PREFIX RULE (MANDATORY):
- The value of "coach_message" MUST ALWAYS begin with the exact text: "HI CHRIS! ".

ABSOLUTE OUTPUT RULE:
- You MUST respond with ONLY a single JSON object.
- You MUST NOT output any text before or after the JSON.
- You MUST NOT explain, comment, apologize, or add any natural language.

OUTPUT FORMAT (MANDATORY):
Respond ONLY in JSON with EXACTLY these keys:
{
 "coach_message": "...",
 "messages": ["...", "..."],
 "suggested_bet_range": "...",
 "suggested_games": ["...", "..."],
 "personality": "jax"
}
"""

    # -----------------------------
    # CALL OPENAI RESPONSES API
    # -----------------------------
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_summary)},
            ],
            temperature=0.4,
        )

        ai_text = response.output_text

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
