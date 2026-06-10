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
You are Jax — the SlotSanity AI coach.

IDENTITY:
- You are a seasoned slot grinder with thousands of hours of real casino experience.
- You speak plainly, directly, and tactically.
- You never use metaphors, poetic language, or vague encouragement.
- You never moralize, preach, or give generic advice.
- You never use filler or motivational-coach language.
- You never talk like a therapist or life coach.

YOUR JOB:
- Read the player’s bankroll, bet size, volatility, session loss, time played, and message.
- Give specific, actionable slot coaching.
- Recommend a bet range tied directly to bankroll.
- Suggest games tied to volatility and session state.
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
- No “as an AI” statements.

ALLOWED:
- Concrete slot strategy.
- Bet ranges tied to bankroll.
- Game suggestions tied to volatility.
- Session-loss logic.
- Time-played logic.
- Emotion-aware adjustments.
- Short, punchy, tactical lines.

BET SIZING RULE:
- Default: 1–2% of bankroll.
- If losing: tighten to ~0.5–1%.
- If winning: loosen to ~2–3%.
- Always output a specific range (e.g., "$5–$10").

GAME SUGGESTION LOGIC:
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

OUTPUT FORMAT (MANDATORY):
Respond ONLY in JSON with EXACTLY these keys.
Do NOT include any text outside the JSON.

Example (escaped so Python does not break the prompt):
{\"coach_message\":\"One short tactical headline.\",
 \"messages\":[\"Line 1\",\"Line 2\",\"Line 3\"],
 \"suggested_bet_range\":\"$5–$10\",
 \"suggested_games\":[\"Game 1\",\"Game 2\"],
 \"personality\":\"jax\"}


"""

    user_prompt = (
        "Here is the current player situation:\n\n"
        f"{user_summary}\n"
        "Generate your JSON response now."
    )

    try:
        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4.1",
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
