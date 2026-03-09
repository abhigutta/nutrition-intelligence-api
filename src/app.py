"""
Nutrition Intelligence Scanner Lambda
Calls OpenAI GPT-5.2-mini, stores result in DynamoDB, returns the analysis.
OpenAI API key is fetched securely from AWS Secrets Manager and cached
in-memory for the lifetime of the Lambda container (warm invocations).
"""

import json
import os
import re
import base64
import boto3
from openai import OpenAI
from datetime import datetime, timezone

# ── AWS Clients ───────────────────────────────────────────
secrets_client = boto3.client("secretsmanager")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["DYNAMODB_TABLE"])
rate_limit_table = dynamodb.Table(os.environ["RATE_LIMIT_TABLE"])

# ── Rate limit config ─────────────────────────────────────
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "50"))

# ── In-memory secret cache (persists across warm invocations) ──
_openai_client_cache: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """
    Fetch the OpenAI API key from Secrets Manager on first call,
    then reuse the cached client for subsequent warm invocations.
    """
    global _openai_client_cache
    if _openai_client_cache is not None:
        return _openai_client_cache

    secret_arn = os.environ["OPENAI_SECRET_ARN"]
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret = json.loads(response["SecretString"])
    api_key = secret["OPENAI_API_KEY"]

    _openai_client_cache = OpenAI(api_key=api_key)
    return _openai_client_cache

# ── System prompt (v7.1) ────────────────────────────────
SYSTEM_PROMPT = r"""SYSTEM PROMPT — Advanced Nutrition Intelligence Scanner (API) v7.1
(Responses MUST be JSON only)

You are an advanced nutrition intelligence system specializing in nutrient density,
bioavailability, metabolic physiology, and food safety/contaminants.

Your job: Given userProfile + userData (scan text, ingredient list, or multiple foods)
+ oldContext, detect each distinct food/product and return a structured nutrition
analysis that is practical, safety-aware, and grounded in physiology. Give cooked and
uncooked if applicable and return as 2 different asks.

────────────────────────────────────────────────────────
EVIDENCE RULES
────────────────────────────────────────────────────────
Prefer strongest evidence:
1) RCTs
2) Controlled feeding trials
3) Meta-analyses of clinical trials
4) Human metabolic physiology studies
5) Mechanistic studies

Observational studies are correlation only; never present them as causation.
Do not invent facts. If a value is unknown, use null and note uncertainty.

────────────────────────────────────────────────────────
SCANNER MODE (MULTI-FOOD)
────────────────────────────────────────────────────────
User input may contain:
- multiple foods (e.g., "eggs, milk, spinach")
- packaged product ingredients
- mixed meals

You MUST:
1) parse and list each distinct detected item in scanAssumptions.detectedItems
2) create one foodsAnalyzed[] object per detected item
3) include scanAssumptions.uncertainties when parsing is ambiguous

────────────────────────────────────────────────────────
CORE NUTRITION APPROACH
────────────────────────────────────────────────────────
Prioritize:
- nutrient density per calorie
- bioavailability (net absorption/utilization)
- protein quality
- metabolic stability
- digestibility/tolerance
- safety (foodborne risk, contamination, excess risks)

Flag and discourage:
- ultra-processed foods
- refined starches/sugars (e.g., corn starch derivatives, refined flours)
- industrial seed oils and deep-fried processed foods

Use metabolic reasoning when relevant:
- glycolysis
- beta-oxidation
- Randle cycle (fuel competition)
- glycogen replenishment/depletion
- mTOR/protein synthesis

────────────────────────────────────────────────────────
PLANT DEFENSE CHEMICALS (HARSH SCORING HEURISTIC)
────────────────────────────────────────────────────────
For EACH food, classify:
defenseChemicalLoad = "none_low" | "moderate" | "high" | "unknown"
and list primaryCompounds (oxalates/phytates/lectins/tannins/etc.) when applicable.

Apply this strict scoring constitution:
Rule1 (Elite): high nutrient density + high bioavailability + none_low
  → overallFoodScore 90–100
Rule2 (Moderate defense chemicals): nutrient-rich but moderate inhibitors
  → overallFoodScore 60–75 (default ~70)
Rule3 (High defense chemicals): high inhibitors even if nutrients look high
  → overallFoodScore 40–50
Rule4 (Low nutrients + inhibitors): low nutrients AND moderate/high inhibitors
  → overallFoodScore 20–30
Rule5 (Ultra-processed): refined starch/sugar + industrial oils/additives
  → overallFoodScore 10–40

Mitigation (boiling, fermentation, soaking, calcium pairing, rotation, smaller servings)
may improve score WITHIN the band but must not break caps unless defenseChemicalLoad
is reclassified lower.

────────────────────────────────────────────────────────
CONTAMINANT / HEAVY METAL SCREEN (MANDATORY)
────────────────────────────────────────────────────────
For EACH food, assess and report meaningful risks:
- mercury risk (especially large predatory fish)
- arsenic risk (rice)
- heavy metals (some cocoa products)
- iodine excess/heavy metals (some seaweed)
- supplement contamination/adulteration risk

Use qualitative risk levels: "low" | "moderate" | "higher" | "unknown"
Provide practical risk reduction steps (rotation, sourcing, preparation).

────────────────────────────────────────────────────────
PROS & CONS (SCORING-SENSITIVE)
────────────────────────────────────────────────────────
Every food must include pros[] and cons[].

If overallFoodScore >= 90:
- 3–6 pros
- 0–1 minor cons (or none)
- do NOT nitpick trivial flaws

If 70–89:
- 3–5 pros
- 1–3 cons

If 40–69:
- balanced pros and cons (2–5 each)

If < 40:
- many serious cons (3–8)
- cons should be stronger and "scary" enough to steer away, BUT must remain factually
  true (e.g., "highly processed + easy to overeat + poor micronutrients + metabolic downside")
- include risk notes (e.g., oxidation, additives, high glycemic load) when applicable

Also include overallPros[] and overallCons[] for the full scan.

────────────────────────────────────────────────────────
SYMPTOMS (OPTIONAL)
────────────────────────────────────────────────────────
If the user mentions symptoms, provide possible nutrition-related hypotheses
(not diagnosis) and low-risk next steps. Include medicalRedFlags if applicable.

────────────────────────────────────────────────────────
RECIPES
────────────────────────────────────────────────────────
If the user asks for a recipe OR the scanned foods can naturally form a meal,
return 1–3 simple nutrient-focused recipes with ingredients, quantities, steps,
nutrient highlights, and safety notes.

────────────────────────────────────────────────────────
OUTPUT (STRICT JSON ONLY)
────────────────────────────────────────────────────────
Return ONLY valid JSON matching this schema (no markdown, no extra text):

{
  "summary": "",
  "scanAssumptions": {
    "detectedItems": [],
    "uncertainties": []
  },
  "overallPros": [],
  "overallCons": [],
  "foodsAnalyzed": [
    {
      "foodName": "",
      "category": "whole_food|packaged_food|supplement|mixed_meal|unknown",
      "servingAssumed": "",
      "pros": [],
      "cons": [],
      "macronutrients": {
        "protein_g": null,
        "fat_g": null,
        "carbs_g": null,
        "calories": null
      },
      "keyMicronutrients": [
        { "name": "", "note": "" }
      ],
      "bioavailabilityNotes": "",
      "digestibilityNotes": "",
      "defenseChemicalAssessment": {
        "defenseChemicalLoad": "none_low|moderate|high|unknown",
        "primaryCompounds": [],
        "ruleApplied": "Rule1|Rule2|Rule3|Rule4|Rule5",
        "scoreBandUsed": "90-100|60-75|40-50|20-30|10-40",
        "mitigationOptions": []
      },
      "contaminantRiskScreen": {
        "heavyMetals": [
          { "risk": "low|moderate|higher|unknown", "reason": "", "riskReduction": "" }
        ],
        "foodborneRisk": [
          { "risk": "low|moderate|higher|unknown", "reason": "", "riskReduction": "" }
        ],
        "processingRisks": [
          { "risk": "low|moderate|higher|unknown", "reason": "", "riskReduction": "" }
        ]
      },
      "metabolicEffects": {
        "glycolysisImpact": "",
        "fatOxidationImpact": "",
        "randleCycleNotes": "",
        "glycogenNotes": "",
        "proteinSynthesisNotes": ""
      },
      "whoShouldBeCareful": [],
      "scores": {
        "nutrientDensityScore": 0,
        "bioavailabilityScore": 0,
        "metabolicHealthScore": 0,
        "digestibilityScore": 0,
        "overallFoodScore": 0
      }
    }
  ],
  "foodInteractions": {
    "bestPairings": [],
    "avoidOrLimitPairings": [],
    "timingStrategy": ""
  },
  "symptomAnalysis": {
    "reportedSymptoms": [],
    "possibleNutritionLinks": [],
    "medicalRedFlags": []
  },
  "recipes": [
    {
      "name": "",
      "goalTag": "high-protein|metabolic-stable|post-workout|easy-digest|high-test-support",
      "ingredients": [
        { "item": "", "amount": "" }
      ],
      "steps": [],
      "nutrientHighlights": [],
      "bioavailabilityTips": [],
      "safetyNotes": []
    }
  ],
  "practicalRecommendations": [
    { "action": "", "reason": "" }
  ],
  "evidenceNotes": [
    {
      "claim": "",
      "evidenceTier": "RCT|FeedingTrial|MetaRCT|MechanisticHuman|Observational|Consensus",
      "notes": ""
    }
  ]
}"""


def _check_and_increment_rate_limit(device_id: str) -> tuple[bool, int, int]:
    """
    Atomically increment today's request count for a device.
    Uses DynamoDB conditional UpdateItem so the counter is race-condition safe.

    Returns:
        (allowed, current_count, limit)
        allowed=False means the device has hit the daily cap.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # TTL = end of today UTC (midnight of next day) in epoch seconds
    tomorrow_midnight = int(
        datetime(
            *[int(p) for p in today.split("-")],
            tzinfo=timezone.utc
        ).timestamp()
    ) + 86400  # +24 h

    try:
        response = rate_limit_table.update_item(
            Key={"deviceId": device_id, "date": today},
            UpdateExpression=(
                "SET #cnt = if_not_exists(#cnt, :zero) + :one, "
                "#ttl = if_not_exists(#ttl, :ttl)"
            ),
            ExpressionAttributeNames={"#cnt": "requestCount", "#ttl": "ttl"},
            ExpressionAttributeValues={
                ":zero": 0,
                ":one": 1,
                ":ttl": tomorrow_midnight,
            },
            ReturnValues="UPDATED_NEW",
        )
        current = int(response["Attributes"]["requestCount"])
        allowed = current <= RATE_LIMIT_MAX
        return allowed, current, RATE_LIMIT_MAX
    except Exception as e:
        # Fail open — don't block the request if DynamoDB has a transient error
        print(f"Rate limit check error (failing open): {e}")
        return True, -1, RATE_LIMIT_MAX


def _normalize_food_name(food_items_raw: str) -> str:
    """
    Create a canonical DynamoDB partition key from the raw food input.
    e.g. "  Spinach , EGGS,milk " → "eggs_milk_spinach"
    """
    items = [item.strip().lower() for item in food_items_raw.split(",") if item.strip()]
    items.sort()
    return "_".join(re.sub(r"[^a-z0-9]+", "-", i) for i in items)


def _call_openai(food_items: str, user_profile: dict | None = None,
                 old_context: dict | None = None) -> dict:
    """Call OpenAI GPT-5.2-mini with the nutrition system prompt."""
    user_message_parts = []
    if user_profile:
        user_message_parts.append(f"userProfile: {json.dumps(user_profile)}")
    user_message_parts.append(f"userData: {food_items}")
    if old_context:
        user_message_parts.append(f"oldContext: {json.dumps(old_context)}")

    user_content = "\n".join(user_message_parts)

    openai_client = _get_openai_client()
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content
    return json.loads(raw_text)


def _scan_image(image_base64: str) -> dict:
    """Send an image to OpenAI vision to identify food items."""
    openai_client = _get_openai_client()

    # Detect MIME type from base64 header or default to jpeg
    if image_base64.startswith("data:"):
        # Already has data URI prefix — use as-is
        image_url = image_base64
    else:
        image_url = f"data:image/jpeg;base64,{image_base64}"

    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a food identification assistant. Analyze the image and "
                    "identify all distinct food items visible. Return ONLY valid JSON "
                    "matching this schema (no markdown, no extra text):\n"
                    '{"foods": [{"name": "food name", "confidence": "high|medium|low"}], '
                    '"description": "brief description of what you see"}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify all food items in this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "low"},
                    },
                ],
            },
        ],
        max_completion_tokens=1000,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content
    return json.loads(raw_text)


def _lookup_in_dynamodb(food_key: str, version: str) -> dict | None:
    """Check DynamoDB for an existing analysis. Returns the item or None."""
    try:
        response = table.get_item(Key={"foodItem": food_key, "version": version})
        item = response.get("Item")
        if item and "analysis" in item:
            return {
                "foodItem": item["foodItem"],
                "version": item["version"],
                "analysis": json.loads(item["analysis"]),
                "createdAt": item.get("createdAt"),
                "source": "cache",
            }
    except Exception as e:
        print(f"DynamoDB lookup error: {e}")
    return None


def _store_in_dynamodb(food_key: str, version: str, analysis: dict,
                       raw_input: str) -> None:
    """Persist the analysis result in DynamoDB."""
    table.put_item(
        Item={
            "foodItem": food_key,
            "version": version,
            "rawInput": raw_input,
            "analysis": json.dumps(analysis),  # stored as JSON string
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": "gpt-5-mini",
        }
    )


def _build_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }


# ── Handler ──────────────────────────────────────────────
def lambda_handler(event, context):
    path = event.get("path", "")
    http_method = event.get("httpMethod", "")

    if path == "/scan" and http_method == "POST":
        return _handle_scan(event)
    elif path == "/analyze" and http_method == "POST":
        return _handle_analyze(event)
    else:
        return _build_response(404, {"error": f"Not found: {http_method} {path}"})


def _handle_scan(event):
    """
    POST /scan
    Body: {
      "image": "<base64-encoded image>",   # required
      "deviceId": "uuid"                   # required
    }
    Returns list of identified food items from the image.
    """
    try:
        body = json.loads(event.get("body", "{}") or "{}")
    except json.JSONDecodeError:
        return _build_response(400, {"error": "Invalid JSON body"})

    device_id = (
        body.get("deviceId")
        or (event.get("headers") or {}).get("x-device-id")
        or (event.get("headers") or {}).get("X-Device-Id")
    )
    if not device_id:
        return _build_response(400, {
            "error": "Missing deviceId",
            "detail": "Send deviceId in the request body or x-device-id header.",
        })

    allowed, current_count, limit = _check_and_increment_rate_limit(device_id)
    if not allowed:
        return _build_response(429, {
            "error": "Daily rate limit exceeded",
            "requestsUsed": current_count,
            "requestsLimit": limit,
        })

    image_base64 = body.get("image")
    if not image_base64:
        return _build_response(400, {
            "error": "Missing required field: image",
            "detail": "Send a base64-encoded image in the 'image' field.",
        })

    try:
        result = _scan_image(image_base64)
    except Exception as e:
        return _build_response(502, {
            "error": "Image scan failed",
            "detail": str(e),
        })

    return _build_response(200, {
        "foods": result.get("foods", []),
        "description": result.get("description", ""),
        "rateLimit": {
            "requestsUsed": current_count,
            "requestsLimit": limit,
            "requestsRemaining": max(0, limit - current_count),
            "resetsAt": "midnight UTC",
        },
    })


def _handle_analyze(event):
    """
    POST /analyze
    Body: {
      "foodItems": "spinach, eggs",        # required
      "deviceId": "uuid",                  # required
      "version": "v1",                     # optional, default "v1"
      "userProfile": { ... },              # optional
      "oldContext": { ... }                # optional
    }
    Checks DynamoDB cache first; calls OpenAI only on cache miss.
    """
    try:
        body = json.loads(event.get("body", "{}") or "{}")
    except json.JSONDecodeError:
        return _build_response(400, {"error": "Invalid JSON body"})

    device_id = (
        body.get("deviceId")
        or (event.get("headers") or {}).get("x-device-id")
        or (event.get("headers") or {}).get("X-Device-Id")
    )
    if not device_id:
        return _build_response(400, {
            "error": "Missing deviceId",
            "detail": "Send deviceId in the request body or x-device-id header.",
        })

    allowed, current_count, limit = _check_and_increment_rate_limit(device_id)
    if not allowed:
        return _build_response(429, {
            "error": "Daily rate limit exceeded",
            "detail": f"Device '{device_id}' has used {current_count}/{limit} requests today. Resets at midnight UTC.",
            "requestsUsed": current_count,
            "requestsLimit": limit,
        })

    food_items = body.get("foodItems")
    if not food_items:
        return _build_response(400, {
            "error": "Missing required field: foodItems",
            "usage": {
                "deviceId": "your-unique-device-id",
                "foodItems": "spinach, eggs",
                "version": "v1 (optional)",
                "userProfile": "{} (optional)",
                "oldContext": "{} (optional)",
            },
        })

    version = body.get("version", "v1")
    user_profile = body.get("userProfile")
    old_context = body.get("oldContext")
    food_key = _normalize_food_name(food_items)

    # ── Check DynamoDB cache first ────────────────────────
    cached = _lookup_in_dynamodb(food_key, version)
    if cached:
        cached["rateLimit"] = {
            "requestsUsed": current_count,
            "requestsLimit": limit,
            "requestsRemaining": max(0, limit - current_count),
            "resetsAt": "midnight UTC",
        }
        return _build_response(200, cached)

    # ── Cache miss — call OpenAI ──────────────────────────
    try:
        analysis = _call_openai(food_items, user_profile, old_context)
    except Exception as e:
        return _build_response(502, {
            "error": "OpenAI API call failed",
            "detail": str(e),
        })

    # ── Store in DynamoDB ────────────────────────────────
    try:
        _store_in_dynamodb(food_key, version, analysis, food_items)
    except Exception as e:
        return _build_response(500, {
            "error": "DynamoDB write failed",
            "detail": str(e),
        })

    return _build_response(200, {
        "foodItem": food_key,
        "version": version,
        "analysis": analysis,
        "source": "openai",
        "rateLimit": {
            "requestsUsed": current_count,
            "requestsLimit": limit,
            "requestsRemaining": max(0, limit - current_count),
            "resetsAt": "midnight UTC",
        },
    })
