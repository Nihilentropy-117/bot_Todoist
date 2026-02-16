"""
Telegram Bot → OpenRouter LLM → Todoist Task Creator

Features:
  - Multimodal input (text, image, or both)
  - Structured output via OpenRouter
  - Confirmation preview before creating tasks
  - Retries with exponential backoff
  - Rate limit awareness
  - Shared HTTP clients

Env vars required:
    TELEGRAM_BOT_TOKEN
    OPENROUTER_API_KEY
    TODOIST_API_KEY
"""

import os
import json
import base64
import logging
import asyncio
import uuid

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
TODOIST_API_KEY = os.environ["TODOIST_API_KEY"]

TODOIST_API = "https://api.todoist.com/rest/v2/tasks"
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"

# Pending confirmations: { batch_id: { "tasks": [...], "chat_id": int } }
pending: dict[str, dict] = {}

# Shared HTTP clients (created at startup, closed at shutdown)
openrouter_client: httpx.AsyncClient | None = None
todoist_client: httpx.AsyncClient | None = None

PRIORITY_LABELS = {1: "", 2: "🟡", 3: "🟠", 4: "🔴"}

# ─── Retry logic ────────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 8]  # seconds


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    """HTTP request with retries, backoff, and rate-limit awareness."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.request(method, url, **kwargs)

            # Rate limited — honor Retry-After or use backoff
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", RETRY_BACKOFF[attempt]))
                log.warning(f"Rate limited. Waiting {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
                continue

            # Server error — retry with backoff
            if resp.status_code >= 500:
                log.warning(f"Server {resp.status_code}. Retrying in {RETRY_BACKOFF[attempt]}s")
                await asyncio.sleep(RETRY_BACKOFF[attempt])
                continue

            resp.raise_for_status()
            return resp

        except httpx.TimeoutException as e:
            last_exc = e
            log.warning(f"Timeout (attempt {attempt + 1}). Retrying in {RETRY_BACKOFF[attempt]}s")
            await asyncio.sleep(RETRY_BACKOFF[attempt])

        except httpx.HTTPStatusError:
            raise  # Don't retry 4xx client errors (429 handled above)

        except httpx.RequestError as e:
            last_exc = e
            log.warning(f"Request error: {e} (attempt {attempt + 1})")
            await asyncio.sleep(RETRY_BACKOFF[attempt])

    raise last_exc or RuntimeError(f"Failed after {MAX_RETRIES} retries")


# ─── Schema ─────────────────────────────────────────────────────────────────

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Main text/title of the task. Supports markdown and hyperlinks.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Extended notes for the task. Empty string if none.",
                    },
                    "due_string": {
                        "type": ["string", "null"],
                        "description": "Natural language due date, e.g. 'tomorrow at 3pm', 'every monday', 'Jan 15'. Null if none.",
                    },
                    "duration": {
                        "type": ["object", "null"],
                        "properties": {
                            "amount": {"type": "integer"},
                            "unit": {"type": "string", "enum": ["minute", "day"]},
                        },
                        "required": ["amount", "unit"],
                        "description": "Estimated duration. Null if not specified.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Label names. Empty array if none.",
                    },
                    "priority": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4],
                        "description": "Priority 1 (normal) to 4 (urgent). Default 1.",
                    },
                },
                "required": ["content", "description", "due_string", "duration", "labels", "priority"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a task extraction engine. The user sends you freeform text, an image (screenshot of a calendar, event flyer, syllabus, schedule, to-do list, etc.), or both.

Your job: parse ALL visible tasks, events, deadlines, and action items into structured task objects.

Rules:
- content: concise task title. Use markdown if the user implies links.
- description: any extra context or notes. For events extracted from images, include relevant details like location, time, speaker, etc. Empty string if none.
- due_string: natural language date/time string Todoist can parse (e.g. "tomorrow", "every friday at 5pm", "Jan 20 at noon", "Feb 15 2025 at 2pm"). Use full dates when visible in images. Null if no date mentioned.
- duration: if duration or time range is visible (e.g. "2:00 PM - 3:30 PM" → 90 minutes), extract it. Null otherwise.
- labels: extract any tags, categories, or labels mentioned or implied by context (e.g. "work", "school", "personal"). Empty array if none.
- priority: 4 = urgent/critical, 3 = high, 2 = medium, 1 = normal (default). Infer from language like "urgent", "important", "ASAP", "low priority", etc.

For images: read ALL text carefully. Extract every distinct event, deadline, or task visible. Preserve exact dates and times as shown.

Do not invent information. Only extract what's stated or strongly implied."""


# ─── LLM ────────────────────────────────────────────────────────────────────


async def parse_tasks_via_llm(text: str | None = None, image_bytes: bytes | None = None) -> list[dict]:
    """Send text and/or image to OpenRouter, get structured task list back."""
    user_content = []

    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    prompt = text or "Extract all tasks, events, and deadlines from this image."
    user_content.append({"type": "text", "text": prompt})

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "task_list",
                "strict": True,
                "schema": TASK_SCHEMA,
            },
        },
        "temperature": 0.1,
    }

    resp = await request_with_retry(
        openrouter_client,
        "POST",
        OPENROUTER_API,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

    data = resp.json()
    raw_content = data["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        log.error(f"LLM returned invalid JSON:\n{raw_content}")
        raise ValueError("LLM returned unparseable response. Check logs.")

    return parsed["tasks"]


# ─── Todoist ─────────────────────────────────────────────────────────────────


async def create_todoist_task(task: dict) -> dict:
    """Create a single task in Todoist via REST API v2."""
    body = {
        "content": task["content"],
        "priority": task.get("priority", 1),
    }

    if task.get("description"):
        body["description"] = task["description"]

    if task.get("due_string"):
        body["due_string"] = task["due_string"]

    if task.get("duration"):
        body["duration"] = task["duration"]["amount"]
        body["duration_unit"] = task["duration"]["unit"]

    if task.get("labels"):
        body["labels"] = task["labels"]

    resp = await request_with_retry(
        todoist_client,
        "POST",
        TODOIST_API,
        headers={
            "Authorization": f"Bearer {TODOIST_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
    )

    return resp.json()


# ─── Preview formatting ─────────────────────────────────────────────────────


def format_preview(tasks: list[dict]) -> str:
    """Build a human-readable preview of parsed tasks."""
    lines = []
    for i, t in enumerate(tasks, 1):
        pri = PRIORITY_LABELS.get(t.get("priority", 1), "")
        line = f"{i}. {pri} {t['content']}".strip()

        details = []
        if t.get("due_string"):
            details.append(f"📅 {t['due_string']}")
        if t.get("duration"):
            details.append(f"⏱ {t['duration']['amount']}{t['duration']['unit'][0]}")
        if t.get("labels"):
            details.append(f"🏷 {', '.join(t['labels'])}")
        if t.get("description"):
            desc = t["description"]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            details.append(f"📝 {desc}")

        if details:
            line += "\n   " + " · ".join(details)

        lines.append(line)

    return "\n\n".join(lines)


# ─── Handlers ────────────────────────────────────────────────────────────────


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parse input → show preview → wait for confirmation."""
    msg = update.message
    text = msg.text or msg.caption or None
    image_bytes = None

    if msg.photo:
        photo = msg.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

    if not text and not image_bytes:
        return

    status = await msg.reply_text("⏳ Parsing...")

    try:
        tasks = await parse_tasks_via_llm(text=text, image_bytes=image_bytes)
    except Exception as e:
        log.exception("LLM parse failed")
        await status.edit_text(f"❌ Parse error: {e}")
        return

    if not tasks:
        await status.edit_text("No tasks found in your message.")
        return

    # Store pending batch and show preview with inline buttons
    batch_id = uuid.uuid4().hex[:12]
    pending[batch_id] = {"tasks": tasks, "chat_id": msg.chat_id}

    preview = format_preview(tasks)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Create all", callback_data=f"confirm:{batch_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"cancel:{batch_id}"),
        ]
    ])

    await status.edit_text(
        f"Found {len(tasks)} task(s):\n\n{preview}",
        reply_markup=keyboard,
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process confirm/cancel button presses."""
    query = update.callback_query
    await query.answer()

    action, batch_id = query.data.split(":", 1)
    batch = pending.pop(batch_id, None)

    if not batch:
        await query.edit_message_text("⚠️ This batch expired or was already processed.")
        return

    if action == "cancel":
        await query.edit_message_text("🚫 Cancelled. No tasks created.")
        return

    # Confirm → create all tasks
    tasks = batch["tasks"]
    await query.edit_message_text(f"⏳ Creating {len(tasks)} task(s)...")

    results = []
    for task in tasks:
        try:
            created = await create_todoist_task(task)
            line = f"✅ {created['content']}"
            if created.get("due"):
                line += f" — {created['due']['string']}"
            line += f"\n   {created['url']}"
            results.append(line)
        except Exception as e:
            log.exception(f"Todoist create failed for: {task['content']}")
            results.append(f"❌ {task['content']}: {e}")

    header = "Created the following task(s):\n\n"
    await query.edit_message_text(header + "\n\n".join(results))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me tasks as text, a screenshot, or both.\n"
        "I'll parse them, show a preview, and wait for your confirmation.\n\n"
        "Examples:\n"
        "• Buy groceries tomorrow\n"
        "• Write report by Friday, ~2 hours, high priority\n"
        "• 📸 Screenshot of a calendar, event flyer, or syllabus"
    )


# ─── Lifecycle ───────────────────────────────────────────────────────────────


async def post_init(app):
    """Create shared HTTP clients on startup."""
    global openrouter_client, todoist_client
    openrouter_client = httpx.AsyncClient(timeout=45)
    todoist_client = httpx.AsyncClient(timeout=15)
    log.info("HTTP clients initialized.")


async def post_shutdown(app):
    """Close shared HTTP clients on shutdown."""
    if openrouter_client:
        await openrouter_client.aclose()
    if todoist_client:
        await todoist_client.aclose()
    log.info("HTTP clients closed.")


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    log.info("Bot running.")
    app.run_polling()


if __name__ == "__main__":
    main()
