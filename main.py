"""
Telegram Bot → OpenRouter LLM → Todoist Task Creator

Features:
  - Multimodal input (text, image, or both)
  - Structured output via OpenRouter
  - Confirmation preview before creating tasks
  - Daily task reminders at 08:00 (configurable timezone)
  - Retries with exponential backoff
  - Rate limit awareness
  - Shared HTTP clients

Env vars required:
    TELEGRAM_BOT_TOKEN
    OPENROUTER_API_KEY
    TODOIST_API_KEY
    REGISTRATION_PASSWORD
Optional:
    REMINDER_TIMEZONE   – IANA timezone for daily reminders (default: UTC)
    OPENROUTER_MODEL    – LLM model to use via OpenRouter
"""

import os
import json
import base64
import logging
import asyncio
import uuid
import hmac
import datetime
from zoneinfo import ZoneInfo

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
REGISTRATION_PASSWORD = os.environ["REGISTRATION_PASSWORD"]

TODOIST_API = "https://api.todoist.com/api/v1/tasks"
TODOIST_FILTER_API = "https://api.todoist.com/api/v1/tasks/filter"
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"

REMINDER_TIMEZONE = os.environ.get("REMINDER_TIMEZONE", "UTC")

# Model must support structured outputs via OpenRouter.
# Known good: openai/gpt-4.1-mini, openai/gpt-4.1-nano, openai/gpt-4o-mini
# Gemini models have issues with nullable schema types through OpenRouter's compatibility layer.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

# Pending confirmations: { batch_id: { "tasks": [...], "chat_id": int, "user_id": int } }
pending: dict[str, dict] = {}

# Registered users: { (chat_id, user_id): {"username": str} }
registered: dict[tuple[int, int], dict] = {}

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
                        "type": "string",
                        "description": "Natural language due date, e.g. 'tomorrow at 3pm', 'every monday', 'Jan 15'. Empty string if none.",
                    },
                    "duration_amount": {
                        "type": "integer",
                        "description": "Estimated duration amount. 0 if not specified.",
                    },
                    "duration_unit": {
                        "type": "string",
                        "enum": ["minute", "day"],
                        "description": "Duration unit. 'minute' by default.",
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
                "required": ["content", "description", "due_string", "duration_amount", "duration_unit", "labels", "priority"],
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
- due_string: natural language date/time string Todoist can parse (e.g. "tomorrow", "every friday at 5pm", "Jan 20 at noon", "Feb 15 2025 at 2pm"). Use full dates when visible in images. Empty string if no date mentioned.
- duration_amount: integer, how long the task takes. 0 if not specified.
- duration_unit: "minute" or "day". Default "minute".
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
        "model": OPENROUTER_MODEL,
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

    try:
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
    except httpx.HTTPStatusError as e:
        body = e.response.text
        log.error(f"OpenRouter HTTP {e.response.status_code}:\n{body}")
        raise ValueError(f"OpenRouter {e.response.status_code}. Check logs.")

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

    if task.get("duration_amount", 0) > 0:
        body["duration"] = {
            "amount": task["duration_amount"],
            "unit": task.get("duration_unit", "minute"),
        }

    if task.get("labels"):
        body["labels"] = task["labels"]

    try:
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
    except httpx.HTTPStatusError as e:
        log.error(f"Todoist HTTP {e.response.status_code}:\n{e.response.text}")
        raise

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
        if t.get("duration_amount", 0) > 0:
            details.append(f"⏱ {t['duration_amount']}{t.get('duration_unit', 'minute')[0]}")
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


# ─── Daily reminders ─────────────────────────────────────────────────────────


async def fetch_today_tasks() -> list[dict]:
    """Fetch tasks due today (and overdue) from Todoist."""
    resp = await request_with_retry(
        todoist_client,
        "GET",
        TODOIST_FILTER_API,
        headers={"Authorization": f"Bearer {TODOIST_API_KEY}"},
        params={"query": "today | overdue", "limit": 200},
    )
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get("results", [])


def format_daily_task_list(tasks: list[dict]) -> str:
    """Format Todoist tasks into a readable daily summary message."""
    if not tasks:
        return "☀️ Your Day\n\nNo tasks for today. Enjoy your free time!"

    lines = ["☀️ Your tasks for today:", ""]

    for i, task in enumerate(tasks, 1):
        pri = PRIORITY_LABELS.get(task.get("priority", 1), "")
        line = f"{i}. {pri} {task['content']}".strip()

        details = []
        due = task.get("due")
        if due:
            if due.get("datetime"):
                try:
                    dt = datetime.datetime.fromisoformat(due["datetime"])
                    details.append(f"🕐 {dt.strftime('%H:%M')}")
                except ValueError:
                    details.append(f"🕐 {due['datetime']}")
            elif due.get("string"):
                details.append(f"📅 {due['string']}")

        if task.get("labels"):
            details.append(f"🏷 {', '.join(task['labels'])}")
        if task.get("description"):
            desc = task["description"]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            details.append(f"📝 {desc}")

        if details:
            line += "\n   " + " · ".join(details)

        lines.append(line)

    lines.append("")
    lines.append(f"📋 {len(tasks)} task(s) total")
    return "\n".join(lines)


async def send_daily_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Job callback: send daily task list to all registered chats."""
    if not registered:
        log.info("No registered users. Skipping daily reminder.")
        return

    try:
        tasks = await fetch_today_tasks()
    except Exception:
        log.exception("Failed to fetch today's tasks for daily reminder")
        return

    message = format_daily_task_list(tasks)

    # Deduplicate by chat_id so each chat gets one message
    sent_chats: set[int] = set()
    for chat_id, _user_id in registered:
        if chat_id in sent_chats:
            continue
        try:
            await context.bot.send_message(chat_id, message)
            sent_chats.add(chat_id)
            log.info(f"Daily reminder sent to chat {chat_id}")
        except Exception:
            log.exception(f"Failed to send daily reminder to chat {chat_id}")


async def send_task_list_to_chat(bot, chat_id: int):
    """Fetch today's tasks and send the formatted list to a single chat."""
    try:
        tasks = await fetch_today_tasks()
    except Exception:
        log.exception("Failed to fetch today's tasks")
        return

    message = format_daily_task_list(tasks)
    try:
        await bot.send_message(chat_id, message)
    except Exception:
        log.exception(f"Failed to send task list to chat {chat_id}")


# ─── Registration ────────────────────────────────────────────────────────────


def is_registered(chat_id: int, user_id: int) -> bool:
    return (chat_id, user_id) in registered


async def register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /register <password>. Stores username + chat_id for the sender."""
    msg = update.message
    user = msg.from_user
    chat_id = msg.chat_id
    user_id = user.id

    if is_registered(chat_id, user_id):
        await msg.reply_text("You are already registered in this chat.")
        return

    if not context.args:
        await msg.reply_text("Usage: /register <password>")
        return

    provided = " ".join(context.args)

    # Best-effort delete of the message to avoid exposing the password in chat
    try:
        await msg.delete()
    except Exception:
        pass

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(provided, REGISTRATION_PASSWORD):
        await context.bot.send_message(chat_id, "Incorrect password.")
        return

    username = user.username or user.full_name or str(user_id)
    registered[(chat_id, user_id)] = {"username": username}
    log.info(f"Registered user {username!r} (id={user_id}) in chat {chat_id}")
    await context.bot.send_message(chat_id, f"Registered successfully. Welcome, {username}!")

    # Send today's tasks right away so the new user sees their day
    await send_task_list_to_chat(context.bot, chat_id)


# ─── Handlers ────────────────────────────────────────────────────────────────


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parse input → show preview → wait for confirmation."""
    msg = update.message

    if not is_registered(msg.chat_id, msg.from_user.id):
        await msg.reply_text("You must register first. Use /register <password>.")
        return

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
    pending[batch_id] = {"tasks": tasks, "chat_id": msg.chat_id, "user_id": msg.from_user.id}

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
    batch = pending.get(batch_id)

    if not batch:
        await query.edit_message_text("⚠️ This batch expired or was already processed.")
        return

    # Only the user who created the batch may act on it
    if query.from_user.id != batch["user_id"]:
        await query.answer("This confirmation is not for you.", show_alert=True)
        return

    pending.pop(batch_id, None)

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
        "Welcome! First, register with:\n"
        "  /register <password>\n\n"
        "Once registered, send me tasks as text, a screenshot, or both.\n"
        "I'll parse them, show a preview, and wait for your confirmation.\n\n"
        "Examples:\n"
        "• Buy groceries tomorrow\n"
        "• Write report by Friday, ~2 hours, high priority\n"
        "• 📸 Screenshot of a calendar, event flyer, or syllabus"
    )


# ─── Lifecycle ───────────────────────────────────────────────────────────────


async def post_init(app):
    """Create shared HTTP clients on startup, schedule daily reminders."""
    global openrouter_client, todoist_client
    openrouter_client = httpx.AsyncClient(timeout=45)
    todoist_client = httpx.AsyncClient(timeout=15)
    log.info("HTTP clients initialized.")

    # Schedule daily reminder at 08:00
    tz = ZoneInfo(REMINDER_TIMEZONE)
    app.job_queue.run_daily(
        send_daily_reminder,
        time=datetime.time(hour=8, minute=0, second=0, tzinfo=tz),
        name="daily_task_reminder",
    )
    log.info(f"Daily reminder scheduled at 08:00 {REMINDER_TIMEZONE}.")

    # Send task list to any already-registered users at boot
    # (Registration is in-memory so this is empty on a fresh start,
    #  but will fire if persistence is added later.)
    if registered:
        sent_chats: set[int] = set()
        for chat_id, _user_id in dict(registered):
            if chat_id not in sent_chats:
                await send_task_list_to_chat(app.bot, chat_id)
                sent_chats.add(chat_id)


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
    app.add_handler(CommandHandler("register", register))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    log.info("Bot running.")
    app.run_polling()


if __name__ == "__main__":
    main()