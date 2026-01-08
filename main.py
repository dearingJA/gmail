import os.path
import json
import re

from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import time
from datetime import datetime, timedelta
import pytz

import base64
from email.message import EmailMessage
from google import genai
from google.genai import types

from dev_secrets import GEMINI_API_KEY

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.events"
]
GMAIL_OAUTH_CLIENT_FILE = "gmail_oauth_client.json"
GMAIL_OAUTH_TOKEN_FILE = "gmail_oauth_token.json"

MODEL = "gemini-2.5-flash-lite"


# Auth / Credentials

def load_credentials():
    if os.path.exists(GMAIL_OAUTH_TOKEN_FILE):
        return Credentials.from_authorized_user_file(
            GMAIL_OAUTH_TOKEN_FILE, SCOPES)
    return None


def save_credentials(creds):
    with open(GMAIL_OAUTH_TOKEN_FILE, "w") as token:
        token.write(creds.to_json())


def authorize_new_credentials():
    flow = InstalledAppFlow.from_client_secrets_file(
        GMAIL_OAUTH_CLIENT_FILE, SCOPES
    )
    return flow.run_local_server(port=0)


def get_valid_credentials():
    creds = load_credentials()

    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise RefreshError("No valid refresh token")
        except RefreshError:
            creds = authorize_new_credentials()

        save_credentials(creds)

    return creds


# Gmail API

def build_gmail_service(creds):
    return build("gmail", "v1", credentials=creds)


def list_sent_message_ids(service, max_results=20):
    results = service.users().messages().list(
        userId="me",
        labelIds=["SENT"],
        maxResults=max_results
    ).execute()

    return results.get("messages", [])


def fetch_full_message(service, message_id):
    return service.users().messages().get(
        userId="me",
        id=message_id,
        format="full"
    ).execute()


def extract_body(payload):
    if payload.get("body", {}).get("data"):
        return decode_base64(payload["body"]["data"])

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            if part.get("body", {}).get("data"):
                return decode_base64(part["body"]["data"])

    for part in payload.get("parts", []):
        text = extract_body(part)
        if text:
            return text

    return ""


def has_attachment(payload):
    if payload.get("filename"):
        return True

    for part in payload.get("parts", []):
        if has_attachment(part):
            return True

    return False


def extract_sent_message(service, message_id):
    msg = fetch_full_message(service, message_id)
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    record = {
        "message_id": msg["id"],
        "thread_id": msg["threadId"],
        "internal_date": int(msg["internalDate"]),
        "subject": get_header(headers, "Subject"),
        "from": get_header(headers, "From"),
        "to": split_addresses(get_header(headers, "To")),
        "cc": split_addresses(get_header(headers, "Cc")),
        "bcc": split_addresses(get_header(headers, "Bcc")),
        "body_text": extract_body(payload),
        "has_attachment": has_attachment(payload),
        "label_ids": msg.get("labelIds", []),
    }

    record["recipient_count"] = len(record["to"]) + len(record["cc"])
    record["cc_count"] = len(record["cc"])
    record["sent_datetime"] = datetime.fromtimestamp(
        record["internal_date"] / 1000
    )

    return record


def build_email_prompt(record):
    return f"""
Analyze the following sent email and decide if a follow-up is required.

Email metadata:
- Subject: {record.get("subject")}
- From: {record.get("from")}
- To: {", ".join(record.get("to", []))}
- CC: {", ".join(record.get("cc", []))}
- Sent at: {record.get("sent_datetime")}
- Recipient count: {record.get("recipient_count")}
- Has attachment: {record.get("has_attachment")}

Email body:
\"\"\"
{record.get("body_text")}
\"\"\"
""".strip()


# Helpers


def get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def split_addresses(value):
    if not value:
        return []
    return [v.strip() for v in value.split(",")]


def decode_base64(data):
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def extract_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned.strip())


# Gemini

def build_followup_draft_system_instruction():
    return """
    You are an email assistant.

Your task:
Write a polite, professional follow-up email draft based on a previously sent email.

Rules:
- Be concise
- Be courteous and non-pushy
- Reference the prior email context
- Do NOT invent new commitments or deadlines
- Output ONLY the email body text (no subject, no JSON)
""".strip()


def build_followup_draft_prompt(record):
    return f"""
Original Subject: {record['subject']}
Recipient(s): {', '.join(record['to'])}
Original Email Body:
{record['body_text']}
"""


def generate_followup_draft(client, record):
    response = client.models.generate_content(
        model=MODEL,
        contents=build_followup_draft_prompt(record),
        config=types.GenerateContentConfig(
            system_instruction=build_followup_draft_system_instruction(),
            temperature=1.0
        )
    )
    return response.text.strip()


def build_followup_system_instruction():
    return """
You are an email assistant.

Your job:
Determine whether a sent email requires a follow-up.

Definitions:
- Follow-up REQUIRED:
  - You requested an action, decision, approval, document, meeting, or response
  - You proposed next steps or a deadline
  - You are waiting on the recipient to do something

- Follow-up NOT required:
  - Informational emails
  - Thank-you notes
  - FYI messages
  - Simple questions with no urgency or action dependency

Return STRICT JSON:
{
  "follow_up_required": true | false,
  "reason": "short explanation",
  "confidence": 0.0 - 1.0
}
""".strip()


def create_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def build_followup_config():
    return types.GenerateContentConfig(
        system_instruction=build_followup_system_instruction(),
        temperature=1.0
    )


def evaluate_followup(client, email_text):
    config = build_followup_config()

    response = client.models.generate_content(
        model=MODEL,
        contents=email_text,
        config=config
    )

    return response.text


def init_gemini():
    return create_gemini_client(GEMINI_API_KEY)


# Gmail Draft

def build_raw_email(record, body_text):
    msg = EmailMessage()

    subject = record["subject"]
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    msg["To"] = ", ".join(record["to"])
    msg["From"] = record["from"]
    msg["Subject"] = subject

    msg.set_content(body_text)

    raw = base64.urlsafe_b64encode(
        msg.as_bytes()
    ).decode("utf-8")

    return raw


def create_gmail_draft(service, raw_message, thread_id):
    draft = (
        service.users()
        .drafts()
        .create(
            userId="me",
            body={
                "message": {
                    "raw": raw_message,
                    "threadId": thread_id,
                }
            },
        )
        .execute()
    )

    return draft


def create_followup_draft(service, client, record):
    draft_body = generate_followup_draft(client, record)
    raw_email = build_raw_email(record, draft_body)

    return create_gmail_draft(
        service,
        raw_email,
        record["thread_id"]
    )


# Calendar


def build_calendar_service(creds):
    return build("calendar", "v3", credentials=creds)


def create_followup_reminder(calendar_service, record, days=3):
    tz = pytz.timezone("America/New_York")

    start_time = datetime.now(tz) + timedelta(days=days)
    end_time = start_time + timedelta(minutes=10)

    event = {
        "summary": "Follow up on email",
        "description": f"Follow up with {', '.join(record['to'])}\n\nSubject: {record['subject']}",
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": str(tz),
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": str(tz),
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    return (
        calendar_service.events()
        .insert(calendarId="primary", body=event)
        .execute()
    )


# Main

def main():
    try:
        creds = get_valid_credentials()
        service = build_gmail_service(creds)
        calendar_service = build_calendar_service(creds)
        client = init_gemini()

        last_message_id = None
        poll_interval = 30
        first = True

        print(f"Starting Gmail Sent box monitoring at {datetime.now()}")
        print(f"Checking for new sent messages every {poll_interval} seconds...")

        while True:
            try:
                ids = list_sent_message_ids(service, max_results=1)

                if not ids:
                    print(f"[{datetime.now()}] No messages in Sent box", end='\r')
                    time.sleep(poll_interval)
                    continue

                current_message_id = ids[0]["id"]

                # Dont eval sent message before start
                if first:
                    last_message_id = current_message_id
                    first = False

                if current_message_id != last_message_id:
                    print(f"\n[{datetime.now()}] New message detected!")

                    try:
                        data = extract_sent_message(service, current_message_id)
                        print(f"Subject: {data.get('subject', 'No subject')}")
                        print(f"To: {data.get('to', 'Unknown')}")

                        prompt = build_email_prompt(data)
                        result = evaluate_followup(client, prompt)
                        parsed_results = extract_json(result)

                        if parsed_results["follow_up_required"]:
                            print("✓ Follow-up needed")
                            # followup_text = generate_followup_draft(client, data)

                            draft = create_followup_draft(service, client, data)
                            print(f"✓ Draft created: {draft['id']}")

                            reminder = create_followup_reminder(calendar_service, data)
                            print(f"✓ Reminder created: {reminder['id']}")
                        else:
                            # print(parsed_results["reason"])
                            print("○ No follow-up needed")

                        last_message_id = current_message_id

                    except Exception as e:
                        print(f"✗ Error processing message: {e}")
                        last_message_id = current_message_id
                else:
                    print(f"[{datetime.now()}] No new messages", end='\r')

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\n\nStopping monitoring...")
                break
            except Exception as e:
                print(f"\n✗ Polling error: {e}")
                print("Retrying in 30 seconds...")
                time.sleep(30)

        print(f"\nMonitoring stopped at {datetime.now()}")

    except HttpError as error:
        print(f"An error occurred: {error}")


# Testing

def testing():
    try:
        creds = get_valid_credentials()
        service = build_gmail_service(creds)
        calendar_service = build_calendar_service(creds)

        client = init_gemini()

        ids = list_sent_message_ids(service, max_results=2)
        for m in ids:
            data = extract_sent_message(service, m["id"])
            print(data)

            prompt = build_email_prompt(data)
            result = evaluate_followup(client, prompt)
            print(result)
            parsed_results = extract_json(result)
            if parsed_results["follow_up_required"]:
                print("Follow-up needed")
                print(generate_followup_draft(client, data))
                draft = create_followup_draft(service, client, data)
                print("Draft created:", draft["id"])
                reminder = create_followup_reminder(calendar_service, data)
                print("Reminder created:", reminder["id"])
            else:
                print("No follow-up needed")

        '''
        ids = list_sent_message_ids(service, max_results=1)
        message_id = ids[0]["id"]
        data = extract_sent_message(service, message_id)
        prompt = build_email_prompt(data)
        print(data)

        client = init_gemini()
        result = evaluate_followup(client, prompt)
        print(result)
        '''

    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
