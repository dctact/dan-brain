import os
import json
from datetime import datetime, timedelta
import pytz
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from todoist_api_python.api import TodoistAPI
import anthropic
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
import pickle
import base64

# --- INITIALIZATION ---

# Restore Google token from environment variable if file doesn't exist
def restore_google_token():
    token_b64 = os.environ.get("GOOGLE_TOKEN_BASE64")
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    token_path = f"{data_dir}/google_token.pickle"

    print(f"[Startup] Token restore - data_dir: {data_dir}, token_path: {token_path}")

    if token_b64 and not os.path.exists(token_path):
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(token_path, 'wb') as f:
                f.write(base64.b64decode(token_b64))
            print(f"[Startup] Restored Google token to {token_path}")
        except Exception as e:
            print(f"[Startup] Failed to restore Google token: {e}")
    elif os.path.exists(token_path):
        print(f"[Startup] Token already exists at {token_path}")

restore_google_token()

# Restore knowledge graph from environment variable if file doesn't exist
def restore_knowledge_graph():
    kg_b64 = os.environ.get("KNOWLEDGE_GRAPH_BASE64")
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    kg_path = f"{data_dir}/knowledge_graph.json"

    if kg_b64 and not os.path.exists(kg_path):
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(kg_path, 'w') as f:
                f.write(base64.b64decode(kg_b64).decode('utf-8'))
            print(f"[Startup] Restored knowledge graph to {kg_path}")
        except Exception as e:
            print(f"[Startup] Failed to restore knowledge graph: {e}")
    elif os.path.exists(kg_path):
        print(f"[Startup] Knowledge graph already exists at {kg_path}")

restore_knowledge_graph()

# Get allowed hosts from environment (for Railway/ngrok flexibility)
ALLOWED_HOST = os.environ.get("ALLOWED_HOST", "dan-brain.ngrok.app")

# Initialize the MCP Server with security settings
mcp = FastMCP(
    "Dan_Master_Brain",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[ALLOWED_HOST, "localhost:*", "127.0.0.1:*"],
        allowed_origins=[f"https://{ALLOWED_HOST}", "http://localhost:*"],
    )
)

# Initialize Todoist API
todoist = TodoistAPI(os.environ.get("TODOIST_API_TOKEN"))

# Initialize Anthropic client (fallback)
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# Initialize Gemini client (primary for large context)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Persistent data path (configurable for Railway vs local Docker)
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

# Persistent Memory Path (Mapped to Docker Volume)
MEMORY_FILE = f"{DATA_DIR}/knowledge_graph.json"
GOOGLE_TOKEN_FILE = f"{DATA_DIR}/google_token.pickle"
GOOGLE_CREDENTIALS_FILE = f"{DATA_DIR}/google_credentials.json"

# Google Calendar scopes
GOOGLE_SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# Recent action items storage (from transcript processing)
RECENT_ACTIONS_FILE = f"{DATA_DIR}/recent_actions.json"

# Transcript archive for pattern detection
TRANSCRIPT_ARCHIVE_FILE = f"{DATA_DIR}/transcript_archive.json"

# Last conversation for short-term memory
LAST_CONVERSATION_FILE = f"{DATA_DIR}/last_conversation.json"

# --- TOOL 1: TIME (The Grounding) ---
@mcp.tool()
def get_current_time() -> str:
    """
    Returns the current date, time, and day of the week.
    CRITICAL: Call this IMMEDIATELY at the start of every session to ground yourself in time.
    """
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y at %I:%M %p (%Z)")

# --- TOOL 2: MEMORY (The Second Brain) ---

# Valid categories for organizing memories
MEMORY_CATEGORIES = [
    "tools",        # Software, apps, dev tools (Claude Code, VSCode, etc.)
    "gear",         # Physical equipment (cameras, computers, etc.)
    "preferences",  # Likes, dislikes, styles, approaches
    "projects",     # Current/past projects and their status
    "skills",       # Languages, frameworks, abilities
    "people",       # Contacts, collaborators
    "workflows",    # How Dan does things
    "other"         # Catch-all
]

@mcp.tool()
def add_memory(subject: str, relation: str, object_entity: str, category: str = "other") -> str:
    """
    Saves a fact to Dan's long-term memory graph. BE PROACTIVE - save anything personal!

    WHEN TO CALL THIS (be aggressive):
    - Dan mentions ANY tool, app, or software he uses -> category: "tools"
    - Dan mentions equipment or gear he owns -> category: "gear"
    - Dan expresses a preference or opinion -> category: "preferences"
    - Dan mentions a project he's working on -> category: "projects"
    - Dan mentions a skill or technology he knows -> category: "skills"
    - Dan mentions a person or collaborator -> category: "people"
    - Dan describes how he does something -> category: "workflows"

    Args:
        subject: Who/what (normalize "I", "me", "my" to "Dan")
        relation: The relationship (e.g., "uses", "owns", "prefers", "knows", "is working on")
        object_entity: The object of the relationship
        category: One of: tools, gear, preferences, projects, skills, people, workflows, other

    Examples:
        "I use Claude Code" -> add_memory("Dan", "uses", "Claude Code", "tools")
        "I prefer dark mode" -> add_memory("Dan", "prefers", "dark mode", "preferences")
        "I'm working on an ElevenLabs integration" -> add_memory("Dan", "is building", "ElevenLabs integration", "projects")
    """
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

    graph = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            try:
                graph = json.load(f)
            except json.JSONDecodeError:
                graph = []

    # Normalize subject
    if subject.lower() in ["i", "me", "my", "myself"]:
        subject = "Dan"

    # Validate category
    if category.lower() not in MEMORY_CATEGORIES:
        category = "other"
    else:
        category = category.lower()

    new_fact = {
        "s": subject,
        "r": relation,
        "o": object_entity,
        "category": category
    }

    # Check for duplicates (ignore category in comparison)
    for fact in graph:
        if fact.get("s") == subject and fact.get("r") == relation and fact.get("o") == object_entity:
            return f"Already known: {subject} {relation} {object_entity}"

    graph.append(new_fact)
    with open(MEMORY_FILE, "w") as f:
        json.dump(graph, f, indent=2)

    return f"Memory Stored [{category}]: {subject} {relation} {object_entity}"

@mcp.tool()
def check_personal_context(topic: str = None, category: str = None) -> str:
    """
    Search Dan's memory. Call this BEFORE making recommendations or assumptions!

    Args:
        topic: Search term (searches across all fields). Optional if category provided.
        category: Filter by category (tools, gear, preferences, projects, skills, people, workflows).
                  If provided alone, returns ALL memories in that category.

    Examples:
        check_personal_context(topic="code") -> finds anything mentioning "code"
        check_personal_context(category="tools") -> lists ALL tools Dan uses
        check_personal_context(topic="camera", category="gear") -> cameras in gear category
    """
    if not os.path.exists(MEMORY_FILE):
        return "No memories yet. Graph is empty."

    with open(MEMORY_FILE, "r") as f:
        try:
            graph = json.load(f)
        except:
            return "Memory file is corrupt or empty."

    if not graph:
        return "No memories stored yet."

    results = []

    for item in graph:
        item_category = item.get("category", "other")
        combined = f"{item['s']} {item['r']} {item['o']}".lower()

        # Filter by category if provided
        if category and item_category != category.lower():
            continue

        # Filter by topic if provided
        if topic and topic.lower() not in combined:
            continue

        cat_tag = f"[{item_category}]" if item_category else ""
        results.append(f"- {cat_tag} {item['s']} {item['r']} {item['o']}")

    if results:
        header = f"Found {len(results)} memories"
        if category:
            header += f" in '{category}'"
        if topic:
            header += f" matching '{topic}'"
        return header + ":\n" + "\n".join(results)

    return f"No memories found" + (f" for topic '{topic}'" if topic else "") + (f" in category '{category}'" if category else "")

@mcp.tool()
def list_all_memories() -> str:
    """
    Returns ALL memories in Dan's knowledge graph, organized by category.
    Each memory has an index [n] for use with cleanup_memories().

    Call this when:
    - Starting complex conversations
    - Dan asks to review or clean up memories
    - You need full context before making recommendations

    When cleaning up: Look for redundancies, similar entries, or things that
    could be consolidated. Use cleanup_memories() with the indices shown.
    """
    if not os.path.exists(MEMORY_FILE):
        return "No memories yet. Graph is empty."

    with open(MEMORY_FILE, "r") as f:
        try:
            graph = json.load(f)
        except:
            return "Memory file is corrupt or empty."

    if not graph:
        return "No memories stored yet."

    # Organize by category with indices
    by_category = {}
    for idx, item in enumerate(graph):
        cat = item.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f"  [{idx}] {item['s']} {item['r']} {item['o']}")

    # Format output
    output = [f"Dan's Knowledge Graph ({len(graph)} memories):"]
    for cat in MEMORY_CATEGORIES:
        if cat in by_category:
            output.append(f"\n[{cat.upper()}]")
            output.extend(by_category[cat])

    return "\n".join(output)

@mcp.tool()
def cleanup_memories(action: str, indices: str, replacement: str = None) -> str:
    """
    Delete or merge memories. Use indices from list_all_memories().

    Args:
        action: "delete" or "merge"
        indices: Comma-separated indices (e.g., "3,7,12")
        replacement: For merge only - "subject|relation|object|category"

    Examples:
        cleanup_memories("delete", "5,8")
        cleanup_memories("merge", "3,7", "Dan|uses|Claude Code|tools")
    """
    if not os.path.exists(MEMORY_FILE):
        return "No memories to clean up."

    with open(MEMORY_FILE, "r") as f:
        try:
            graph = json.load(f)
        except:
            return "Memory file is corrupt."

    # Parse indices
    try:
        idx_list = [int(i.strip()) for i in indices.split(",")]
    except:
        return "Invalid indices format. Use comma-separated numbers like '3,7,12'"

    # Validate indices
    for idx in idx_list:
        if idx < 0 or idx >= len(graph):
            return f"Index {idx} out of range. Valid range: 0-{len(graph)-1}"

    if action == "delete":
        # Delete in reverse order to maintain indices
        deleted = []
        for idx in sorted(idx_list, reverse=True):
            deleted.append(f"{graph[idx]['s']} {graph[idx]['r']} {graph[idx]['o']}")
            del graph[idx]

        with open(MEMORY_FILE, "w") as f:
            json.dump(graph, f, indent=2)

        return f"Deleted {len(deleted)} memories:\n" + "\n".join(f"  - {d}" for d in deleted)

    elif action == "merge":
        if not replacement:
            return "Merge requires a replacement string: 'subject|relation|object|category'"

        parts = replacement.split("|")
        if len(parts) != 4:
            return "Replacement must have 4 parts: subject|relation|object|category"

        subject, relation, obj, category = [p.strip() for p in parts]

        if category.lower() not in MEMORY_CATEGORIES:
            category = "other"

        # Delete old entries (reverse order)
        deleted = []
        for idx in sorted(idx_list, reverse=True):
            deleted.append(f"{graph[idx]['s']} {graph[idx]['r']} {graph[idx]['o']}")
            del graph[idx]

        # Add new consolidated entry
        new_fact = {"s": subject, "r": relation, "o": obj, "category": category.lower()}
        graph.append(new_fact)

        with open(MEMORY_FILE, "w") as f:
            json.dump(graph, f, indent=2)

        return f"Merged {len(deleted)} memories into one:\n" + \
               "Removed:\n" + "\n".join(f"  - {d}" for d in deleted) + \
               f"\nAdded: {subject} {relation} {obj} [{category}]"

    else:
        return "Invalid action. Use 'delete' or 'merge'"

# --- TOOL 3: TODOIST (Action Layer) ---
@mcp.tool()
def add_task(task_name: str, due_date: str = "today", priority: int = 1) -> str:
    """
    Creates a new task in Dan's Todoist. Use this to capture action items!

    WHEN TO CALL THIS:
    - Dan commits to doing something -> capture it immediately
    - Dan mentions a deadline or appointment to remember
    - You identify an action item from the conversation
    - Dan asks you to remind him about something

    Args:
        task_name: Clear, actionable task description (start with verb)
        due_date: Natural language - 'today', 'tomorrow at 10am', 'next friday', 'in 3 days'
        priority: 4 (Urgent/red), 3 (High/orange), 2 (Medium/yellow), 1 (Normal/none)

    Examples:
        add_task("Call dentist to schedule cleaning", "tomorrow", 2)
        add_task("Review project proposal", "friday at 2pm", 3)
        add_task("Buy birthday gift for Liz", "today", 4)

    Pro tip: Be specific. "Work on project" is bad. "Draft intro section of proposal" is good.
    """
    try:
        t = todoist.add_task(content=task_name, due_string=due_date, priority=priority)
        due_info = t.due.string if t.due else due_date
        return f"Task Added: '{t.content}' (Due: {due_info}, Priority: {priority})"
    except Exception as e:
        return f"Error adding task: {str(e)}"

@mcp.tool()
def get_tasks(label: str = None) -> str:
    """
    Lists Dan's current tasks from Todoist. Use this for context and accountability!

    WHEN TO CALL THIS:
    - At conversation start to understand what Dan should be working on
    - When Dan seems unfocused - remind him what's on his plate
    - Before suggesting new tasks - check what's already there
    - When Dan asks "what should I do?" or "what's on my list?"

    Args:
        label: Optional filter (e.g. 'work', 'personal'). Usually omit for full picture.

    Returns tasks with priority markers and due dates. Use this info to:
    - Hold Dan accountable to existing commitments
    - Avoid adding duplicate tasks
    - Help prioritize what to tackle next
    """
    try:
        # get_tasks returns an iterator of task batches in v3.x
        all_tasks = []
        if label:
            task_iterator = todoist.get_tasks(label=label)
        else:
            task_iterator = todoist.get_tasks()

        # Collect tasks from iterator (limit to 50 to avoid overwhelming response)
        for task_batch in task_iterator:
            all_tasks.extend(task_batch)
            if len(all_tasks) >= 50:
                break

        if not all_tasks:
            return "No tasks found."

        # Format output
        output = []
        for t in all_tasks[:50]:
            due_str = t.due.string if t.due else "No due date"
            priority_marker = "!" * t.priority if t.priority > 1 else ""
            output.append(f"- {priority_marker}{t.content} (Due: {due_str})")

        return f"Found {len(all_tasks)} tasks:\n" + "\n".join(output)
    except Exception as e:
        return f"Error fetching tasks: {str(e)}"

# --- TOOL 4: GOOGLE CALENDAR ---

def get_google_calendar_service():
    """Get authenticated Google Calendar service."""
    creds = None

    print(f"[Calendar] Looking for token at: {GOOGLE_TOKEN_FILE}")
    print(f"[Calendar] File exists: {os.path.exists(GOOGLE_TOKEN_FILE)}")

    # Load existing token
    if os.path.exists(GOOGLE_TOKEN_FILE):
        with open(GOOGLE_TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    # Refresh or return None if no valid creds
    if creds and creds.expired and creds.refresh_token:
        print(f"[Calendar] Token expired, refreshing...")
        try:
            creds.refresh(GoogleAuthRequest())
            with open(GOOGLE_TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
            print(f"[Calendar] Token refreshed successfully")
        except Exception as e:
            print(f"[Calendar] Token refresh failed: {e}")
            creds = None

    if not creds or not creds.valid:
        print(f"[Calendar] No valid creds - creds exists: {creds is not None}, valid: {creds.valid if creds else 'N/A'}")
        return None

    print(f"[Calendar] Building calendar service...")
    return build('calendar', 'v3', credentials=creds)

@mcp.tool()
def get_calendar_events(hours_ahead: int = 24, include_past_today: bool = False) -> str:
    """
    Get Dan's upcoming calendar events from Google Calendar. Essential for time awareness!

    WHEN TO CALL THIS (be proactive):
    - IMMEDIATELY at conversation start - know Dan's schedule before anything else
    - Before suggesting task timing - avoid scheduling conflicts
    - When Dan asks "what do I have today?", "am I free?", "what's my schedule?"
    - Before recommending deep work blocks - find the gaps
    - When Dan mentions a meeting or appointment - verify details
    - When planning tomorrow - use hours_ahead=36 to see next day

    Args:
        hours_ahead: How many hours to look forward (default 24). Use 36+ to see tomorrow.
        include_past_today: If True, includes events from earlier today (default False)

    Returns formatted list with:
    - Event time (in Dan's timezone: America/New_York)
    - Event title/summary
    - Location (if specified)
    - Duration context

    WHAT TO DO WITH THIS INFO:
    - Identify free blocks for focused work (look for 2+ hour gaps)
    - Note travel time needed between locations
    - Remind Dan of upcoming commitments ("You have X in 30 minutes")
    - Suggest realistic task deadlines based on availability
    - Factor in prep time before important meetings

    IMPORTANT: This is READ-ONLY. You cannot create, modify, or delete events.
    If Dan asks to add/change calendar events, tell him you can only view the calendar
    and he'll need to make changes directly in Google Calendar.

    Examples:
        get_calendar_events() -> next 24 hours
        get_calendar_events(hours_ahead=48) -> next 2 days
        get_calendar_events(hours_ahead=8) -> just today's remaining events
    """
    service = get_google_calendar_service()
    if not service:
        return "Google Calendar not connected. Run setup_google_calendar first."

    try:
        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz)
        end_time = now + timedelta(hours=hours_ahead)

        events_result = service.events().list(
            calendarId='primary',
            timeMin=now.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])

        if not events:
            return f"No calendar events in the next {hours_ahead} hours."

        output = [f"Calendar events (next {hours_ahead} hours):"]
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            # Parse and format time nicely
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                start_str = start_dt.astimezone(tz).strftime("%I:%M %p")
            except:
                start_str = start

            title = event.get('summary', 'No title')
            location = event.get('location', '')

            entry = f"- {start_str}: {title}"
            if location:
                entry += f" ({location})"
            output.append(entry)

        return "\n".join(output)

    except Exception as e:
        return f"Error fetching calendar: {str(e)}"

@mcp.tool()
def setup_google_calendar_url() -> str:
    """
    Get the OAuth URL to connect Google Calendar.
    User must visit this URL and authorize, then provide the auth code.
    """
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        return f"Missing google_credentials.json in {DATA_DIR}/. Download from Google Cloud Console."

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            GOOGLE_CREDENTIALS_FILE,
            GOOGLE_SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        return f"Visit this URL to authorize:\n{auth_url}\n\nThen call complete_google_calendar_setup with the authorization code."
    except Exception as e:
        return f"Error generating auth URL: {str(e)}"

@mcp.tool()
def complete_google_calendar_setup(auth_code: str) -> str:
    """
    Complete Google Calendar OAuth with the authorization code.

    Args:
        auth_code: The code from Google after authorizing
    """
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        return "Missing google_credentials.json"

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            GOOGLE_CREDENTIALS_FILE,
            GOOGLE_SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        flow.fetch_token(code=auth_code)
        creds = flow.credentials

        with open(GOOGLE_TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

        return "Google Calendar connected successfully!"
    except Exception as e:
        return f"Error completing setup: {str(e)}"


# --- TOOL 5: TRANSCRIPT PROCESSING ---

@mcp.tool()
def process_transcript(transcript: str, agent_name: str = "Agent") -> str:
    """
    Process a voice conversation transcript and extract action items.
    Uses Claude to intelligently extract tasks, commitments, and priorities.

    Args:
        transcript: The full conversation transcript text
        agent_name: Name of the agent in the conversation

    Returns extracted action items and saves them for get_rest_of_day.
    """
    if not transcript or len(transcript.strip()) < 20:
        return "Transcript too short to process."

    try:
        prompt = f"""Analyze this voice conversation transcript and extract actionable information.

TRANSCRIPT:
{transcript}

Extract and return as JSON:
{{
    "action_items": [
        {{
            "task": "description of what needs to be done",
            "owner": "user" or "agent",
            "priority": "high", "medium", or "low",
            "deadline": "specific time/date mentioned or null",
            "source": "brief quote from transcript"
        }}
    ],
    "commitments": ["things the user said they would do"],
    "key_decisions": ["important decisions made"],
    "follow_ups": ["things to check on later"]
}}

Only include items that are clearly actionable. Be concise. If nothing actionable, return empty arrays."""

        # Use Gemini for large context processing
        response = gemini_model.generate_content(prompt)
        result_text = response.text

        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = {"action_items": [], "commitments": [], "key_decisions": [], "follow_ups": []}

        # Save to recent actions file
        tz = pytz.timezone("America/New_York")
        timestamp = datetime.now(tz).isoformat()

        recent_actions = []
        if os.path.exists(RECENT_ACTIONS_FILE):
            with open(RECENT_ACTIONS_FILE, 'r') as f:
                try:
                    recent_actions = json.load(f)
                except:
                    recent_actions = []

        # Add new extraction
        recent_actions.append({
            "timestamp": timestamp,
            "agent_name": agent_name,
            "extracted": extracted
        })

        # Keep only last 10 conversations
        recent_actions = recent_actions[-10:]

        with open(RECENT_ACTIONS_FILE, 'w') as f:
            json.dump(recent_actions, f, indent=2)

        # Also archive transcript for pattern detection
        transcript_archive = []
        if os.path.exists(TRANSCRIPT_ARCHIVE_FILE):
            with open(TRANSCRIPT_ARCHIVE_FILE, 'r') as f:
                try:
                    transcript_archive = json.load(f)
                except:
                    transcript_archive = []

        transcript_archive.append({
            "timestamp": timestamp,
            "agent_name": agent_name,
            "transcript": transcript[:5000]  # Limit size
        })

        # Keep last 20 transcripts for pattern analysis
        transcript_archive = transcript_archive[-20:]

        with open(TRANSCRIPT_ARCHIVE_FILE, 'w') as f:
            json.dump(transcript_archive, f, indent=2)

        # Store a short summary for quick retrieval
        with open(LAST_CONVERSATION_FILE, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "agent_name": agent_name,
                "summary": extracted.get("key_decisions", [])[:3],
                "action_items": [item.get("task") for item in extracted.get("action_items", []) if item.get("owner") == "user"][:5],
                "commitments": extracted.get("commitments", [])[:5],
                "transcript_preview": transcript[:1500]
            }, f, indent=2)

        # Auto-trigger pattern detection every 5 transcripts
        if len(transcript_archive) >= 5 and len(transcript_archive) % 5 == 0:
            try:
                pattern_result = detect_patterns(min_transcripts=5)
                print(f"[Auto Pattern Detection] {pattern_result[:200]}...")
            except Exception as e:
                print(f"[Auto Pattern Detection] Error: {e}")

        # Format response
        output = [f"Processed conversation with {agent_name}:"]

        if extracted.get("action_items"):
            output.append("\nACTION ITEMS:")
            for item in extracted["action_items"]:
                priority_marker = "!" * (3 if item.get("priority") == "high" else 2 if item.get("priority") == "medium" else 1)
                output.append(f"  {priority_marker} {item['task']}")
                if item.get("deadline"):
                    output.append(f"      Due: {item['deadline']}")

        if extracted.get("commitments"):
            output.append("\nYOUR COMMITMENTS:")
            for c in extracted["commitments"]:
                output.append(f"  → {c}")

        if extracted.get("follow_ups"):
            output.append("\nFOLLOW-UPS:")
            for f in extracted["follow_ups"]:
                output.append(f"  • {f}")

        return "\n".join(output)

    except Exception as e:
        return f"Error processing transcript: {str(e)}"


# --- TOOL 6: PATTERN DETECTION ---

@mcp.tool()
def get_previous_conversation() -> str:
    """
    Call this at the start of every conversation to get context for continuity.

    Returns a summary of the last conversation including:
    - When it happened
    - Key discussion points
    - Action items mentioned
    - A snippet of the conversation

    Use this to maintain continuity across sessions.
    """
    if not os.path.exists(LAST_CONVERSATION_FILE):
        return "No previous conversation found. This appears to be our first chat!"

    try:
        with open(LAST_CONVERSATION_FILE, 'r') as f:
            last = json.load(f)

        output = [f"PREVIOUS CONVERSATION ({last.get('agent_name', 'Agent')}) - {last.get('timestamp', 'Unknown time')[:16]}"]
        output.append("=" * 40)

        if last.get("commitments"):
            output.append("\n🎯 DAN'S COMMITMENTS (hold him accountable!):")
            for c in last["commitments"]:
                output.append(f"  → {c}")

        if last.get("action_items"):
            output.append("\n📋 ACTION ITEMS from last time:")
            for item in last["action_items"]:
                output.append(f"  • {item}")

        if last.get("summary"):
            output.append("\n💡 KEY DECISIONS:")
            for s in last["summary"]:
                output.append(f"  - {s}")

        if last.get("transcript_preview"):
            # Show last bit of conversation for context
            preview = last["transcript_preview"][-500:]
            output.append(f"\n📝 Last conversation snippet:\n{preview}...")

        return "\n".join(output)

    except Exception as e:
        return f"Error reading previous conversation: {str(e)}"


@mcp.tool()
def detect_patterns(min_transcripts: int = 5) -> str:
    """
    Analyze archived transcripts to detect behavioral patterns.
    Only saves patterns that appear across multiple conversations.

    Args:
        min_transcripts: Minimum number of transcripts needed before analysis (default 5)

    This tool looks for:
    - Communication preferences (what approaches get action vs resistance)
    - Work patterns (productive times, task types completed vs avoided)
    - Motivation triggers (what gets Dan unstuck)
    - Follow-through indicators (project types completed vs abandoned)

    Only saves patterns observed in 2+ transcripts. Quality over quantity.
    """
    if not os.path.exists(TRANSCRIPT_ARCHIVE_FILE):
        return "No transcripts archived yet. Have more conversations first."

    with open(TRANSCRIPT_ARCHIVE_FILE, 'r') as f:
        try:
            archive = json.load(f)
        except:
            return "Error reading transcript archive."

    if len(archive) < min_transcripts:
        return f"Only {len(archive)} transcripts archived. Need at least {min_transcripts} for meaningful pattern detection."

    # Load existing memories to avoid duplicates
    existing_patterns = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            try:
                graph = json.load(f)
                existing_patterns = [
                    f"{m['s']} {m['r']} {m['o']}"
                    for m in graph
                    if m.get('category') == 'preferences'
                ]
            except:
                pass

    # Prepare transcripts for analysis
    transcript_texts = []
    for entry in archive[-15:]:  # Analyze last 15 transcripts
        transcript_texts.append(f"[{entry['timestamp'][:10]}]\n{entry['transcript'][:2000]}")

    combined = "\n\n---\n\n".join(transcript_texts)

    try:
        prompt = f"""Analyze these conversation transcripts between Dan and his AI assistant to identify RECURRING behavioral patterns.

TRANSCRIPTS:
{combined}

EXISTING KNOWN PATTERNS (don't repeat these):
{chr(10).join(existing_patterns[:20]) if existing_patterns else "None yet"}

Find patterns that appear in AT LEAST 2 different conversations. Focus on:
- Communication styles that work (direct vs gentle, questions vs statements)
- Task/project types Dan follows through on vs avoids
- Times/conditions when Dan is productive vs stuck
- What approaches get action vs resistance
- Motivation triggers that work

Return as JSON:
{{
    "patterns": [
        {{
            "observation": "specific pattern observed",
            "evidence_count": number of conversations showing this,
            "memory_format": "Dan [relation] [object]"
        }}
    ],
    "insufficient_data": ["patterns noticed once but need more evidence"]
}}

CRITICAL: Only include patterns with evidence_count >= 2. Be specific and actionable, not vague."""

        # Use Gemini for large context pattern analysis
        response = gemini_model.generate_content(prompt)
        result_text = response.text

        # Parse response
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if not json_match:
            return "Could not parse pattern analysis results."

        analysis = json.loads(json_match.group())
        patterns = analysis.get("patterns", [])
        insufficient = analysis.get("insufficient_data", [])

        # Save confirmed patterns to memory
        saved_count = 0
        for pattern in patterns:
            if pattern.get("evidence_count", 0) >= 2:
                memory_format = pattern.get("memory_format", "")
                if memory_format and memory_format not in existing_patterns:
                    # Parse and save
                    parts = memory_format.split(" ", 2)
                    if len(parts) >= 3:
                        subject = parts[0]
                        # Find relation - everything between subject and last meaningful phrase
                        rest = " ".join(parts[1:])
                        # Simple heuristic: relation is first 2-4 words
                        words = rest.split()
                        if len(words) >= 2:
                            relation = " ".join(words[:3])
                            obj = " ".join(words[3:]) if len(words) > 3 else words[-1]
                            result = add_memory(subject, relation, obj, "preferences")
                            if "Stored" in result:
                                saved_count += 1

        # Format output
        output = [f"Pattern Analysis ({len(archive)} transcripts analyzed):"]

        if patterns:
            output.append(f"\nCONFIRMED PATTERNS ({len(patterns)} found, {saved_count} new saved):")
            for p in patterns:
                output.append(f"  • {p['observation']} (seen {p.get('evidence_count', '?')}x)")

        if insufficient:
            output.append(f"\nNEED MORE EVIDENCE ({len(insufficient)} tentative):")
            for i in insufficient[:5]:
                output.append(f"  ? {i}")

        if not patterns and not insufficient:
            output.append("\nNo clear patterns detected yet. Keep having conversations!")

        return "\n".join(output)

    except Exception as e:
        return f"Error analyzing patterns: {str(e)}"


# --- TOOL 7: REST OF DAY (Unified Action View) ---

@mcp.tool()
def get_rest_of_day() -> str:
    """
    THE BIG PICTURE: Get everything Dan needs to do, all in one place.
    Combines: Calendar events + Todoist tasks + Recent conversation action items

    WHEN TO CALL THIS:
    - At the START of conversations for full context
    - When Dan asks "what should I focus on?" or "what's my day look like?"
    - When helping Dan prioritize or plan
    - After adding tasks to show the updated picture

    This is your PRIMARY tool for understanding Dan's obligations. It shows:
    - 📅 SCHEDULED: Calendar events for the next 18 hours
    - ✅ TASKS: Todoist items due today or overdue
    - 💬 FROM CONVERSATIONS: Action items extracted from recent voice chats

    Use this to give Dan a clear, prioritized view of what needs attention.
    """
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)

    output = [f"REST OF YOUR DAY - {now.strftime('%A, %B %d')}"]
    output.append("=" * 40)

    # --- CALENDAR EVENTS ---
    output.append("\n📅 SCHEDULED:")
    service = get_google_calendar_service()
    if service:
        try:
            end_time = now + timedelta(hours=18)  # Look 18 hours ahead
            events_result = service.events().list(
                calendarId='primary',
                timeMin=now.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            if events:
                for event in events:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        start_str = start_dt.astimezone(tz).strftime("%I:%M %p")
                    except:
                        start_str = start
                    output.append(f"  {start_str} - {event.get('summary', 'No title')}")
            else:
                output.append("  No more events today")
        except Exception as e:
            output.append(f"  (Calendar error: {str(e)})")
    else:
        output.append("  (Calendar not connected)")

    # --- TODOIST TASKS ---
    output.append("\n✅ TASKS (Todoist):")
    try:
        all_tasks = []
        # Get all active tasks
        task_iterator = todoist.get_tasks()
        for task_batch in task_iterator:
            all_tasks.extend(task_batch)
            if len(all_tasks) >= 50:
                break

        # Filter for today/overdue
        today_str = now.strftime("%Y-%m-%d")
        today_tasks = []
        for t in all_tasks:
            if t.due and t.due.date:
                due_date = str(t.due.date)[:10]  # Handle both date and datetime
                if due_date <= today_str:
                    today_tasks.append(t)

        if today_tasks:
            # Sort by priority (4 = urgent, 1 = normal)
            today_tasks.sort(key=lambda t: -t.priority)
            for task in today_tasks[:15]:
                priority_marker = "‼️" if task.priority >= 3 else "❗" if task.priority == 2 else "  "
                output.append(f"  {priority_marker} {task.content}")
        else:
            output.append("  No tasks due today")
    except Exception as e:
        output.append(f"  (Todoist error: {str(e)})")

    # --- RECENT CONVERSATION ACTIONS ---
    output.append("\n💬 FROM RECENT CONVERSATIONS:")
    if os.path.exists(RECENT_ACTIONS_FILE):
        try:
            with open(RECENT_ACTIONS_FILE, 'r') as f:
                recent_actions = json.load(f)

            # Get actions from last 24 hours
            cutoff = (now - timedelta(hours=24)).isoformat()
            recent = [a for a in recent_actions if a.get("timestamp", "") > cutoff]

            if recent:
                for session in recent[-3:]:  # Last 3 conversations
                    extracted = session.get("extracted", {})
                    action_items = extracted.get("action_items", [])
                    commitments = extracted.get("commitments", [])

                    for item in action_items:
                        if item.get("owner") == "user":
                            priority = item.get("priority", "medium")
                            marker = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                            output.append(f"  {marker} {item['task']}")

                    for c in commitments:
                        output.append(f"  → {c}")
            else:
                output.append("  No recent conversation action items")
        except:
            output.append("  (Error reading recent actions)")
    else:
        output.append("  No recent conversations processed")

    output.append("\n" + "=" * 40)
    return "\n".join(output)


# --- REST API FOR iOS APP ---
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request

async def api_process_transcript(request: Request):
    """REST endpoint for iOS app to process transcripts."""
    try:
        body = await request.json()
        transcript = body.get("transcript", "")
        agent_name = body.get("agent_name", "Agent")

        result = process_transcript(transcript, agent_name)
        return JSONResponse({"success": True, "result": result})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def api_get_rest_of_day(request: Request):
    """REST endpoint for iOS app to get unified rest-of-day view."""
    try:
        result = get_rest_of_day()
        return JSONResponse({"success": True, "result": result})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def api_get_rest_of_day_structured(request: Request):
    """REST endpoint returning structured JSON for iOS app display."""
    try:
        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz)

        response_data = {
            "date": now.strftime("%A, %B %d"),
            "timestamp": now.isoformat(),
            "calendar_events": [],
            "todoist_tasks": [],
            "conversation_actions": []
        }

        # Calendar events
        service = get_google_calendar_service()
        if service:
            try:
                end_time = now + timedelta(hours=18)  # Look 18 hours ahead
                events_result = service.events().list(
                    calendarId='primary',
                    timeMin=now.isoformat(),
                    timeMax=end_time.isoformat(),
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                events = events_result.get('items', [])

                for event in events:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        start_str = start_dt.astimezone(tz).strftime("%I:%M %p")
                    except:
                        start_str = start

                    response_data["calendar_events"].append({
                        "time": start_str,
                        "title": event.get('summary', 'No title'),
                        "location": event.get('location', '')
                    })
            except Exception as e:
                response_data["calendar_error"] = str(e)

        # Todoist tasks
        try:
            all_tasks = []
            task_iterator = todoist.get_tasks()
            for task_batch in task_iterator:
                all_tasks.extend(task_batch)
                if len(all_tasks) >= 50:
                    break

            # Filter for today/overdue
            today_str = now.strftime("%Y-%m-%d")
            today_tasks = []
            for t in all_tasks:
                if t.due and t.due.date:
                    due_date = str(t.due.date)[:10]
                    if due_date <= today_str:
                        today_tasks.append(t)

            today_tasks.sort(key=lambda t: -t.priority)
            for task in today_tasks[:15]:
                response_data["todoist_tasks"].append({
                    "content": task.content,
                    "priority": task.priority,
                    "due": task.due.string if task.due else None
                })
        except Exception as e:
            response_data["todoist_error"] = str(e)

        # Recent conversation actions
        if os.path.exists(RECENT_ACTIONS_FILE):
            try:
                with open(RECENT_ACTIONS_FILE, 'r') as f:
                    recent_actions = json.load(f)

                cutoff = (now - timedelta(hours=24)).isoformat()
                recent = [a for a in recent_actions if a.get("timestamp", "") > cutoff]

                for session in recent[-3:]:
                    extracted = session.get("extracted", {})
                    for item in extracted.get("action_items", []):
                        if item.get("owner") == "user":
                            response_data["conversation_actions"].append({
                                "task": item.get("task", ""),
                                "priority": item.get("priority", "medium"),
                                "deadline": item.get("deadline"),
                                "source": "conversation"
                            })
                    for c in extracted.get("commitments", []):
                        response_data["conversation_actions"].append({
                            "task": c,
                            "priority": "medium",
                            "source": "commitment"
                        })
            except:
                pass

        return JSONResponse({"success": True, "data": response_data})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def api_health(request: Request):
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "service": "dan-brain"})


async def api_get_last_conversation(request: Request):
    """Debug endpoint to check last conversation context."""
    if not os.path.exists(LAST_CONVERSATION_FILE):
        return JSONResponse({"exists": False, "message": "No last conversation file"})

    try:
        with open(LAST_CONVERSATION_FILE, 'r') as f:
            data = json.load(f)
        return JSONResponse({"exists": True, "data": data})
    except Exception as e:
        return JSONResponse({"exists": True, "error": str(e)})


# --- SERVER ENTRY POINT ---
if __name__ == "__main__":
    import uvicorn
    from starlette.routing import Route

    # Get the MCP SSE app
    mcp_app = mcp.sse_app()

    # Add REST API routes to the MCP app for iOS
    mcp_app.routes.insert(0, Route("/api/process-transcript", api_process_transcript, methods=["POST"]))
    mcp_app.routes.insert(0, Route("/api/rest-of-day", api_get_rest_of_day, methods=["GET"]))
    mcp_app.routes.insert(0, Route("/api/rest-of-day/structured", api_get_rest_of_day_structured, methods=["GET"]))
    mcp_app.routes.insert(0, Route("/api/health", api_health, methods=["GET"]))
    mcp_app.routes.insert(0, Route("/api/last-conversation", api_get_last_conversation, methods=["GET"]))

    # Run with uvicorn
    uvicorn.run(mcp_app, host="0.0.0.0", port=8000)
