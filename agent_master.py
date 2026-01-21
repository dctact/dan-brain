import os
import json
from datetime import datetime, timedelta
import pytz
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
# todoist_api_python removed - using local task storage instead
import anthropic
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
import pickle
import base64
import requests

# --- INITIALIZATION ---

# Restore Google token from environment variable if file doesn't exist
def restore_google_token():
    token_b64 = os.environ.get("GOOGLE_TOKEN_BASE64")
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    token_path = f"{data_dir}/google_token.pickle"

    print(f"[Startup] Token restore - data_dir: {data_dir}, token_path: {token_path}")

    if token_b64:
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(token_path, 'wb') as f:
                f.write(base64.b64decode(token_b64))
            print(f"[Startup] Restored Google token to {token_path}")
        except Exception as e:
            print(f"[Startup] Failed to restore Google token: {e}")

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

# Todoist API removed - using local task storage instead

# Initialize Anthropic client (fallback)
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# Initialize Gemini client (primary for large context)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Search API keys
CONTEXT7_API_KEY = os.environ.get("CONTEXT7_API_KEY", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")

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

# Last brainstorm session (separate from task-based conversations)
LAST_BRAINSTORM_FILE = f"{DATA_DIR}/last_brainstorm.json"

# Brainstorm Knowledge Base - persistent storage for insights, ideas, questions
KNOWLEDGE_BASE_FILE = f"{DATA_DIR}/brainstorm_knowledge_base.json"

# Persistent Tasks storage (replacing Todoist)
TASKS_FILE = f"{DATA_DIR}/tasks.json"

# Categories for organizing knowledge base items
INSIGHT_CATEGORIES = [
    "self",           # Personal insights about Dan (values, patterns, identity)
    "work",           # Professional/career insights
    "relationships",  # Interpersonal insights
    "creativity",     # Creative process, ideas, artistic
    "systems",        # How things work, mental models
    "goals",          # Aspirations, directions, priorities
    "patterns",       # Recurring themes, behaviors
    "uncategorized"   # Default
]

# --- AGENT TYPE CONFIGURATION ---
# Brainstorm agents: focus on ideas, exploration, reflection (no task creation)
# Task agents: focus on action items, commitments, accountability
BRAINSTORM_AGENTS = ["Nova"]  # Add new brainstorm agent names here
TASK_AGENTS = ["Zara"]  # Task-focused agent (Vic removed)

def is_brainstorm_agent(agent_name: str) -> bool:
    """Check if an agent is configured for brainstorm mode."""
    return agent_name in BRAINSTORM_AGENTS


# --- KNOWLEDGE BASE HELPERS ---

import uuid

def _load_knowledge_base() -> dict:
    """Load the knowledge base or create empty structure."""
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass

    return {
        "version": 1,
        "last_updated": datetime.now(pytz.timezone("America/New_York")).isoformat(),
        "insights": [],
        "ideas": [],
        "questions": [],
        "connections": [],
        "sessions": []
    }


def _save_knowledge_base(kb: dict) -> None:
    """Save the knowledge base to disk."""
    kb["last_updated"] = datetime.now(pytz.timezone("America/New_York")).isoformat()
    os.makedirs(os.path.dirname(KNOWLEDGE_BASE_FILE), exist_ok=True)
    with open(KNOWLEDGE_BASE_FILE, 'w') as f:
        json.dump(kb, f, indent=2)


def _check_semantic_duplicate(new_item: str, existing_items: list, item_type: str = "insight") -> dict:
    """
    Check if a new item is semantically similar to existing items.
    Uses Gemini for semantic comparison.

    Returns: {"is_duplicate": bool, "similar_id": str or None, "similarity": float}
    """
    if not existing_items or not new_item:
        return {"is_duplicate": False, "similar_id": None, "similarity": 0.0}

    # Limit comparison to most recent 30 items to avoid context overflow
    items_to_check = existing_items[-30:]
    items_text = "\n".join([f"[{item['id']}] {item['content']}" for item in items_to_check])

    prompt = f"""Compare this NEW {item_type} to the EXISTING {item_type}s below.

NEW {item_type.upper()}:
{new_item}

EXISTING {item_type.upper()}S:
{items_text}

Return JSON only (no markdown):
{{
    "is_duplicate": true if NEW means essentially the same as any EXISTING item,
    "most_similar_id": "id of most similar existing item" or null,
    "similarity_score": 0.0 to 1.0 (1.0 = identical meaning)
}}

A score >= 0.8 means duplicate. Be conservative - similar themes aren't duplicates unless same core meaning."""

    try:
        response = gemini_model.generate_content(prompt)
        result_text = response.text

        import re
        json_match = re.search(r'\{[\s\S]*?\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "is_duplicate": result.get("is_duplicate", False),
                "similar_id": result.get("most_similar_id"),
                "similarity": result.get("similarity_score", 0.0)
            }
    except Exception as e:
        print(f"[KnowledgeBase] Dedup error: {e}")

    return {"is_duplicate": False, "similar_id": None, "similarity": 0.0}


def _categorize_insight(content: str) -> str:
    """Use Gemini to categorize an insight into one of INSIGHT_CATEGORIES."""
    prompt = f"""Categorize this insight into ONE of these categories:
- self (personal insights about identity, values, patterns)
- work (professional/career)
- relationships (interpersonal)
- creativity (creative process, artistic)
- systems (how things work, mental models)
- goals (aspirations, priorities)
- patterns (recurring themes, behaviors)
- uncategorized (if unclear)

INSIGHT: {content}

Return only the category name, nothing else."""

    try:
        response = gemini_model.generate_content(prompt)
        category = response.text.strip().lower()
        if category in INSIGHT_CATEGORIES:
            return category
    except:
        pass

    return "uncategorized"


def add_to_knowledge_base(
    item_type: str,
    content: str,
    session_id: str = None,
    category: str = None,
    starred: bool = False,
    status: str = None,
    notes: str = None
) -> dict:
    """
    Add an insight, idea, or question to the knowledge base.
    Checks for duplicates and merges if similar.

    Args:
        item_type: "insight" | "idea" | "question" | "connection"
        content: The text content
        session_id: Which brainstorm session this came from
        category: One of INSIGHT_CATEGORIES (auto-detected if not provided)
        starred: Mark as important
        status: For ideas: "explored" | "parked" | "needs_more" | "implemented"
        notes: Additional context (for ideas)

    Returns: {"action": "added" | "merged" | "skipped", "id": str, "message": str}
    """
    kb = _load_knowledge_base()
    tz = pytz.timezone("America/New_York")
    timestamp = datetime.now(tz).isoformat()

    # Determine which list to use
    if item_type == "insight":
        items_list = kb["insights"]
    elif item_type == "idea":
        items_list = kb["ideas"]
    elif item_type == "question":
        items_list = kb["questions"]
    elif item_type == "connection":
        items_list = kb["connections"]
    else:
        return {"action": "skipped", "id": None, "message": f"Unknown item type: {item_type}"}

    # Check for duplicates
    dedup_result = _check_semantic_duplicate(content, items_list, item_type)

    if dedup_result["is_duplicate"] and dedup_result["similar_id"]:
        # Merge: add session to existing item, possibly upgrade starred
        for item in items_list:
            if item["id"] == dedup_result["similar_id"]:
                if session_id and session_id not in item.get("source_sessions", []):
                    item.setdefault("source_sessions", []).append(session_id)
                if starred:
                    item["starred"] = True
                item["updated_at"] = timestamp
                item["confidence"] = max(item.get("confidence", 0.5), dedup_result["similarity"])
                _save_knowledge_base(kb)
                return {
                    "action": "merged",
                    "id": item["id"],
                    "message": f"Merged with existing {item_type} (similarity: {dedup_result['similarity']:.2f})"
                }

    # Auto-categorize if not provided (for insights, ideas, questions)
    if not category and item_type in ["insight", "idea", "question"]:
        category = _categorize_insight(content)

    # Create new item
    new_id = str(uuid.uuid4())[:8]
    new_item = {
        "id": new_id,
        "content": content,
        "category": category or "uncategorized",
        "starred": starred,
        "source_sessions": [session_id] if session_id else [],
        "created_at": timestamp,
        "updated_at": timestamp,
        "confidence": 1.0
    }

    # Add type-specific fields
    if item_type == "idea":
        new_item["status"] = status or "explored"
        new_item["notes"] = notes
    elif item_type == "question":
        new_item["status"] = "open"
        new_item["answer"] = None
    elif item_type == "connection":
        new_item["links"] = []

    items_list.append(new_item)
    _save_knowledge_base(kb)

    return {
        "action": "added",
        "id": new_id,
        "message": f"Added new {item_type} [{category}]"
    }


def get_knowledge_base(
    category: str = None,
    starred_only: bool = False,
    item_type: str = None
) -> dict:
    """
    Retrieve items from the knowledge base.

    Args:
        category: Filter by category
        starred_only: Only return starred items
        item_type: Filter by type ("insight", "idea", "question", "connection")

    Returns: Full knowledge base structure with optional filtering
    """
    kb = _load_knowledge_base()

    def filter_items(items):
        result = items
        if category:
            result = [i for i in result if i.get("category") == category]
        if starred_only:
            result = [i for i in result if i.get("starred")]
        return result

    if item_type:
        if item_type == "insight":
            return {"insights": filter_items(kb["insights"])}
        elif item_type == "idea":
            return {"ideas": filter_items(kb["ideas"])}
        elif item_type == "question":
            return {"questions": filter_items(kb["questions"])}
        elif item_type == "connection":
            return {"connections": filter_items(kb["connections"])}

    return {
        "version": kb["version"],
        "last_updated": kb["last_updated"],
        "insights": filter_items(kb["insights"]),
        "ideas": filter_items(kb["ideas"]),
        "questions": filter_items(kb["questions"]),
        "connections": filter_items(kb["connections"]),
        "sessions": kb["sessions"]
    }


def star_knowledge_item(item_id: str, item_type: str, starred: bool = True) -> dict:
    """Toggle starred status on a knowledge base item."""
    kb = _load_knowledge_base()

    if item_type == "insight":
        items_list = kb["insights"]
    elif item_type == "idea":
        items_list = kb["ideas"]
    elif item_type == "question":
        items_list = kb["questions"]
    else:
        return {"success": False, "message": f"Unknown item type: {item_type}"}

    for item in items_list:
        if item["id"] == item_id:
            item["starred"] = starred
            item["updated_at"] = datetime.now(pytz.timezone("America/New_York")).isoformat()
            _save_knowledge_base(kb)
            return {"success": True, "message": f"Updated starred status to {starred}"}

    return {"success": False, "message": f"Item not found: {item_id}"}


# --- TASK STORAGE HELPERS ---

def _load_tasks() -> list:
    """Load tasks from persistent storage or return empty list."""
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


def _save_tasks(tasks: list) -> None:
    """Save tasks to persistent storage."""
    os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
    with open(TASKS_FILE, 'w') as f:
        json.dump(tasks, f, indent=2)


def _create_task(
    title: str,
    priority: int = 3,
    due_date: str = None,
    notes: str = None,
    source_session: str = None
) -> dict:
    """Create a new task and save to storage. Returns the created task."""
    tasks = _load_tasks()
    tz = pytz.timezone("America/New_York")
    timestamp = datetime.now(tz).isoformat()

    new_task = {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "priority": priority,  # 1=urgent, 2=high, 3=medium, 4=low
        "due_date": due_date,
        "notes": notes,
        "is_completed": False,
        "source_session": source_session,
        "created_at": timestamp,
        "completed_at": None
    }

    tasks.append(new_task)
    _save_tasks(tasks)
    return new_task


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


# --- SEARCH TOOLS ---

@mcp.tool()
def search_documentation(query: str, library: str = None) -> str:
    """
    Search official documentation for libraries, frameworks, and APIs.
    USE THIS for implementation details - it returns REAL docs, not summaries.

    WHEN TO CALL THIS:
    - Dan asks HOW to do something with a specific tool/library
    - Dan needs code examples, API references, or setup instructions
    - Dan is debugging or troubleshooting a specific library
    - Dan mentions a library name and wants to know how it works

    DO NOT USE when Dan is still CHOOSING or wants OPINIONS - use search_web instead.

    Args:
        query: What you're trying to learn (e.g., "create agent with knowledge base")
        library: Specific library name (e.g., "elevenlabs", "nextjs", "react") - PROVIDE THIS when you know it!

    Examples:
        - "How do I create an agent?" + library="elevenlabs" -> ElevenLabs agent docs
        - "useEffect cleanup" + library="react" -> React hooks documentation
        - "set up OAuth" + library="nextauth" -> NextAuth configuration docs
        - Dan says "I want to use ElevenLabs agents" -> search_documentation("agents", "elevenlabs")
    """
    if not CONTEXT7_API_KEY:
        return "Error: CONTEXT7_API_KEY not configured"

    headers = {"Authorization": f"Bearer {CONTEXT7_API_KEY}"}
    base_url = "https://context7.com/api/v2"

    try:
        # Step 1: Search for the library if provided, otherwise try to extract from query
        search_library = library or query.split()[0]  # Fallback: use first word of query

        search_response = requests.get(
            f"{base_url}/libs/search",
            headers=headers,
            params={"libraryName": search_library, "query": query},
            timeout=10
        )

        if search_response.status_code != 200:
            return f"Error searching for library: {search_response.status_code} - {search_response.text}"

        response_data = search_response.json()
        libraries = response_data.get("results", []) if isinstance(response_data, dict) else response_data

        if not libraries:
            return f"No documentation found for '{search_library}'. Try search_web for general information."

        # Use the best match
        best_match = libraries[0]
        library_id = best_match.get("id")

        # Step 2: Get documentation context
        context_response = requests.get(
            f"{base_url}/context",
            headers=headers,
            params={"libraryId": library_id, "query": query, "type": "txt"},
            timeout=15
        )

        if context_response.status_code != 200:
            return f"Error fetching documentation: {context_response.status_code}"

        docs = context_response.text

        if not docs or len(docs.strip()) < 50:
            return f"Limited documentation found for '{query}' in {best_match.get('name', search_library)}. Try rephrasing or use search_web."

        # Format response
        return f"""## Documentation: {best_match.get('name', search_library)}

{docs}

---
Source: Context7 ({library_id})"""

    except requests.Timeout:
        return "Documentation search timed out. Please try again."
    except Exception as e:
        return f"Error searching documentation: {str(e)}"


@mcp.tool()
def search_web(query: str) -> str:
    """
    Search the web for opinions, reviews, comparisons, and current information.
    Returns SYNTHESIZED answers with citations - not just links. USE THIS LIBERALLY.

    WHEN TO CALL THIS (prefer this when uncertain):
    - Dan asks if something is GOOD or worth using -> reviews, real user opinions
    - Dan is CHOOSING between tools/approaches -> comparisons, recommendations
    - Dan wants CURRENT info -> news, recent releases, pricing, updates
    - Dan asks about ANYTHING that isn't specific library documentation
    - Dan mentions a tool without knowing if he wants to use it yet

    THIS IS YOUR DEFAULT for research. When in doubt between search_documentation
    and search_web, USE THIS ONE - it's better to get opinions than dump API docs.

    Args:
        query: Be specific! (e.g., "Is Tavily good for AI search?" not just "Tavily")

    Examples:
        - "Is Tavily any good?" -> Real opinions from Reddit, HN, developers
        - "Best chess opening against d4" -> Analysis with sources
        - "NextAuth vs Clerk for authentication" -> Comparison with recommendation
        - "What's new in React 19" -> Current information with citations
        - Dan says "I'm thinking about using Supabase" -> search_web("Is Supabase good? Real user experiences")
    """
    if not PERPLEXITY_API_KEY:
        return "Error: PERPLEXITY_API_KEY not configured"

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Provide clear, synthesized answers based on current web information. Include specific details, real user opinions when available, and cite your sources."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            timeout=30
        )

        if response.status_code != 200:
            return f"Error from Perplexity: {response.status_code} - {response.text}"

        result = response.json()

        # Extract the answer
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not answer:
            return "No results found. Try rephrasing your query."

        # Extract citations if available
        citations = result.get("citations", [])

        formatted_response = f"""## Web Search Results

{answer}"""

        if citations:
            formatted_response += "\n\n---\n### Sources\n"
            for i, citation in enumerate(citations, 1):
                formatted_response += f"{i}. {citation}\n"

        return formatted_response

    except requests.Timeout:
        return "Web search timed out. Please try again."
    except Exception as e:
        return f"Error searching web: {str(e)}"


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


def _format_memory_age(created_at: str) -> str:
    """Format how long ago a memory was created."""
    if not created_at:
        return "(old - no date)"

    try:
        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz)
        created = datetime.fromisoformat(created_at)

        # Make sure created is timezone-aware
        if created.tzinfo is None:
            created = tz.localize(created)

        delta = now - created
        days = delta.days
        hours = delta.seconds // 3600

        if days == 0:
            if hours == 0:
                return "(just now)"
            elif hours == 1:
                return "(1 hour ago)"
            else:
                return f"({hours} hours ago)"
        elif days == 1:
            return "(yesterday)"
        elif days < 7:
            return f"({days} days ago)"
        elif days < 14:
            return "(1 week ago)"
        elif days < 30:
            weeks = days // 7
            return f"({weeks} weeks ago)"
        elif days < 60:
            return "(1 month ago)"
        elif days < 365:
            months = days // 30
            return f"({months} months ago)"
        else:
            years = days // 365
            return f"({years} year{'s' if years > 1 else ''} ago)"
    except:
        return ""


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

    # Add timestamp for context aging
    tz = pytz.timezone("America/New_York")
    timestamp = datetime.now(tz).isoformat()

    new_fact = {
        "s": subject,
        "r": relation,
        "o": object_entity,
        "category": category,
        "created_at": timestamp
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
        age_tag = _format_memory_age(item.get("created_at", ""))
        results.append(f"- {cat_tag} {item['s']} {item['r']} {item['o']} {age_tag}".strip())

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

    # Organize by category with indices and age
    by_category = {}
    for idx, item in enumerate(graph):
        cat = item.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        age_tag = _format_memory_age(item.get("created_at", ""))
        by_category[cat].append(f"  [{idx}] {item['s']} {item['r']} {item['o']} {age_tag}".strip())

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

# --- TOOL 3: TASKS (Local Storage - replaces Todoist) ---

@mcp.tool()
def add_task(title: str, priority: int = 3, due_date: str = None, notes: str = None) -> str:
    """
    Creates a new task. Use this to capture action items!

    WHEN TO CALL THIS:
    - Dan commits to doing something -> capture it immediately
    - Dan mentions a deadline or appointment to remember
    - You identify an action item from the conversation
    - Dan asks you to remind him about something

    Args:
        title: Clear, actionable task description (start with verb)
        priority: 1 (Urgent/red), 2 (High/orange), 3 (Medium/default), 4 (Low/grey)
        due_date: ISO date string like "2024-01-15" or natural language like "tomorrow"
        notes: Optional additional context

    Examples:
        add_task("Call dentist to schedule cleaning", priority=2, due_date="tomorrow")
        add_task("Review project proposal", priority=1, due_date="2024-01-20")
        add_task("Buy birthday gift for Liz", priority=2)

    Pro tip: Be specific. "Work on project" is bad. "Draft intro section of proposal" is good.
    """
    try:
        task = _create_task(title=title, priority=priority, due_date=due_date, notes=notes)
        due_info = due_date if due_date else "No due date"
        priority_labels = {1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}
        return f"Task Added: '{task['title']}' (Due: {due_info}, Priority: {priority_labels.get(priority, 'Medium')})"
    except Exception as e:
        return f"Error adding task: {str(e)}"


@mcp.tool()
def get_tasks(include_completed: bool = False) -> str:
    """
    Lists Dan's current tasks. Use this for context and accountability!

    WHEN TO CALL THIS:
    - At conversation start to understand what Dan should be working on
    - When Dan seems unfocused - remind him what's on his plate
    - Before suggesting new tasks - check what's already there
    - When Dan asks "what should I do?" or "what's on my list?"

    Args:
        include_completed: If True, includes completed tasks. Default False.

    Returns tasks with priority markers and due dates. Use this info to:
    - Hold Dan accountable to existing commitments
    - Avoid adding duplicate tasks
    - Help prioritize what to tackle next
    """
    try:
        all_tasks = _load_tasks()

        if not include_completed:
            all_tasks = [t for t in all_tasks if not t.get("is_completed", False)]

        if not all_tasks:
            return "No tasks found."

        # Sort by priority (1=urgent first) then by due date
        all_tasks.sort(key=lambda t: (t.get("priority", 3), t.get("due_date") or "9999"))

        output = []
        priority_markers = {1: "🔴", 2: "🟠", 3: "🔵", 4: "⚪"}
        for t in all_tasks[:50]:
            due_str = t.get("due_date") or "No due date"
            marker = priority_markers.get(t.get("priority", 3), "")
            completed = "✓ " if t.get("is_completed") else ""
            output.append(f"- {completed}{marker} [{t['id']}] {t['title']} (Due: {due_str})")

        return f"Found {len(all_tasks)} tasks:\n" + "\n".join(output)
    except Exception as e:
        return f"Error fetching tasks: {str(e)}"


@mcp.tool()
def complete_task(task_id: str) -> str:
    """
    Mark a task as completed.

    Args:
        task_id: The task ID (shown in brackets when listing tasks)

    Use this when Dan says he's done with something or you confirm completion.
    """
    try:
        tasks = _load_tasks()
        tz = pytz.timezone("America/New_York")

        for task in tasks:
            if task["id"] == task_id:
                task["is_completed"] = True
                task["completed_at"] = datetime.now(tz).isoformat()
                _save_tasks(tasks)
                return f"Task completed: '{task['title']}'"

        return f"Task not found: {task_id}"
    except Exception as e:
        return f"Error completing task: {str(e)}"


@mcp.tool()
def delete_task(task_id: str) -> str:
    """
    Permanently delete a task.

    Args:
        task_id: The task ID (shown in brackets when listing tasks)

    Use when a task is no longer relevant (not just completed).
    """
    try:
        tasks = _load_tasks()

        for i, task in enumerate(tasks):
            if task["id"] == task_id:
                deleted_title = task["title"]
                del tasks[i]
                _save_tasks(tasks)
                return f"Task deleted: '{deleted_title}'"

        return f"Task not found: {task_id}"
    except Exception as e:
        return f"Error deleting task: {str(e)}"


@mcp.tool()
def update_task(task_id: str, title: str = None, priority: int = None, due_date: str = None, notes: str = None) -> str:
    """
    Update an existing task.

    Args:
        task_id: The task ID (shown in brackets when listing tasks)
        title: New title (optional)
        priority: New priority 1-4 (optional)
        due_date: New due date (optional)
        notes: New notes (optional)

    Only provided fields will be updated.
    """
    try:
        tasks = _load_tasks()

        for task in tasks:
            if task["id"] == task_id:
                if title is not None:
                    task["title"] = title
                if priority is not None:
                    task["priority"] = priority
                if due_date is not None:
                    task["due_date"] = due_date
                if notes is not None:
                    task["notes"] = notes
                _save_tasks(tasks)
                return f"Task updated: '{task['title']}'"

        return f"Task not found: {task_id}"
    except Exception as e:
        return f"Error updating task: {str(e)}"

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

def _process_brainstorm_transcript(transcript: str, agent_name: str) -> dict:
    """
    Process a brainstorm conversation - focuses on ideas, insights, and exploration
    rather than action items. Returns structured data for iOS app.
    """
    prompt = f"""Analyze this brainstorming conversation between Dan and his AI thinking partner.
This is NOT a task-focused conversation - focus on IDEAS, INSIGHTS, and EXPLORATION.

TRANSCRIPT:
{transcript}

Extract and return as JSON:
{{
    "type": "brainstorm",
    "key_insights": [
        "Important realizations or 'aha moments' from the conversation"
    ],
    "ideas_explored": [
        {{
            "idea": "Description of an idea that was discussed",
            "status": "explored" | "parked" | "needs_more",
            "notes": "Any important context or considerations"
        }}
    ],
    "questions_raised": [
        "Open questions that emerged but weren't fully resolved"
    ],
    "new_facts_about_dan": [
        "New information Dan revealed about himself, his preferences, goals, or situation"
    ],
    "connections_made": [
        "Links between different ideas, past experiences, or concepts"
    ],
    "synthesis": "A 2-3 sentence summary capturing the essence of this brainstorm session"
}}

Guidelines:
- Focus on the THINKING, not tasks to do
- Capture the exploration journey, not just conclusions
- Note ideas that were "parked" for later vs fully explored
- Extract any new personal facts that could enrich Dan's knowledge graph
- Be thoughtful about the synthesis - what was the real value of this conversation?
- If the conversation was shallow or off-topic, be honest about that in synthesis"""

    response = gemini_model.generate_content(prompt)
    result_text = response.text

    # Extract JSON from response
    import re
    json_match = re.search(r'\{[\s\S]*\}', result_text)
    if json_match:
        extracted = json.loads(json_match.group())
        extracted["type"] = "brainstorm"  # Ensure type is set
    else:
        extracted = {
            "type": "brainstorm",
            "key_insights": [],
            "ideas_explored": [],
            "questions_raised": [],
            "new_facts_about_dan": [],
            "connections_made": [],
            "synthesis": "Could not process this conversation."
        }

    # --- AUTO-ADD TO KNOWLEDGE BASE ---
    session_id = str(uuid.uuid4())[:8]
    kb_stats = {"added": 0, "merged": 0}

    # Add insights
    for insight in extracted.get("key_insights", []):
        if insight and len(insight) > 10:
            result = add_to_knowledge_base("insight", insight, session_id=session_id)
            if result["action"] == "added":
                kb_stats["added"] += 1
            elif result["action"] == "merged":
                kb_stats["merged"] += 1

    # Add ideas
    for idea_obj in extracted.get("ideas_explored", []):
        idea_text = idea_obj.get("idea", "") if isinstance(idea_obj, dict) else str(idea_obj)
        status = idea_obj.get("status", "explored") if isinstance(idea_obj, dict) else "explored"
        notes = idea_obj.get("notes") if isinstance(idea_obj, dict) else None
        if idea_text and len(idea_text) > 10:
            result = add_to_knowledge_base(
                "idea", idea_text,
                session_id=session_id,
                status=status,
                notes=notes
            )
            if result["action"] == "added":
                kb_stats["added"] += 1
            elif result["action"] == "merged":
                kb_stats["merged"] += 1

    # Add questions
    for question in extracted.get("questions_raised", []):
        if question and len(question) > 10:
            result = add_to_knowledge_base("question", question, session_id=session_id)
            if result["action"] == "added":
                kb_stats["added"] += 1
            elif result["action"] == "merged":
                kb_stats["merged"] += 1

    # Add connections
    for connection in extracted.get("connections_made", []):
        if connection and len(connection) > 10:
            result = add_to_knowledge_base("connection", connection, session_id=session_id)
            if result["action"] == "added":
                kb_stats["added"] += 1
            elif result["action"] == "merged":
                kb_stats["merged"] += 1

    # Record session in knowledge base
    kb = _load_knowledge_base()
    kb["sessions"].append({
        "id": session_id,
        "timestamp": datetime.now(pytz.timezone("America/New_York")).isoformat(),
        "agent_name": agent_name,
        "synthesis": extracted.get("synthesis", ""),
        "insight_count": len(extracted.get("key_insights", [])),
        "idea_count": len(extracted.get("ideas_explored", []))
    })
    # Keep last 50 sessions
    kb["sessions"] = kb["sessions"][-50:]
    _save_knowledge_base(kb)

    print(f"[KnowledgeBase] Session {session_id}: {kb_stats['added']} added, {kb_stats['merged']} merged")
    extracted["_kb_session_id"] = session_id
    extracted["_kb_stats"] = kb_stats

    return extracted


def _process_task_transcript(transcript: str, agent_name: str) -> dict:
    """
    Process a task-focused conversation - extracts action items, commitments, etc.
    Returns structured data for iOS app. Auto-saves tasks to persistent storage.
    """
    prompt = f"""Analyze this voice conversation transcript and extract actionable information.

TRANSCRIPT:
{transcript}

Extract and return as JSON:
{{
    "type": "task",
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

    response = gemini_model.generate_content(prompt)
    result_text = response.text

    # Extract JSON from response
    import re
    json_match = re.search(r'\{[\s\S]*\}', result_text)
    if json_match:
        extracted = json.loads(json_match.group())
        extracted["type"] = "task"  # Ensure type is set
    else:
        extracted = {
            "type": "task",
            "action_items": [],
            "commitments": [],
            "key_decisions": [],
            "follow_ups": []
        }

    # --- AUTO-SAVE TASKS TO PERSISTENT STORAGE ---
    session_id = str(uuid.uuid4())[:8]
    tasks_added = 0
    priority_map = {"high": 1, "medium": 3, "low": 4}

    for item in extracted.get("action_items", []):
        # Only save tasks owned by the user
        if item.get("owner") == "user":
            task_text = item.get("task", "").strip()
            if task_text and len(task_text) > 5:
                priority_str = item.get("priority", "medium").lower()
                priority = priority_map.get(priority_str, 3)

                _create_task(
                    title=task_text,
                    priority=priority,
                    due_date=item.get("deadline"),
                    notes=item.get("source"),
                    source_session=session_id
                )
                tasks_added += 1

    # Also save commitments as tasks
    for commitment in extracted.get("commitments", []):
        if commitment and len(commitment.strip()) > 5:
            _create_task(
                title=commitment.strip(),
                priority=2,  # Commitments are high priority
                source_session=session_id
            )
            tasks_added += 1

    print(f"[Tasks] Session {session_id}: {tasks_added} tasks auto-saved")
    extracted["_tasks_session_id"] = session_id
    extracted["_tasks_added"] = tasks_added

    return extracted


@mcp.tool()
def process_transcript(transcript: str, agent_name: str = "Agent") -> str:
    """
    Process a voice conversation transcript. Automatically detects agent type
    and uses appropriate processing (task-focused vs brainstorm-focused).

    Args:
        transcript: The full conversation transcript text
        agent_name: Name of the agent in the conversation

    Returns extracted insights/action items and saves them for retrieval.
    """
    if not transcript or len(transcript.strip()) < 20:
        return "Transcript too short to process."

    try:
        # Branch based on agent type
        is_brainstorm = is_brainstorm_agent(agent_name)
        if is_brainstorm:
            extracted = _process_brainstorm_transcript(transcript, agent_name)
        else:
            extracted = _process_task_transcript(transcript, agent_name)

        # Save to recent actions file (both types go here for unified tracking)
        tz = pytz.timezone("America/New_York")
        timestamp = datetime.now(tz).isoformat()

        recent_actions = []
        if os.path.exists(RECENT_ACTIONS_FILE):
            with open(RECENT_ACTIONS_FILE, 'r') as f:
                try:
                    recent_actions = json.load(f)
                except:
                    recent_actions = []

        # Add new extraction with type indicator
        recent_actions.append({
            "timestamp": timestamp,
            "agent_name": agent_name,
            "type": extracted.get("type", "task"),
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
            "type": extracted.get("type", "task"),
            "transcript": transcript[:5000]  # Limit size
        })

        # Keep last 20 transcripts for pattern analysis
        transcript_archive = transcript_archive[-20:]

        with open(TRANSCRIPT_ARCHIVE_FILE, 'w') as f:
            json.dump(transcript_archive, f, indent=2)

        # Store summary based on conversation type
        if is_brainstorm:
            # Save brainstorm-specific summary
            with open(LAST_BRAINSTORM_FILE, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "type": "brainstorm",
                    "synthesis": extracted.get("synthesis", ""),
                    "key_insights": extracted.get("key_insights", [])[:5],
                    "ideas_explored": extracted.get("ideas_explored", [])[:5],
                    "questions_raised": extracted.get("questions_raised", [])[:5],
                    "new_facts_about_dan": extracted.get("new_facts_about_dan", [])[:5],
                    "connections_made": extracted.get("connections_made", [])[:5],
                    "transcript_preview": transcript[:1500]
                }, f, indent=2)
            # Also update last conversation for continuity
            with open(LAST_CONVERSATION_FILE, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "type": "brainstorm",
                    "summary": [extracted.get("synthesis", "Brainstorm session")],
                    "action_items": [],  # Brainstorms don't have action items
                    "commitments": [],
                    "transcript_preview": transcript[:1500]
                }, f, indent=2)
        else:
            # Task-focused summary (original behavior)
            with open(LAST_CONVERSATION_FILE, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "type": "task",
                    "summary": extracted.get("key_decisions", [])[:3],
                    "action_items": [item.get("task") for item in extracted.get("action_items", []) if item.get("owner") == "user"][:5],
                    "commitments": extracted.get("commitments", [])[:5],
                    "transcript_preview": transcript[:1500]
                }, f, indent=2)

        # Auto-trigger pattern detection every 5 transcripts (both types)
        if len(transcript_archive) >= 5 and len(transcript_archive) % 5 == 0:
            try:
                pattern_result = detect_patterns(min_transcripts=5)
                print(f"[Auto Pattern Detection] {pattern_result[:200]}...")
            except Exception as e:
                print(f"[Auto Pattern Detection] Error: {e}")

        # Format response based on type
        if is_brainstorm:
            output = [f"Processed brainstorm with {agent_name}:"]

            if extracted.get("synthesis"):
                output.append(f"\n📝 SYNTHESIS:\n  {extracted['synthesis']}")

            if extracted.get("key_insights"):
                output.append("\n💡 KEY INSIGHTS:")
                for insight in extracted["key_insights"]:
                    output.append(f"  • {insight}")

            if extracted.get("ideas_explored"):
                output.append("\n🎯 IDEAS EXPLORED:")
                for idea in extracted["ideas_explored"]:
                    status_icon = {"explored": "✓", "parked": "⏸", "needs_more": "…"}.get(idea.get("status", ""), "")
                    output.append(f"  {status_icon} {idea.get('idea', '')}")

            if extracted.get("questions_raised"):
                output.append("\n❓ OPEN QUESTIONS:")
                for q in extracted["questions_raised"]:
                    output.append(f"  • {q}")

            if extracted.get("new_facts_about_dan"):
                output.append("\n🧠 NEW FACTS (potential memories):")
                for fact in extracted["new_facts_about_dan"]:
                    output.append(f"  • {fact}")
        else:
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
                for f_item in extracted["follow_ups"]:
                    output.append(f"  • {f_item}")

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
    Analyze conversation history to discover what ACTUALLY works with Dan.
    Extracts psychological and behavioral patterns the agent can USE.

    Args:
        min_transcripts: Minimum conversations needed before analysis (default 5)

    PURPOSE: Turn conversation data into actionable agent intelligence.
    The patterns found here should directly inform HOW the agent talks to Dan,
    WHEN to push vs back off, and WHAT approaches convert talk into action.

    Patterns are saved to memory and automatically inform future conversations.
    Only patterns observed 2+ times are saved — no speculation.
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
        prompt = f"""Analyze these conversations to find what works with Dan. Be CONSERVATIVE.

TRANSCRIPTS:
{combined}

EXISTING PATTERNS (check for overlap — update these rather than duplicate):
{chr(10).join(existing_patterns[:30]) if existing_patterns else "None yet"}

Find 1-3 HIGH-SIGNAL patterns only. Categories: motivation, resistance, communication, action, energy.

Return as JSON:
{{
    "new_patterns": [
        {{
            "type": "motivation|resistance|communication|action|energy",
            "observation": "specific pattern",
            "evidence_count": number of convos showing this,
            "agent_instruction": "what agent should DO"
        }}
    ],
    "updates_to_existing": [
        {{
            "existing_pattern": "quote the existing pattern to update",
            "update": "stronger/refined version based on new evidence"
        }}
    ],
    "skip": ["patterns noticed but not enough evidence yet"]
}}

RULES:
- MAX 3 new patterns — only genuinely new insights
- If similar to existing pattern, put in updates_to_existing instead
- evidence_count must be >= 3 (not 2)
- No vague stuff like "prefers directness" — be specific and actionable"""

        # Use Gemini for large context pattern analysis
        response = gemini_model.generate_content(prompt)
        result_text = response.text

        # Parse response
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if not json_match:
            return "Could not parse pattern analysis results."

        analysis = json.loads(json_match.group())
        new_patterns = analysis.get("new_patterns", [])
        updates = analysis.get("updates_to_existing", [])
        skipped = analysis.get("skip", [])

        # Save new patterns (max 3, evidence >= 3)
        saved_count = 0
        for pattern in new_patterns[:3]:
            if pattern.get("evidence_count", 0) >= 3:
                observation = pattern.get("observation", "")
                instruction = pattern.get("agent_instruction", "")
                pattern_type = pattern.get("type", "other")
                
                if observation:
                    relation = f"pattern ({pattern_type})"
                    obj = f"{observation} → AGENT: {instruction}" if instruction else observation
                    
                    result = add_memory("Dan", relation, obj, "preferences")
                    if "Stored" in result:
                        saved_count += 1

        # Format output
        output = [f"Pattern Analysis ({len(archive)} conversations):"]

        if new_patterns:
            output.append(f"\nNEW PATTERNS ({len(new_patterns)} found, {saved_count} saved):")
            for p in new_patterns[:3]:
                output.append(f"  • [{p.get('type', '?')}] {p.get('observation', '?')}")

        if updates:
            output.append(f"\nUPDATES TO EXISTING ({len(updates)}):")
            for u in updates:
                output.append(f"  ↻ {u.get('update', '?')}")

        if skipped:
            output.append(f"\nNEED MORE DATA: {len(skipped)} patterns pending")

        if not new_patterns and not updates:
            output.append("\nNo new patterns. Things are working well.")

        return "\n".join(output)

    except Exception as e:
        return f"Error analyzing patterns: {str(e)}"


# --- TOOL 7: REST OF DAY (Unified Action View) ---

@mcp.tool()
def get_rest_of_day() -> str:
    """
    THE BIG PICTURE: Get everything Dan needs to do, all in one place.
    Combines: Calendar events + Tasks + Recent conversation action items

    WHEN TO CALL THIS:
    - At the START of conversations for full context
    - When Dan asks "what should I focus on?" or "what's my day look like?"
    - When helping Dan prioritize or plan
    - After adding tasks to show the updated picture

    This is your PRIMARY tool for understanding Dan's obligations. It shows:
    - 📅 SCHEDULED: Calendar events for the next 18 hours
    - ✅ TASKS: Active tasks sorted by priority
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

    # --- TASKS ---
    output.append("\n✅ TASKS:")
    try:
        all_tasks = _load_tasks()

        # Filter for incomplete tasks, optionally due today or overdue
        today_str = now.strftime("%Y-%m-%d")
        active_tasks = [t for t in all_tasks if not t.get("is_completed", False)]

        # Sort by priority (1=urgent first)
        active_tasks.sort(key=lambda t: t.get("priority", 3))

        if active_tasks:
            priority_markers = {1: "🔴", 2: "🟠", 3: "🔵", 4: "⚪"}
            for task in active_tasks[:15]:
                marker = priority_markers.get(task.get("priority", 3), "")
                due_info = f" (Due: {task['due_date']})" if task.get("due_date") else ""
                output.append(f"  {marker} {task['title']}{due_info}")
        else:
            output.append("  No active tasks")
    except Exception as e:
        output.append(f"  (Tasks error: {str(e)})")

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

        # Process the transcript (saves to files)
        result = process_transcript(transcript, agent_name)

        # Return structured data based on agent type
        is_brainstorm = is_brainstorm_agent(agent_name)

        if is_brainstorm and os.path.exists(LAST_BRAINSTORM_FILE):
            with open(LAST_BRAINSTORM_FILE, 'r') as f:
                structured_data = json.load(f)
            return JSONResponse({
                "success": True,
                "result": result,
                "type": "brainstorm",
                "data": structured_data
            })
        elif os.path.exists(LAST_CONVERSATION_FILE):
            with open(LAST_CONVERSATION_FILE, 'r') as f:
                structured_data = json.load(f)
            return JSONResponse({
                "success": True,
                "result": result,
                "type": structured_data.get("type", "task"),
                "data": structured_data
            })
        else:
            return JSONResponse({
                "success": True,
                "result": result,
                "type": "brainstorm" if is_brainstorm else "task"
            })
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

        # Local tasks (replaces Todoist)
        try:
            all_tasks = _load_tasks()
            active_tasks = [t for t in all_tasks if not t.get("is_completed", False)]

            # Sort by priority (1=urgent first)
            active_tasks.sort(key=lambda t: t.get("priority", 3))

            for task in active_tasks[:15]:
                response_data["todoist_tasks"].append({
                    "content": task.get("title", ""),
                    "priority": task.get("priority", 3),
                    "due": task.get("due_date"),
                    "id": task.get("id"),
                    "notes": task.get("notes")
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


async def api_get_last_brainstorm(request: Request):
    """Get the last brainstorm session data for iOS app display."""
    if not os.path.exists(LAST_BRAINSTORM_FILE):
        return JSONResponse({
            "success": False,
            "exists": False,
            "message": "No brainstorm session found"
        })

    try:
        with open(LAST_BRAINSTORM_FILE, 'r') as f:
            data = json.load(f)
        return JSONResponse({
            "success": True,
            "exists": True,
            "data": data
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "exists": True,
            "error": str(e)
        }, status_code=500)


async def api_get_knowledge_base(request: Request):
    """GET /api/knowledge-base - Returns full knowledge base for iOS."""
    try:
        # Parse query params for filtering
        category = request.query_params.get("category")
        starred_only = request.query_params.get("starred_only", "").lower() == "true"
        item_type = request.query_params.get("item_type")

        kb = get_knowledge_base(
            category=category,
            starred_only=starred_only,
            item_type=item_type
        )

        return JSONResponse({
            "success": True,
            "data": kb
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def api_star_knowledge_item(request: Request):
    """POST /api/knowledge-base/star - Toggle starred status on an item."""
    try:
        body = await request.json()
        item_id = body.get("item_id")
        item_type = body.get("item_type")
        starred = body.get("starred", True)

        if not item_id or not item_type:
            return JSONResponse({
                "success": False,
                "error": "Missing item_id or item_type"
            }, status_code=400)

        result = star_knowledge_item(item_id, item_type, starred)

        return JSONResponse({
            "success": result["success"],
            "message": result["message"]
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# --- TASK REST API ENDPOINTS ---

async def api_get_tasks(request: Request):
    """GET /api/tasks - Returns all tasks for iOS."""
    try:
        include_completed = request.query_params.get("include_completed", "").lower() == "true"
        tasks = _load_tasks()

        if not include_completed:
            tasks = [t for t in tasks if not t.get("is_completed", False)]

        # Sort by priority then due date
        tasks.sort(key=lambda t: (t.get("priority", 3), t.get("due_date") or "9999"))

        return JSONResponse({
            "success": True,
            "data": tasks
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def api_create_task(request: Request):
    """POST /api/tasks - Create a new task."""
    try:
        body = await request.json()
        title = body.get("title")

        if not title:
            return JSONResponse({
                "success": False,
                "error": "Missing title"
            }, status_code=400)

        task = _create_task(
            title=title,
            priority=body.get("priority", 3),
            due_date=body.get("due_date"),
            notes=body.get("notes"),
            source_session=body.get("source_session")
        )

        return JSONResponse({
            "success": True,
            "data": task
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def api_update_task(request: Request):
    """PUT /api/tasks/{id} - Update an existing task."""
    try:
        task_id = request.path_params.get("task_id")
        body = await request.json()

        tasks = _load_tasks()
        task_found = None

        for task in tasks:
            if task["id"] == task_id:
                task_found = task
                if "title" in body:
                    task["title"] = body["title"]
                if "priority" in body:
                    task["priority"] = body["priority"]
                if "due_date" in body:
                    task["due_date"] = body["due_date"]
                if "notes" in body:
                    task["notes"] = body["notes"]
                if "is_completed" in body:
                    task["is_completed"] = body["is_completed"]
                    if body["is_completed"] and not task.get("completed_at"):
                        task["completed_at"] = datetime.now(pytz.timezone("America/New_York")).isoformat()
                break

        if not task_found:
            return JSONResponse({
                "success": False,
                "error": f"Task not found: {task_id}"
            }, status_code=404)

        _save_tasks(tasks)

        return JSONResponse({
            "success": True,
            "data": task_found
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def api_delete_task(request: Request):
    """DELETE /api/tasks/{id} - Delete a task."""
    try:
        task_id = request.path_params.get("task_id")
        tasks = _load_tasks()

        for i, task in enumerate(tasks):
            if task["id"] == task_id:
                deleted_task = tasks.pop(i)
                _save_tasks(tasks)
                return JSONResponse({
                    "success": True,
                    "message": f"Deleted: {deleted_task['title']}"
                })

        return JSONResponse({
            "success": False,
            "error": f"Task not found: {task_id}"
        }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


async def api_complete_task(request: Request):
    """POST /api/tasks/{id}/complete - Toggle task completion."""
    try:
        task_id = request.path_params.get("task_id")
        tasks = _load_tasks()
        tz = pytz.timezone("America/New_York")

        for task in tasks:
            if task["id"] == task_id:
                # Toggle completion
                task["is_completed"] = not task.get("is_completed", False)
                if task["is_completed"]:
                    task["completed_at"] = datetime.now(tz).isoformat()
                else:
                    task["completed_at"] = None
                _save_tasks(tasks)
                return JSONResponse({
                    "success": True,
                    "data": task
                })

        return JSONResponse({
            "success": False,
            "error": f"Task not found: {task_id}"
        }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


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
    mcp_app.routes.insert(0, Route("/api/last-brainstorm", api_get_last_brainstorm, methods=["GET"]))
    mcp_app.routes.insert(0, Route("/api/knowledge-base", api_get_knowledge_base, methods=["GET"]))
    mcp_app.routes.insert(0, Route("/api/knowledge-base/star", api_star_knowledge_item, methods=["POST"]))

    # Task API routes
    mcp_app.routes.insert(0, Route("/api/tasks", api_get_tasks, methods=["GET"]))
    mcp_app.routes.insert(0, Route("/api/tasks", api_create_task, methods=["POST"]))
    mcp_app.routes.insert(0, Route("/api/tasks/{task_id}", api_update_task, methods=["PUT"]))
    mcp_app.routes.insert(0, Route("/api/tasks/{task_id}", api_delete_task, methods=["DELETE"]))
    mcp_app.routes.insert(0, Route("/api/tasks/{task_id}/complete", api_complete_task, methods=["POST"]))

    # Run with uvicorn
    uvicorn.run(mcp_app, host="0.0.0.0", port=8000)
