# Search Tools for Dan's Brain

**Created:** 2026-01-21
**Status:** Ready for implementation

## Problem

Current search via Exa (through ElevenLabs) is unreliable:
- Can't fetch full documentation (failed on ElevenLabs own docs)
- Only returns marketing materials for tool comparisons
- Can't synthesize opinions or find real user reviews

## Solution

Add two specialized search tools that play to different strengths:

| Tool | Backend | Use Case |
|------|---------|----------|
| `search_documentation` | Context7 | Library/framework docs, code examples, API references |
| `search_web` | Perplexity Sonar | Reviews, opinions, comparisons, current events, general research |

## Tool 1: `search_documentation`

### Purpose
Fetch library and framework documentation with code examples.

### API
```python
@mcp.tool()
def search_documentation(query: str, library: str = None) -> str:
    """
    Search documentation for libraries, frameworks, and APIs.

    Args:
        query: What you're trying to learn or do
        library: Optional - specific library name (e.g., "elevenlabs", "nextjs")

    Returns:
        Relevant documentation with code examples
    """
```

### Implementation
1. If `library` provided, resolve to Context7 library ID
2. If not provided, extract library name from query or search broadly
3. Call Context7 `query-docs` endpoint
4. Return formatted documentation

### Examples
- "How do I create an agent with a knowledge base?" + library="elevenlabs" → ElevenLabs agent docs
- "NextAuth session handling" → NextAuth documentation
- "Stockfish API moves" → Stockfish docs

---

## Tool 2: `search_web`

### Purpose
Search the web for opinions, reviews, comparisons, and current information. Returns synthesized answers with citations.

### API
```python
@mcp.tool()
def search_web(query: str) -> str:
    """
    Search the web for reviews, opinions, comparisons, and current information.
    Returns a synthesized answer with citations.

    Args:
        query: What you want to know

    Returns:
        Synthesized answer with source citations
    """
```

### Implementation
1. Call Perplexity Sonar API with query
2. Return synthesized response with citations
3. Include source URLs for reference

### Examples
- "Is Tavily any good for AI search?" → Synthesized review with Reddit/HN opinions
- "Best chess opening against d4 for aggressive players" → Analysis with sources
- "What's new in React 19?" → Current information with citations

---

## System Prompt Additions

Add to the agent's system prompt for proper tool routing:

```
SEARCH TOOL SELECTION:

Use `search_documentation` when the user needs:
- How to USE something they've already chosen (setup, configuration, API calls, code examples)
- Error messages or troubleshooting for a specific library
- Syntax, parameters, or method signatures
- Official documentation or API references

Use `search_web` when the user needs:
- Whether something is GOOD (reviews, opinions, user experiences)
- What tool/library to CHOOSE (comparisons, alternatives, recommendations)
- Current information (news, recent releases, pricing, updates)
- General research on any topic
- Real-world experiences from developers

When ambiguous, prefer `search_web` - it's better to get opinions about a library
than to dump API docs when the user is still deciding.

Examples:
- "How do I set up authentication in NextAuth?" → search_documentation
- "Is NextAuth any good?" → search_web
- "I want to use a tool for auth" → search_web (still choosing)
- "What's the best chess opening against d4?" → search_web
- "Show me the Stockfish API" → search_documentation
- "Should I use Stockfish or Leela?" → search_web
```

---

## Environment Variables

```
PERPLEXITY_API_KEY=pplx-xxxx
```

Note: Context7 is already available as an MCP server, no additional API key needed.

---

## Data Flow

```
User asks question
       │
       ▼
Agent determines intent
       │
       ├─── Needs docs ───► search_documentation ───► Context7 ───► Docs + examples
       │
       └─── Needs opinions/research ───► search_web ───► Perplexity ───► Synthesized answer
```

---

## Error Handling

### search_documentation
- Library not found in Context7 → Return helpful message, suggest `search_web` as fallback
- No results for query → Return "No documentation found for X. Try rephrasing or use search_web for general info."

### search_web
- Perplexity API error → Return error message with suggestion to try again
- Rate limited → Return "Search temporarily unavailable, please try again in a moment."

---

## Testing Plan

### search_documentation
1. Query ElevenLabs agent docs (the original failure case)
2. Query a popular library (React, NextJS)
3. Query with and without explicit library parameter
4. Query for something Context7 doesn't have

### search_web
1. "Is Tavily any good?" - should return synthesized opinions
2. "Best alternative to Exa for AI search" - should compare tools
3. "What's new with OpenAI today" - should return current info
4. Compare results to what Exa was returning

---

## Implementation Steps

1. [ ] Add Perplexity API integration
2. [ ] Implement `search_documentation` tool using Context7
3. [ ] Implement `search_web` tool using Perplexity Sonar
4. [ ] Update agent system prompt with routing guidance
5. [ ] Add PERPLEXITY_API_KEY to environment
6. [ ] Test both tools with real queries
7. [ ] Deploy and validate with voice conversations
