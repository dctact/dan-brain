# Feature: Automatic Conversation Summary Email

**Status:** Parked
**Created:** 2026-01-21
**Priority:** After search tools are implemented

## Overview

After every conversation >30 seconds, automatically synthesize and email a summary to Dan so he can follow up when he gets to his computer.

## Trigger

- Conversation ends (silence threshold or explicit goodbye)
- Duration was >30 seconds

## Synthesis Prompt

Extract from the conversation:
- **Decisions made** - "You decided X is the right tool"
- **What works / doesn't work** - "This chess opening works against Y but fails against Z"
- **Action items with context** - "Download [App] from the App Store"
- **Research findings** - Key learnings from any searches
- **Open questions** - Things still unresolved

## Email Format

```
Subject: [Dan's Brain] {topic}

---

## Decisions
- ...

## What Works
- ...

## What Doesn't Work
- ...

## Action Items
- [ ] ...

## Open Questions
- ...

---
Conversation: {date} at {time} ({duration})
```

## Implementation

### New Components

| Component | Purpose |
|-----------|---------|
| `_send_email()` | Internal function - sends via Resend or SendGrid |
| `_synthesize_conversation_summary()` | Uses Gemini to extract decisions/actions/findings |
| Post-conversation hook | Triggers email after conversations >30s |

### New Environment Variables

- `RESEND_API_KEY` or `SENDGRID_API_KEY`
- `EMAIL_TO` (Dan's email address)

### Integration Point

Likely hooks into `process_transcript()` or runs as a separate post-conversation step after transcript processing completes.

## Open Questions

- Exact silence threshold for "conversation ended"
- Should high-priority items (action items with deadlines) get flagged differently?
- Batching if multiple short conversations happen back-to-back?
