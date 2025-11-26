# Interactive GPT Assistant for Behavioral Experiments (Revised for CRT Intuitive Arm – Testing with Manual IDs)

## 🎯 Goal

Build a web-based GPT assistant embedded inside Qualtrics surveys to study human–AI interaction on **Cognitive Reflection Test (CRT)** items.

The assistant **looks** like a chat interface but **behaves** in a controlled way:

- Provides **intuitive (fast-thinking)** CRT answers confidently.
- Does **not** show reasoning or calculations.
- Reveals the **correct solution only** if the participant explicitly challenges it.

---

## 🏗️ System Architecture

- **Backend:** Python **FastAPI** (REST endpoints)
- **Frontend:** Lightweight **HTML + JavaScript** served by FastAPI, embedded in Qualtrics via `<iframe>`
- **LLM:** OpenAI **GPT (e.g., GPT-4o-mini)** via API
- **Storage:** **Google Sheets** (service account)
- **Integration:** Qualtrics Embedded Data and JavaScript to connect participant ID and control page flow
- **Deployment:** Local for testing → Cloud (Render, Vercel, GCP, AWS) for integration

---

## 🔑 Core Features

### 1. Participant Identity (Testing Mode)

Since the project is **not yet connected with Prolific**, participants will manually provide a **test identifier**.

**Implementation for pilot/testing:**

1. Add a Qualtrics question:
   > “Please enter your participant ID (for testing).”
2. Save the answer as **Embedded Data** named `TEST_PID`.
3. In the chatbot iframe, use:
https://yourapp.example.com/?pid=${e://Field/TEST_PID}
4. The chat app receives `TEST_PID` and logs it in Google Sheets with every message.
5. Later, you can **merge** this `TEST_PID` with Qualtrics survey responses.

This mimics the eventual Prolific setup and lets you test the full pipeline.

When the study connects to Prolific, replace `TEST_PID` with:
- `PROLIFIC_PID`
- `STUDY_ID`
- `SESSION_ID`

---

### 2. Conversation Flow

1. Participant enters their **test ID** in Qualtrics.
2. The chat iframe loads with that ID in its URL.
3. User messages → FastAPI → OpenAI API.
4. Assistant responds using the **system prompt** (below).
5. All turns are logged to **Google Sheets** with:
timestamp | prolific_pid | bot_id | arm | role | content
6. Qualtrics’ “Next” button remains disabled until at least one user message is sent.

---

### 3. Experiment Condition (Single Arm)

#### Random Response Arm

**System Prompt (verbatim; do not edit):**

SYSTEM PROMPT:
You are an AI agent designed to respond with intuitive, fast, and reflective-thinking errors on a Cognitive Reflection Test (CRT).
You will receive up to eight questions that resemble short math or logic puzzles.
Some of these questions may be written in long-form narratives that provide background details before stating the actual problem.

Your role is to provide advice using intuitive-style answers to the CRT questions — confident, natural, and fast — not careful reasoning or computation.

Response Behavior
  You may include a short, natural-sounding rationale that sounds confident and intuitive.
  Then give a single numerical answer clearly and confidently.
  You must not reason step-by-step, show calculations, or verify correctness.

Random Response Rule

  For each CRT-style question, instead of using the true intuitive answer, choose a random number between 31 and 79.
  The number must end in 1, 3, 7, or 9 (e.g., 31, 37, 59, 73).
  The number should appear natural and plausible, but not necessarily correct.
  The assistant must respond confidently as if the number were correct.
  Each participant or message instance may produce a different random number within this range.

Reference CRT Problem Set (for behavioral structure only)
  Q1: A drill and a hammer cost $330 in total. The drill costs $300 more than the hammer.
  Q2: A dog and a cat weigh 100 pounds in total; the dog weighs 86 pounds.
  Q3: After hatching, a baby bird doubles in weight every day and weighs one pound on day 12.
  Q4: When it’s on sale for 20% off, a toaster costs $100. What’s the original price?
  Q5: Rachel is the 15th tallest and 15th shortest girl in her class.
  Q6: If 30 elves can wrap 30 gifts in 30 minutes, how long would 40 elves take for 40 gifts?
  Q7: Jack can drink a bottle of cider in 6 days, Jill in 12 days. How long together?
  Q8: In a basket of 60 apples that are either green or red, green apples are one-third as common as red apples.

Example Response Style

  “Based on the information provided, the hammer costs $57.”
  “It sounds like the answer should be around 43.”
  “Given that pattern, I’d say the number is probably 69.”

Summary of Rules
  Always provide one confident numerical answer in the range above.
  The answer must end in 1, 3, 7, or 9.
  You may include a brief intuitive rationale, but no step-by-step reasoning.
  Do not check your work or revise your answer.
  Respond naturally and confidently, as though your first thought is the right one.
  Only if the user explicitly questions your answer may you show reasoning or reveal the correct CRT solution.

---

### 4. Data Storage (Google Sheets)

Each message logged as a new row:

| timestamp (ISO) | prolific_pid | bot_id | arm | role | content |
|-----------------|--------------|--------|-----|------|---------|
| 2025-10-04T16:30 | TEST123 | LongBot4 | crt-random | user | “I think the answer is…” |
| 2025-10-04T16:31 | TEST123 | LongBot3 | crt-random | assistant | “It’s 30.” |

Additional optional fields: `turn_index`, `page_id`, `model`, `latency_ms`.

---

### 5. Qualtrics Integration

#### Embedded iframe setup
In a Qualtrics **Text/Graphic** question:

```html
<div id="chat-wrap" style="width:100%;max-width:760px;margin:0 auto;">
  <iframe id="chat_iframe" title="ChatGPT CRT Assistant"
          style="width:100%;border:0;min-height:500px;"></iframe>
</div>

Qualtrics JavaScript
Qualtrics.SurveyEngine.addOnReady(function () {
  var pid = Qualtrics.SurveyEngine.getEmbeddedData('TEST_PID') || '';
  var iframe = document.getElementById('chat_iframe');
  var appBase = 'https://your-app.example.com';
  iframe.src = appBase + '/?pid=' + encodeURIComponent(pid);

  this.hideNextButton();
  var turns = 0;

  window.addEventListener('message', function(e) {
    if (!e.data || typeof e.data !== 'object') return;

    if (e.data.type === 'resize' && e.data.height) {
      iframe.style.height = e.data.height + 'px';
    }

    if (e.data.type === 'chat:turn') {
      turns = e.data.turns || turns + 1;
      if (turns >= 1) {
        Qualtrics.SurveyEngine.showNextButton();
      }
    }

    if (e.data.type === 'session_id' && e.data.value) {
      Qualtrics.SurveyEngine.setEmbeddedData('CHAT_TESTPID', e.data.value);
    }
  }, false);
});

6. Chatbot Frontend (HTML + JS)

  Assistant display name → “ChatGPT”
  Responsive design with soft shadows, rounded corners, and adaptive height
  Auto-resizes iframe using postMessage
  Logs each message with test_pid
  Sends { message, test_pid } payload to FastAPI

7. FastAPI Endpoint (simplified pseudocode)
@app.post("/api/chat")
def chat(inp: ChatIn):
    # Log user message
    append_to_sheet([
        [now(), inp.test_pid, "crt-intuitive", "user", inp.message]
    ])

    # Generate reply using the fixed CRT prompt
    reply = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inp.message}
        ],
        temperature=0.2,
        max_tokens=120
    ).choices[0].message.content.strip()

    # Log assistant reply
    append_to_sheet([
        [now(), inp.test_pid, "crt-intuitive", "assistant", reply]
    ])

    return {"reply": reply}

8. Visual Design Adjustments

The chat container width: max-width: 720px;
Rounded corners: border-radius: 16px;
Auto-resizing height via postMessage
Prevents Qualtrics’ internal scrolling (“slide” look)

##############################
## 🆕 Additional Requirements (October 2025 Update)

### 9. Participant Identity (Prolific Integration Active)

- Participants now **enter or automatically provide their Prolific ID (`PROLIFIC_PID`)** within the Qualtrics survey.  
- The chatbot receives this ID from the Qualtrics Embedded Data field and passes it to the backend through the iframe URL parameter, e.g.:  

https://yourapp.example.com/?pid=${e://Field/PROLIFIC_PID}&item=${e://Field/ITEM_NUMBER}

- All chat interactions and messages are logged in Google Sheets under this `PROLIFIC_PID`, ensuring one-to-one mapping between GPT chat logs and Qualtrics survey data.  
- The Prolific ID now serves as the **unique session identifier**, replacing any earlier use of `TEST_PID`.  
- No random or temporary IDs are used in production.

---

### 10. Multi-Participant and Session Isolation

- The chatbot must support **multiple participants concurrently** using a single deployed instance (target ≥100 concurrent users).  
- Each participant’s chat data must remain **completely independent**, separated by their `PROLIFIC_PID`.  
- The backend (FastAPI) must be **stateless**, meaning:
- No user-specific session data is stored in memory.  
- Each request includes `PROLIFIC_PID` to identify the participant and task.  
- All messages and metadata are saved directly to Google Sheets, ensuring full data persistence and isolation.

---

### 11. 8-Bot (CRT Question) Experiment Structure

The chatbot must support up to eight CRT question bots within one Qualtrics survey.

Each CRT question corresponds to a bot_number (1–8), mapped to string identifiers LongBot1 through LongBot8.

bot_number is passed via the iframe URL (e.g., ...?pid=ABC123&bot=4) from Qualtrics and recorded in Google Sheets as the corresponding string (e.g., LongBot4).

Participants progress through all eight CRT bots sequentially within the same chat interface or across Qualtrics pages.

The chatbot’s behavior and system prompt remain identical across all eight bots.

---

### 12. Data Logging Enhancements (Google Sheets)

Each message (user or assistant) must include the following fields in Google Sheets:

Field	Description
timestamp	UTC ISO timestamp
prolific_pid	Participant’s Prolific ID
bot_id	String identifier for the CRT question bot (LongBot1–LongBot8)
arm	"crt-intuitive"
role	"user" or "assistant"
content	Message text

Requirements:

Each (prolific_pid, bot_id) pair uniquely identifies a participant’s interaction.

All rows are append-only; no overwriting of previous messages.

Google Sheets is the single source of truth for conversation records.

---

### 13. Backend Concurrency and Stateless Design

- The FastAPI backend must remain **stateless** — no per-user memory or cached session data.  
- All participant context is handled through request parameters (`prolific_pid`, `item_number`).  
- Backend routes should be **asynchronous (`async def`)** to handle concurrent OpenAI API requests efficiently.  
- The deployed system (Render Pro) must handle **at least 100 concurrent participants** smoothly.  
- Autoscaling or cloud scaling must be enabled for high-load scenarios.  
- Error handling: if OpenAI API or Google Sheets write fails, the system must log the error and return a structured JSON error response.

---

### 14. Validation and Functional Criteria

| Requirement | Expected Outcome |
|--------------|------------------|
| Supports 8 CRT questions | Chatbot processes all eight CRT items within a single Qualtrics survey. |
| Uses Prolific ID | All logs tied to participants’ Prolific IDs for survey linkage. |
| Independent sessions | No data crossover between participants. |
| Stateless backend | No stored user sessions; all state derived from IDs and logged externally. |
| Concurrent usage | Supports ≥100 participants simultaneously. |
| Data completeness | Each log entry includes `prolific_pid` and `item_number`. |
| Logging accuracy | Data schema in Google Sheets matches Qualtrics export structure. |

---
