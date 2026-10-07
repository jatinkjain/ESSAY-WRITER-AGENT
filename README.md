# 📝 AI Essay Writer

An agentic AI essay-writing system built with **LangGraph** that researches a topic, creates an essay, critiques the draft, performs additional research, and generates a revised final essay.

The project uses a graph-based workflow to demonstrate how multiple AI tasks can be connected into a multi-step agent.

---

## 🚀 Features

-  Automatic essay planning
-  Web research using Tavily
-  AI-powered essay generation
-  Automatic essay critique
-  Additional research based on critique
-  Automatic essay revision
-  Multi-step LangGraph workflow
-  Simple terminal-based interface

---

## 🧠 Agent Workflow

The application follows this workflow:

```text
START
  │
  ▼
Planner
  │
  ▼
Research Plan
  │
  ▼
Generate Essay
  │
  ▼
Should Continue?
  │
  ├──────────────► END
  │
  ▼
Reflect / Critique
  │
  ▼
Research Critique
  │
  ▼
Generate Revised Essay
  │
  ▼
Should Continue?
  │
  ▼
END
```

