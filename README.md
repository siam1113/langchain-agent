# LangChain Agent

Experiments with LangChain and Playwright: building a browser-driving agent powered by an LLM.

## Layout

```text
src/
  bot.py           Main agent entry point
  playwright.py    Playwright browser tooling used by the agent
  stateful.py       Stateful agent experiment
  requirements.txt  Python dependencies
Diagrams.md         Architecture diagrams and notes
New_Learnings.md     Notes on things learned while building this
```

## Getting Started

```bash
cd src
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python bot.py
```
