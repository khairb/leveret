<p align="center">
  <img src="assets/scout.png" alt="Scout" width="200" />
</p>

<h1 align="center">Scout</h1>

<p align="center">
  <strong>An agent writes you a Playwright scraper, then leaves and comes back on its own when the script breaks.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/scout/"><img src="https://img.shields.io/pypi/v/scout.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scout/"><img src="https://img.shields.io/pypi/pyversions/scout.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

```bash
pip install scout
```

```python
from scout import Scout
from pydantic import BaseModel, Field

class Listing(BaseModel):
    title: str
    price: float = Field(gt=0, lt=10000)
    location: str
    seller_response_rate: float = Field(ge=0, le=1)

scraper = Scout.create(
    task="""
        Go to example-marketplace.com/bikes, filter to listings posted
        in the last week under $500, paginate through every page,
        click into each detail page, and grab the title, price,
        location, and seller response rate.
    """,
    schema=Listing,
    output="bikes_scraper.py",
)

# Done. You now have bikes_scraper.py — a real Playwright file
# you can read, edit, commit, and run on a cron with no model in the loop.
```

That's the whole pitch. Keep reading if you want to know why it works that way.

## Why this exists

If you've written scrapers for a living, you know the ambush. The scraper runs fine for four months. Then on a Tuesday, the site ships a redesign, your selectors stop matching, and you find out from an empty dashboard. You spend twenty unglamorous minutes fixing it. You do this six times a year per scraper. You have eleven scrapers.

If you've tried the AI scraping tools — Browser Use, Firecrawl, ScrapeGraphAI's smart mode — you know the other problem. They work. They also call an LLM on every single scrape. You scale up to a real recurring job, do the math on the per-request bill, and close the tab.

Scout is built on a different idea: **the model should write the scraper, not be the scraper.**

You describe the job in a sentence. An agent opens a real browser, navigates the site the way you would, writes a Playwright script in pieces, runs each piece, debugs its own selectors when they miss by actually looking at the page, and iterates until the output matches your schema. This takes one to ten minutes depending on the site.

When it's done, the agent is gone. What you have is a plain Python file. It runs on a cron in seconds, with no API key in the job, no per-request bill, and no model in the loop. You own it the same way you'd own a scraper you wrote yourself — because, structurally, it *is* a scraper you wrote yourself, just with the boring part done by something else.

## When the site breaks

Every run validates against your schema. If the script crashes, returns too few rows, or returns rows that fail validation, Scout knows. With `auto_heal=True`, the agent gets called back in automatically — it receives the original task, the original schema, the broken script, and the actual error, and writes a fresh script. Your cron job keeps working. You don't get paged.

```python
scraper = Scout.create(
    task="...",
    schema=Listing,
    output="bikes_scraper.py",
    auto_heal=True,
)
```

The model is on call, not on staff. You pay for it when something actually changes, which is the only time you should be paying for it.

## What's actually in the file

The file Scout writes is a normal Playwright script. No Scout imports, no runtime dependency on this library, no framework lock-in. You can read it. You can edit it. You can delete the parts you don't like and keep the parts you do. You can commit it to git and review the diff like any other code. If you uninstall Scout tomorrow, your scrapers keep running.

This matters because the thing you actually want from a scraper is a file you control. Everything else is somebody else's roadmap.

## How it compares

|  | Scout | Browser Use / Firecrawl / ScrapeGraphAI |
|---|---|---|
| LLM calls per scrape (steady state) | 0 | 1 per scrape |
| Output | A Playwright file you own | A managed service result |
| Cost at 10k scrapes/month | One-time generation + occasional repair | Per-request, forever |
| What you have if the vendor disappears | Your scrapers | Nothing |
| Detection profile | Patchright (low) | Varies |

This isn't a knock on those tools — they're solving a different problem. If you want a one-shot scrape of an unfamiliar site and you don't care about the artifact, use them. If you want a scraper you can put in production and forget about, that's what Scout is for.

## Install

```bash
pip install scout
playwright install chromium
```

You'll need an API key for the model that writes and repairs scripts. Set it once:

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
```

The key is only used when Scout is *writing* or *repairing* a script. The scripts it produces don't need it.

## Status

Scout is new. It works, it's tested on a growing list of real sites, and it's being used in production by a small number of people including me. There will be sites it can't handle yet. If you find one, open an issue with the URL and the task description — those reports are the most useful thing you can contribute right now.

## License

MIT.

---

<p align="center">
  <sub>Scout goes out, does the job, and comes back when you need it.</sub>
</p>