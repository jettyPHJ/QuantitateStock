from pydantic import BaseModel
from typing import List, Tuple
from datetime import date, timedelta


class Evaluation(BaseModel):
    title: str
    date: str
    industry_policy_score: float
    peer_competition_score: float
    market_sentiment_score: float
    macro_geopolitics_score: float
    reason: str


def news_prompt(stock_code: str, year: int, month: int, price_changes: List[Tuple[date, float]]) -> str:
    max_up_day = max(price_changes, key=lambda x: x[1])
    max_down_day = min(price_changes, key=lambda x: x[1])

    up_date_str = max_up_day[0].strftime("%Y-%m-%d")
    up_prev_str = (max_up_day[0] - timedelta(days=1)).strftime("%Y-%m-%d")
    up_pct = f"+{max_up_day[1]:.2f}%"

    down_date_str = max_down_day[0].strftime("%Y-%m-%d")
    down_prev_str = (max_down_day[0] - timedelta(days=1)).strftime("%Y-%m-%d")
    down_pct = f"{max_down_day[1]:.2f}%"

    return f"""You are a top-tier financial analyst. Your task is to identify and analyze the 2 most influential news events that clearly explain the **largest positive** and **largest negative** stock price moves for "{stock_code}" in {month}_{year}.

📈 HISTORICAL PRICE MOVES:
The actual daily % change in closing price for {stock_code} in {month}_{year} is shown below:
{price_changes}

🔍 FOCUS DATES:
Only consider news **published on the following dates**:
- {up_prev_str} and {up_date_str} for the **positive price move of {up_pct}**
- {down_prev_str} and {down_date_str} for the **negative price move of {down_pct}**

📌 OBJECTIVE:
From the news published on these 4 days only, select:
- 1 news item that **positively impacted** the stock (linked to the {up_date_str} move).
- 1 news item that **negatively impacted** the stock (linked to the {down_date_str} move).

🧠 GUIDANCE FOR SELECTION:
- Match each news item to the same-day or next-day price move it plausibly influenced.
- Summarize the most impactful and clearly causal events.
- Avoid duplicative or speculative content.

📝 OUTPUT FORMAT:
Return exactly 2 items in the following format:

---
**Title:** [Headline of the news]  
**Date:** [YYYY-MM-DD, news published date]  
**Summary:** [Concise and factual summary of what happened]  
**Impact:** [Positive / Negative]  
**Observed Price Move:** [% price change]  
**Impact Analysis:** [Explain clearly how this caused the stock price movement]
---

🔒 BOUNDARY CONDITIONS:
- Only use news **published on {up_prev_str}, {up_date_str}, {down_prev_str}, or {down_date_str}**.
- Do not include news from other dates.
- Prioritize clarity, causality, and factual accuracy.
"""


def scoring_prompt(stock_code: str, year: int, month: int, news: str) -> str:
    return f"""You are a professional financial analyst. For each of the following news events from {month}_{year} related to the company or stock code "{stock_code}", evaluate the **actual impact** on the stock from four distinct dimensions.

📊 **Impact Dimensions** (score each from -1.0 to +1.0):

1. **Industry & Policy Impact**  
   Impact of industry-wide regulatory changes, government policy shifts, or market structure transformation.  

2. **Peer Competition Impact**  
   Impact of competitor actions (e.g., price wars, product launches, M&A) or the company’s **own** strategic and product decisions.

3. **Market & Sentiment Impact**  
   Reactions by analysts, institutional investors, or media coverage that shape short-term or medium-term market expectations.

4. **Macro & Geopolitical Impact**  
   Influence of large-scale economic forces or geopolitical events (e.g., interest rates, inflation, war, global supply chains).

---

🧠 **Instructions**:

- **Each dimension must be scored separately** on a scale from -1.0 to +1.0, in increments of 0.2.
- Provide a **clear, specific, and logically sound explanation** for each score.  
- Explanation should **explicitly describe the cause-effect chain**:  
  “Event ➜ triggers X ➜ which causes Y ➜ which leads to impact Z on the company.”  
- Avoid vague terms like “bad for the company” or “market reacted negatively”. Always explain **why**.
- If a dimension is not affected, score `0.0` and simply state why.
- Use strong scores (±0.8 / ±1.0) only when the effect is **direct, material, and observable**.

📏 **Scoring Scale** (discrete, applies to all dimensions):

- `+1.0`: Extreme positive — direct and major benefit to core business, valuation, or competitive standing  
- `+0.8`: Strong positive — very favorable, clearly advantageous and likely to affect stock meaningfully  
- `+0.6`: Moderate positive — likely beneficial, but not game-changing  
- `+0.4`: Mild positive — small upside, possibly indirect or long-term  
- `+0.2`: Minimal positive — slight advantage or weak signal  
- ` 0.0`: No meaningful impact — neutral or unrelated  
- `-0.2`: Minimal negative — slight risk or weak concern  
- `-0.4`: Mild negative — small downside, possibly indirect or temporary  
- `-0.6`: Moderate negative — likely harmful to near-term outlook or operations  
- `-0.8`: Strong negative — clearly detrimental and likely to influence stock materially  
- `-1.0`: Extreme negative — direct and significant threat to fundamentals or market value

---

Please analyze and score the following news events in {month}_{year}:  
{news}
"""
