from pydantic import BaseModel


class Evaluation(BaseModel):
    title: str
    date: str
    industry_policy_score: float
    peer_competition_score: float
    market_sentiment_score: float
    macro_geopolitics_score: float
    reason: str


def news_prompt(stock_code: str, year: int, month: int, price_changes: list) -> str:
    return f"""You are a top-tier financial analyst. Your task is to identify and analyze the top 2 most influential news from {month}_{year} that significantly impacted the stock price of "{stock_code}".

📈 HISTORICAL PRICE MOVES:
Here is the actual daily % change in closing price for {stock_code} in {month}_{year}:
{price_changes}

📌 OBJECTIVE:
From all news items published during {month}_{year}, select exactly 2 events:
- 1 news items with a **clear positive impact** on the stock’s price.
- 1 news items with a **clear negative impact** on the stock’s price.

Use the daily price changes above to identify and validate potential causal links between news and price movement.

🧠 GUIDANCE FOR SELECTION:
- Prioritize events that clearly explain large price swings (Match each news item to the largest same-day or next-day price change that logically aligns).
- Merge duplicate or ongoing news threads into a single summarized item.

📝 OUTPUT FORMAT:
Return exactly 2 items in this structure:

---
**Title:** [Headline of the event]  
**Date:** [YYYY-MM-DD]  
**Summary:** [Concise and factual summary of what happened]  
**Impact:** [Positive / Negative]  
**Observed Price Move:** [% price change]  
**Impact Analysis:** [Explain clearly how this caused the stock price movement]
---

🔒 BOUNDARY CONDITIONS:
- Only use news published in {month}_{year} (publication date, not actual event occurrence date).
- Avoid vague, speculative, or unverified information.
- Precision and causality are more important than coverage.
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

- **Each dimension must be scored separately** on a scale from -1.0 to +1.0.
- Provide a **clear, specific, and logically sound explanation** for each score.  
- Explanation should **explicitly describe the cause-effect chain**:  
  “Event ➜ triggers X ➜ which causes Y ➜ which leads to impact Z on the company.”
- Avoid vague terms like “bad for the company” or “market reacted negatively”. Always explain **why**.

- If a dimension is not affected, score `0.0` and simplely state why.
- Only use strong scores (≥ |0.6|) when the effect is **direct, material, and observable**.

📏 **Scoring Scale** (applies to all dimensions):

- `+0.6 to +1.0`: Strong positive, direct and significant impact  
- `+0.3 to +0.5`: Moderate positive impact  
- `-0.2 to +0.2`: Neutral / marginal / indirect  
- `-0.3 to -0.5`: Moderate negative impact  
- `-0.6 to -1.0`: Strong negative, direct and significant impact  

---

Please analyze and score the following news events in {month}_{year}:  
{news}
"""
