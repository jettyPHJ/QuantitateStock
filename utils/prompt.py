from pydantic import BaseModel
from typing import List, Optional, Literal
import datetime
import numpy as np
from dataclasses import dataclass


class Evaluation(BaseModel):
    title: str
    date: str
    industry_policy_score: float
    peer_competition_score: float
    market_sentiment_score: float
    macro_geopolitics_score: float
    reason: str


@dataclass
class PriceChangeRecord:
    date: datetime.date
    stock_pct_chg: Optional[float] = None
    block_pct_chg: Optional[float] = None


@dataclass
class AttributionRecord:
    date: datetime.date
    stock_pct_chg: float
    block_pct_chg: float
    direction: Literal["positive", "negative"]
    divergence: Literal["same_direction", "opposite_direction"]
    alignment_type: Literal["aligned", "amplified", "divergent"]
    likely_cause_category: str
    likely_causes: List[str]


LIKELY_CAUSE_LIBRARY = {
    "Macroeconomic/Industry": [
        "Macroeconomic Factors",
        "Sector-wide News",
        "Policy or Regulatory Changes",
    ],
    "Company Specific": [
        "Financial Results",
        "Product Launch or Recall",
        "Management Change",
        "Mergers & Acquisitions",
        "Unusual Trading Activity",
    ],
    "Market Technical": [
        "Analyst Rating Change",
        "Abnormal Volume",
        "Short Selling Activity",
    ],
}


def get_analyse_records(price_change_records: List[PriceChangeRecord]) -> List[AttributionRecord]:
    if not price_change_records:
        return []

    valid_records = [r for r in price_change_records if r.stock_pct_chg is not None]
    if not valid_records:
        return []

    stock_changes = [r.stock_pct_chg for r in valid_records]
    abs_stock_changes = np.abs(stock_changes)
    top_10_percentile = np.percentile(abs_stock_changes, 90)

    analyse_records = [
        r for r in valid_records if abs(r.stock_pct_chg) >= 5 and abs(r.stock_pct_chg) >= top_10_percentile
    ]

    attribution_records: List[AttributionRecord] = []
    amplified_threshold = 3.0  # 可调参数

    for r in analyse_records:
        if r.stock_pct_chg is None or r.block_pct_chg is None:
            continue

        direction = "positive" if r.stock_pct_chg > 0 else "negative"
        divergence = "same_direction" if r.stock_pct_chg * r.block_pct_chg >= 0 else "opposite_direction"
        pct_diff = abs(r.stock_pct_chg - r.block_pct_chg)

        if divergence == "opposite_direction":
            alignment_type = "divergent"
            likely_cause_category = "Company Specific"
        elif pct_diff < amplified_threshold:
            alignment_type = "aligned"
            likely_cause_category = "Macroeconomic/Industry"
        else:
            alignment_type = "amplified"
            likely_cause_category = "Company Specific"

        likely_causes = LIKELY_CAUSE_LIBRARY[likely_cause_category]

        attribution_records.append(
            AttributionRecord(date=r.date, stock_pct_chg=r.stock_pct_chg, block_pct_chg=r.block_pct_chg,
                              direction=direction, divergence=divergence, alignment_type=alignment_type,
                              likely_cause_category=likely_cause_category, likely_causes=likely_causes))

    return attribution_records


def news_prompt(stock_code: str, record: AttributionRecord) -> str:
    date_str = record.date.strftime("%Y-%m-%d")
    direction_text = "rose" if record.direction == "positive" else "fell"

    divergence_map = {
        "same_direction": "moved in the same direction as its sector",
        "opposite_direction": "moved in the opposite direction of its sector"
    }
    alignment_map = {
        "aligned": "The stock and sector moved similarly, suggesting macroeconomic or industry-level influence.",
        "amplified": "The stock moved more significantly than the sector, possibly due to company-specific amplification.",
        "divergent": "The stock diverged from the sector trend, indicating potential major company-specific news."
    }

    # 拼接 Prompt
    prompt = f"""You are a top-tier financial analyst. Your task is to identify **1 most likely news event** explaining the abnormal stock price movement of "{stock_code}" on {date_str}.

📈 STOCK MOVEMENT CONTEXT:
- The stock {direction_text} by {record.stock_pct_chg:.2f}% on {date_str}, while the sector changed by {record.block_pct_chg:.2f}%.
- The stock {divergence_map[record.divergence]}.
- Interpretation: {alignment_map[record.alignment_type]}

📅 TIME WINDOW:
Only consider news published from {record.date - datetime.timedelta(days=2)} to {record.date} (inclusive). Do not include earlier or later events.

🎯 OBJECTIVE:
From this 3-day window, select **one news item** that most plausibly caused the observed price movement. Your reasoning must follow the **structured cause-effect chain** below.

🏷️ LIKELY CAUSE CATEGORIES:
- Category: {record.likely_cause_category}
- Suggested Subtypes: {", ".join(record.likely_causes)}

📝 OUTPUT FORMAT:
---
**Title:** [News headline]  
**Date:** [YYYY-MM-DD, news published date]  
**Summary:** [Concise, factual summary of the news]  
**Observed Movement:** Stock {direction_text} by {record.stock_pct_chg:.2f}% vs sector {record.block_pct_chg:.2f}%  
**Cause Category:** {record.likely_cause_category} ➜ [Select one from: {", ".join(record.likely_causes)}]  

**Impact Chain (Required – 5 stages):**
1. **Triggering Event:** What specifically happened? (e.g., earnings release, policy change, executive resignation)  
2. **Immediate Effect:** What was the immediate, measurable impact? (e.g., revenue down 10%, profit warning issued)  
3. **Company-Level Impact:** How did this affect the company's business, strategy, or financial outlook?  
4. **Investor Interpretation:** How did investors interpret this event? Did it alter expectations, sentiment, or valuation assumptions?  
5. **Stock Reaction:** How did the stock move in response to the investor interpretation on {date_str}?
---

🔒 CONSTRAINTS:
- Do NOT include news outside the 3-day window.
- You must select a cause subtype from the provided list.
- The **Impact Chain** is mandatory. Do not skip or collapse steps.
"""

    return prompt


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
