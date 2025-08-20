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
    likely_causes: List[str]


def get_analyse_records(price_change_records: List[PriceChangeRecord]) -> List[AttributionRecord]:
    """
    筛选出个股涨跌幅在前10%且涨跌幅绝对值不小于5%的记录，并进行归因分析。
    """
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
            likely_causes = ["Major Corporate News", "Product/Tech Breakthrough", "Unusual Trading Activity"]
        else:
            if pct_diff < amplified_threshold:
                alignment_type = "aligned"
                likely_causes = ["Macroeconomic Factors", "Market Sentiment", "Industry/Sector News"]
            else:
                alignment_type = "amplified"
                likely_causes = ["Company Specific Event (Amplified by Market)", "Financial Results", "Management News"]

        attribution_records.append(
            AttributionRecord(date=r.date, stock_pct_chg=r.stock_pct_chg, block_pct_chg=r.block_pct_chg,
                              direction=direction, divergence=divergence, alignment_type=alignment_type,
                              likely_causes=likely_causes))

    return attribution_records


def news_prompt(stock_code: str, record: AttributionRecord) -> str:
    """
    根据 AttributionRecord 生成用于新闻归因分析的 Prompt。
    要求模型在分析日期的前三天（含当日）范围内，找出导致股价大幅波动的主要新闻。
    """
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

    prompt = f"""You are a top-tier financial analyst. Your task is to identify the most likely news events that explain the abnormal price movement of stock "{stock_code}" on {date_str}.

📈 STOCK MOVEMENT CONTEXT:
- The stock {direction_text} by {record.stock_pct_chg:.2f}% on {date_str}, while the sector changed by {record.block_pct_chg:.2f}%.
- The stock {divergence_map[record.divergence]}.
- Interpretation: {alignment_map[record.alignment_type]}

📅 TIME WINDOW:
Only consider news published from {record.date - datetime.timedelta(days=2)} to {record.date} (inclusive).

🎯 OBJECTIVE:
Identify 1 key news events within this 3-day window that most likely caused the stock’s abnormal movement. Focus on the following potential causes:
- {", ".join(record.likely_causes)}

📝 OUTPUT FORMAT:
---
**Title:** [Headline of the news]  
**Date:** [YYYY-MM-DD, news published date]  
**Summary:** [Concise and factual summary of what happened]  
**Impact:** [Positive / Negative]  
**Observed Price Move:** [% block and stock price change]  
**Impact Analysis:** [Explain clearly how this caused the stock price movement]
---

🔒 BOUNDARY CONDITIONS:
- Do not include news from other dates.
- Prioritize clarity, causality, and factual accuracy.
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
