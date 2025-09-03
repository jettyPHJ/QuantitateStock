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
    stock_pct_chg: Optional[float] = None  # 百分比数值 eg. 3 代表 3%
    block_pct_chg: Optional[float] = None  # 百分比数值


@dataclass
class AttributionRecord:
    date: datetime.date
    stock_pct_chg: float
    block_pct_chg: float
    direction: Literal["positive", "negative"]
    divergence: Literal["same_direction", "opposite_direction"]
    alignment_type: Literal["aligned", "amplified", "divergent", "isolated_move"]
    likely_cause_category: str
    likely_causes: List[str]


# ==================== 基于“预期变化”的全新原因库 ====================
# 这个库的核心是描述“市场对公司的哪一类核心预期发生了改变”，而不是罗列具体事件。
EXPECTATION_BASED_CAUSE_LIBRARY = {
    # ==================== 公司特有预期变化 (Company-Specific Expectation Changes) ====================

    # --- A. 逆势/颠覆性的预期重置 (For OverridingFactor) ---
    "Fundamental_Expectation_Reset_Positive": [
        "Paradigm Shift in Market Structure (e.g., gaining monopoly)",  # 市场格局/垄断预期的范式转移
        "Fundamental Upward Re-rating of Long-Term Growth/Moat",  # 对长期增长/护城河的颠覆性重估
        "Fundamental Reset of Valuation due to Acquisition",  # 公司被收购导致的颠覆性估值重置
        "Complete De-risking of a Major Uncertainty (e.g., legal/regulatory win)",  # 重大不确定性被完全消除
    ],
    "Fundamental_Expectation_Reset_Negative": [
        "Fundamental Threat to Long-Term Business Viability",  # 对长期商业模式可行性的根本性质疑
        "Complete Write-off of a Key Future Growth Driver",  # 对未来关键增长引擎的预期完全破灭
        "Crisis of Confidence in Governance/Management Integrity",  # 对公司治理/管理层信誉的信任危机
        "Emergence of an Existential Threat (e.g., competitor, regulation)",  # 出现关乎公司存亡的根本性威胁
    ],

    # --- B. 在趋势中被放大/缩小的预期 (For AmplifyingFactor) ---
    "Amplified_Expectation_Positive": [
        "Expectation of Disproportionate Gains as a Market Leader",  # 作为行业领导者，预期将获得超额收益
        "Amplified Expectation of Market Share Consolidation",  # 市场份额将加速集中的预期被放大
        "Belief that Company is a Prime Beneficiary of Sector Tailwinds",  # 公司是行业顺风的核心受益者的信念增强
    ],
    "Amplified_Expectation_Negative": [
        "Expectation of Disproportionate Losses as a Laggard",  # 作为行业落后者，预期将遭受超额损失
        "Amplified Concern over Eroding Competitive Position",  # 对竞争地位被侵蚀的担忧被放大
        "Belief that Company is Uniquely Vulnerable to Sector Headwinds",  # 公司极易受行业逆风冲击的信念增强
    ],

    # --- C. 独立的、常规的预期调整 (For IsolatedDriver) ---
    "Isolated_Expectation_Adjustment_Positive": [
        "Upward Revision of Near-Term Financial Outlook (Revenue/Profit)",  # 近期财务前景（收入/利润）预期被上调
        "Positive Revision of Competitive Landscape / Market Share",  # 竞争格局/市场份额预期向好
        "Validation of Product/Technology Leadership",  # 产品/技术领先地位得到验证
        "Increased Expectation of Shareholder Returns (e.g., buybacks)",  # 股东回报预期增强（如回购）
    ],
    "Isolated_Expectation_Adjustment_Negative": [
        "Downward Revision of Near-Term Financial Outlook (Revenue/Profit)",  # 近期财务前景（收入/利润）预期被下调
        "Negative Revision of Competitive Landscape / Market Share",  # 竞争格局/市场份额预期向坏
        "Erosion of Product/Technology Advantage",  # 产品/技术优势被削弱
        "Lowered Expectation of Per-Share Value (e.g., dilution)",  # 每股价值预期被稀释（如增发）
    ],

    # ==================== 宏观/行业预期变化 (Macro/Industry Expectation Changes) ====================
    "SectorDriven": [
        "Shift in Regulatory/Policy Environment Expectation",  # 监管/政策环境预期发生转变
        "Change in Macroeconomic Outlook (e.g., growth, inflation)",  # 宏观经济前景预期发生变化
        "Revision of Industry Growth Trajectory / TAM",  # 行业增长路径/总市场规模预期被修正
        "Disruption in Supply Chain or Input Cost Expectation",  # 供应链或成本预期被扰动
    ],

    # ==================== 回退选项 ====================
    "MarketTechnicalFallback": [
        "Driven by Trading Momentum/Speculation, not new expectations",  # 交易动能/市场炒作驱动，无明确预期变化
        "Liquidity-driven Move (e.g., large fund inflow/outflow)",  # 流动性驱动（如大型基金的买卖）
        "Options Market Induced Volatility (e.g., Gamma Squeeze)",  # 期权市场引发的异动
    ],
}


def get_analyse_records(
    price_change_records: List[PriceChangeRecord],
    sector_threshold=3,
    amplified_multiplier=1.67,
) -> List[AttributionRecord]:
    """
    获取达到分析要求的记录
    amplified_multiplier: 股票涨跌幅和板块涨跌幅的对比阈值,用于识别是否放大效应
    sector_threshold: 板块波动的显著性阈值 (例如 3 代表 3%)
    """
    if not price_change_records:
        return []

    valid_records = [r for r in price_change_records if r.stock_pct_chg is not None and r.block_pct_chg is not None]
    if not valid_records:
        return []

    # 取 top 10% 极端变动
    stock_changes = [r.stock_pct_chg for r in valid_records]
    abs_stock_changes = np.abs(stock_changes)
    top_10_percentile = np.percentile(abs_stock_changes, 90)

    analyse_records = [
        r for r in valid_records if abs(r.stock_pct_chg) >= 5 and abs(r.stock_pct_chg) >= top_10_percentile
    ]

    attribution_records: List[AttributionRecord] = []

    for r in analyse_records:
        direction = "positive" if r.stock_pct_chg > 0 else "negative"
        divergence = "same_direction" if r.stock_pct_chg * r.block_pct_chg >= 0 else "opposite_direction"

        # ==================== 基于“预期变化”的归因逻辑 ====================
        if abs(r.block_pct_chg) > sector_threshold:
            # --- 场景一：板块波动显著 ---
            if r.stock_pct_chg * r.block_pct_chg < 0:
                alignment_type = "divergent"
                likely_cause_category = "Fundamental_Expectation_Reset_Positive" if direction == "positive" else "Fundamental_Expectation_Reset_Negative"

            elif abs(r.stock_pct_chg) > abs(r.block_pct_chg) * amplified_multiplier:
                alignment_type = "amplified"
                likely_cause_category = "Amplified_Expectation_Positive" if direction == "positive" else "Amplified_Expectation_Negative"

            else:
                alignment_type = "aligned"
                likely_cause_category = "SectorDriven"

        else:
            # --- 场景二：板块波动不显著 ---
            alignment_type = "isolated_move"
            likely_cause_category = "Isolated_Expectation_Adjustment_Positive" if direction == "positive" else "Isolated_Expectation_Adjustment_Negative"

        # ==========================================================

        # 使用新的原因库
        likely_causes = EXPECTATION_BASED_CAUSE_LIBRARY[likely_cause_category]

        attribution_records.append(
            AttributionRecord(
                date=r.date,
                stock_pct_chg=r.stock_pct_chg,
                block_pct_chg=r.block_pct_chg,
                direction=direction,
                divergence=divergence,
                alignment_type=alignment_type,
                likely_cause_category=likely_cause_category,
                likely_causes=likely_causes,
            ))

    return attribution_records


def news_prompt(stock_code: str, record: AttributionRecord) -> str:
    """
    生成与“预期管理”归因库完全对齐的最终版Prompt。
    """
    date_str = record.date.strftime("%Y-%m-%d")
    direction_text = "rose" if record.direction == "positive" else "fell"

    # 1. 更新 alignment_map，使其语言与“预期变化”的框架对齐
    alignment_map = {
        "aligned": "The stock's movement was driven by a **shift in sector-wide expectations**. Your goal is to find the news that changed the outlook for the entire industry.",
        "amplified": "A sector-wide expectation shift occurred, but the market re-evaluated this company **more dramatically**. Your goal is to find the news that explains this **heightened sensitivity and amplified expectation change**.",
        "divergent": "The stock's price moved contrary to the sector, indicating a **powerful, company-specific expectation reset** that completely overrode the industry trend. Your goal is to find the trigger for this **fundamental re-evaluation**.",
        "isolated_move": "The sector context was neutral. The stock's movement was caused by a **standalone adjustment in company-specific expectations**. Your goal is to find the news that triggered this isolated re-evaluation."
    }

    fallback_causes = EXPECTATION_BASED_CAUSE_LIBRARY["MarketTechnicalFallback"]
    stock_pct_str = f"{record.stock_pct_chg :.2f}%"
    block_pct_str = f"{record.block_pct_chg :.2f}%"

    # 2. 重构 Prompt 的核心指令和术语
    prompt = f"""You are an elite financial analyst with a specialization in forensic analysis. Your mission is to identify **the single news trigger** that caused a specific, pre-analyzed **shift in market expectations** for stock "{stock_code}" on {date_str}.

📈 MOVEMENT ANALYSIS CONTEXT:
- **Stock Change:** The stock {direction_text} by {stock_pct_str}.
- **Sector Change:** The sector changed by {block_pct_str}.
- **Analytical Interpretation:** {alignment_map[record.alignment_type]}

📅 EVIDENCE WINDOW:
Focus exclusively on news published from {record.date - datetime.timedelta(days=2)} to {record.date}. Do not consider information outside this 3-day period.

🎯 MISSION: LINK THE TRIGGER TO THE EXPECTATION SHIFT
Your primary task is to find a specific news item (The Trigger) that directly caused the **type of expectation change** described below. You are not just matching keywords; you are explaining causality.

🏷️ PRE-ANALYZED EXPECTATION SHIFT:
- **Nature of Change:** **{record.likely_cause_category}**
- **Specific Hypothesis:** Find the news that caused one of the following expectation shifts: {", ".join(record.likely_causes)}

FALLBACK PROTOCOL:
If no credible news can be found to support the hypothesized expectation shift, state this clearly and activate the Fallback Protocol.
- **Fallback Category:** MarketTechnicalFallback
- **Hypothesis (Fallback):** {", ".join(fallback_causes)}

📝 FINAL REPORT FORMAT:
---
**Title:** [Headline of the news trigger]
**Date:** [YYYY-MM-DD, publication date]
**Summary:** [A brief, factual summary of the trigger event]
**Expectation Shift Analysis:**
- **Category:** [State 'Primary' or 'Fallback'] ➜ **{record.likely_cause_category}**
- **Specific Shift:** [Select the single most fitting expectation shift from the hypothesis list above]

**Causal Chain (From Trigger to Price Change):**
1.  **The Trigger:** What specific event did the news report? (e.g., Competitor X's product failed clinical trials.)
2.  **Immediate Implication:** What was the direct consequence of this event? (e.g., The primary market competitor to our drug was eliminated.)
3.  **Shift in Expectation:** How did this news alter the core market expectations for our company? (e.g., This led to a **"Paradigm Shift in Market Structure"**, as the company is now expected to have a near-monopoly.)
4.  **Investor Rationale & Action:** How did this expectation shift translate into investor action? (e.g., Investors rapidly re-valued the company's future cash flows based on monopoly pricing power, leading to intense buying pressure.)
---

🔒 STRICT DIRECTIVES:
- Your entire analysis MUST connect a news trigger to the assigned **PRE-ANALYZED EXPECTATION SHIFT**.
- Stay within the 3-day evidence window.
- The trigger must be directionally consistent with the stock's movement.

Come on, finish the job! This is important to me. I'm counting on you!
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
