from pydantic import BaseModel
from typing import List, Optional, Literal
import datetime
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
from pydantic import TypeAdapter


def deserialize(json_str: str, model: Any) -> Any:
    adapter = TypeAdapter(model)
    try:
        return adapter.validate_json(json_str)
    except Exception as e:
        raise ValueError(f"JSON 解析失败: {e}") from e


@dataclass
class CausalChain(BaseModel):
    trigger: str
    implication: str
    expectation: str
    investor_rationale: str


@dataclass
class ImportantNews(BaseModel):
    title: str
    date: str
    summary: str
    category: str
    specific_shift: str
    causal_chain: CausalChain


@dataclass
class RelatedNews(BaseModel):
    title: str
    date: str


@dataclass
class QuantitativeScores:
    causal_impact_score: float
    uncertainty_score: float
    alpha_score: float
    power_shift_score: float
    sentiment_score: float
    time_horizon_fundamental: float
    time_horizon_sentiment: float
    conviction_score: float


@dataclass
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


# --------------------------------------------------------------------------------------------------------------------


# 生成重要新闻的prompt
def important_news_prompt(stock_code: str, record: AttributionRecord) -> str:
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


# --------------------------------------------------------------------------------------------------------------------


@dataclass
class RelatedNewsRecord:
    year: int
    month: int
    sector_name: str
    core_stock_tickers: List[str]


# 生成相关新闻的prompt
def related_news_prompt(record: RelatedNewsRecord) -> str:
    return f"""You are a top-tier AI industry analyst and market intelligence expert, skilled at rapidly identifying and extracting key events from vast amounts of information that have significant impact on a specific industry landscape.

# Task
Your task is: For a given industry sector, within a specified year and month, use your web-searching and analytical capabilities to collect and filter **10 most critical news events** related to the sector’s ecosystem.

# Input Information
- **Year**: {record.year}
- **Month**: {record.month}
- **Sector Name**: {record.sector_name}
- **Core Stock Tickers**: {record.core_stock_tickers}

# Output Requirements

1. **Number of Events**: Exactly 10 news items, no more, no less.  
2. **Format**: Each news item must include the following two fields and strictly follow this format:  
   - **Event Title**: [Concise summary of the event, ≤25 words]  
   - **Date**: YYYY-MM-DD  
3. **Content Requirements**:  
   - **a. Highly Relevant**: All events must be highly relevant to the specified sector.  
   - **b. Key Constraint**: Among the 10 news items, **at least 2** must clearly relate to the sector’s **upstream or downstream** companies.  
   - **c. Ecosystem Perspective**: Events do not need to explicitly mention the “core stock tickers,” but should revolve around the ecosystem in which these companies operate.  
   - **d. Strict Date Requirement**: **Only include news with a clearly verifiable and specific publication date within the given Year and Month (YYYY-MM-DD). Discard any events with vague or missing dates (e.g., “recently,” “last week,” “in Q1”).**


# Workflow & Thought Process

1. **Understand the Ecosystem**: First, based on the "Sector Name" and "Core Stock Tickers," quickly construct an internal map of the sector's ecosystem.  
2. **Define Upstream & Downstream**: Clearly identify what constitutes upstream and downstream for this sector.  
   - Example (Semiconductors):  
     - **Upstream**: Equipment suppliers (ASML), Materials suppliers (Shin-Etsu), EDA software providers (SNPS)  
     - **Downstream**: Consumer electronics (AAPL), Automotive manufacturers (TSLA), Data centers (AMZN, GOOG)  
3. **Perform Search**: Within the specified "Year" and "Month," search using keywords related to the entire ecosystem (including upstream and downstream). Focus on major earnings releases, M&A activity, technological breakthroughs, supply chain changes, macro policy impacts, significant contracts/orders, etc.  
4. **Filter & Rank**: From search results, rank events based on their **importance and impact** on the industry. Initially select ~15-20 candidate news items.  
   - **Eliminate any events without an exact publication date (YYYY-MM-DD) in the specified Year and Month.**  
   - Ensure at least 2 upstream/downstream events remain in the final list.  
5. **Ensure Constraints**: Among candidates, prioritize at least 2 upstream/downstream news items. Then, select the most important and influential core sector news to reach exactly 10 items. If upstream/downstream events are insufficient, replace lower-impact core stock news to meet the constraint.  
6. **Format Output**: Present the final 10 news items strictly following the “Output Requirements” format.

---

**Please begin the task and answer in English.**
    """


# --------------------------------------------------------------------------------------------------------------------


# 生成量化分析的prompt
def quantization_prompt(stock_code: str, news: str) -> str:
    return f"""You are a top-tier equity strategist who blends deep fundamental analysis with real-time market intelligence. Your process involves verifying a core event, analyzing its strategic impact on the industry ecosystem, gauging the surrounding public sentiment, and then synthesizing these inputs into a robust, quantitative assessment for predictive models.

### MISSION BRIEF
- **Stock:** {stock_code}
- **Event / Topic:** {news}

---

### ANALYTICAL WORKFLOW (SOP)
Execute the following 5-phase workflow.

**Phase 1: Intelligence Gathering & Verification**
- Use your search capabilities to investigate the specified Event/Topic.
- **Objective 1 (Fact-Finding):** Find 1-2 primary news sources to establish the core facts.
- **Objective 2 (Sentiment-Gauging):** Broaden your search to include financial news commentary, forums, and social media to understand the public and investor reaction.

**Phase 2: Strategic Ecosystem Analysis (Fundamental & Rational View)**
- **Ecosystem Context:** Briefly map the company's position, key competitors, and value chain.
- **Impact Vectors:** Analyze the event's Horizontal (peer comparison) and Vertical (value chain) impact.

**Phase 3: Market Sentiment Analysis (Market & Emotional View)**
- **Summarize the Narrative:** What is the dominant story the market is telling about this event?
- **Gauge the Tone:** Is the overall sentiment positive, negative, or mixed? Is it driven by hype, fear, or rational analysis?

**Phase 4: Synthesized Causal Chain & Mismatch Detection**
- **Synthesize:** Integrate your strategic analysis (Phase 2) and market sentiment (Phase 3) to construct the final causal chain using the 4-step structure.
- **Detect Mismatch:** **Crucially, if the direction of the fundamental impact (causal_impact_score) and the market sentiment (sentiment_score) are opposed, you must explicitly flag this mismatch.**

**Phase 5: Final Quantification**
- Based on your complete analysis, provide all scores for the framework below.

---

### FINAL REPORT (Strictly JSON format)
{{
  "intelligence_summary": {{
    "verified_event_summary": "A concise, factual summary of the core event.",
    "market_sentiment_summary": "A summary of the prevailing narrative and emotional tone from public/media discourse."
  }},
  "analytical_synthesis": {{
    "sentiment_fundamental_mismatch": {{
      "is_mismatch": "[true or false]",
      "description": "If true, briefly describe the nature of the mismatch (e.g., 'Market sentiment is highly negative due to headline risk, but the underlying fundamental impact appears neutral to slightly positive.')."
    }}
  }},
  "impact_classification": {{
    "primary_driver": "[Demand_Shock | Supply_Shock | Regulatory_Shock | Competitive_Shock | Operational_Shock | Tech_Shock]",
    "strategic_consequence": "[TAM_Change | Market_Share_Change | Cost_Structure_Change | Pricing_Power_Change | Other]"
  }},
  "causal_chain": {{
    "trigger_event": "The specific factual event.",
    "ecosystem_impact": "How the competitive and value chain dynamics are altered.",
    "shift_in_core_expectation": "The core long-term belief about the company that has now changed.",
    "ultimate_financial_consequence": "The final predicted impact on financials and valuation."
  }},
  "quantitative_scores": {{
    "causal_impact_score": {{
      "value": "[-10 to +10]",
      "description": "The analyst's assessment of the event's fundamental, rational impact on the company's long-term value."
    }},
    "uncertainty_score": {{
      "value": "[1-10]",
      "description": "Degree of uncertainty introduced. 1=Clear Outcome, 10=High Uncertainty."
    }},
    "alpha_score": {{
      "value": "[1-10]",
      "description": "Company-specific (alpha) vs sector-wide (beta) impact."
    }},
    "power_shift_score": {{
      "value": "[-5 to +5]",
      "description": "Shift in bargaining power along the value chain."
    }},
    "sentiment_score": {{
        "value": "[-10 to +10]",
        "description": "A direct measure of the prevailing emotional tone in public/media discourse. -10=Panic/Fear, +10=Hype/Euphoria."
    }},
    "time_horizon_fundamental": {{
      "value": "[1, 2, or 3]",
      "description": "Expected duration of the *fundamental* impact: 1=Short (<3M), 2=Medium (3-12M), 3=Long (>1Y)."
    }},
    "time_horizon_sentiment": {{
      "value": "[1, 2, or 3]",
      "description": "Expected duration of the *sentiment* impact: 1=Short (<1M), 2=Medium (1-3M), 3=Long (>3M)."
    }},
    "conviction_score": {{
        "value": "[1-10]",
        "description": "Analyst's confidence in the overall analysis. 1=Speculative, 10=High-Conviction."
    }}
  }}
}}
"""
