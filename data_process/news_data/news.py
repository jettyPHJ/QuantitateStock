import os
from google import genai
from google.genai import types
from datetime import datetime
from pydantic import BaseModel
from typing import List
from pydantic import TypeAdapter


class Evaluation(BaseModel):
    title: str
    date: str
    industry_policy_score: float
    peer_competition_score: float
    reason: str


class GeminiFinanceAnalyzer:

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("环境变量 'GEMINI_API_KEY' 未设置")
        self.client = genai.Client()

    def create_news_prompt(self, stock_code: str, year: int, month: int) -> str:
        return f"""You are a top-tier financial analyst. Your task is to identify and analyze the top 3 most influential news from {month}_{year} that significantly impacted the stock price of "{stock_code}".

📌 OBJECTIVE
From all news items published in the given month, select exactly 3 that had the largest actual impact on the stock's price. Selection must strictly follow the Impact Hierarchy below — if fewer than 3 events exist in higher tiers, fill remaining slots from lower tiers.

🏗️ IMPACT HIERARCHY (from most direct to least):
1. Industry & Policy  
   - Industry-level disruptions (e.g., supply chain crisis, competitive threats)
   - Regulatory actions, subsidies, investigations, antitrust moves

2. Peer Competition  
   - Product launches or failures by the company or its key competitors
   - Major strategic moves by the company or its key competitors (M&A, price wars, leadership changes) that directly affect competitive landscape

3. Market & Sentiment  
   - Influential analyst rating changes or price targets
   - Reputable short-seller reports or major media investigations

4. Macroeconomic / Geopolitical  
   - Events like CPI shocks, rate changes, global conflict (only if clearly linked to the company)

🧠 RULES FOR SELECTION:
Step 1: Review all news items published in {month}_{year}. Discard any with negligible or no price impact.
Step 2: Merge duplicates or ongoing threads into one if they represent the same price-moving event.
Step 3: Sort remaining events by Impact Hierarchy first, then by magnitude of actual price reaction (largest absolute % move on the same or next trading day).
Step 4: Select the top 3 after sorting. If fewer than 3 exist, still output exactly 3 by including the next highest tier.

📝 OUTPUT FORMAT:
Return exactly 3 items using this structure:

---
**Title:** [Headline of the event]  
**Date:** [YYYY-MM-DD or YYYY-MM-15 if unknown]  
**Summary:** [Concise and factual summary of what happened]  
**Impact Dimension:** [Choose one from: Company Fundamentals / Industry & Policy / Market & Sentiment / Macro & Geopolitics] 
**Observed Price Move:** [% price change] 
**Impact Analysis:** [Describe clearly how this caused stock price movement. Example: “Industry-wide chip shortage worsened → Raised ASPs across peers → Investors revised growth outlook upward → Stock rose.”]
---

🔒 BOUNDARY CONDITIONS:
- Only use news published in {month}_{year} (publication date, not actual event occurrence date).
- Avoid vague, speculative, or unverified information.
- Precision and causality are more important than coverage
"""

    def create_scoring_prompt(self, stock_code: str, year: int, news: str) -> str:
        return f"""You are a professional financial analyst. For each of the following news events from {year} related to the company or stock code "{stock_code}", please assess the impact **from two independent perspectives**:

1. **Industry Policy Impact** (range: -1.0 to +1.0)  
2. **Peer Competition Impact** (range: -1.0 to +1.0)

Also provide a short, factual explanation for each score, strictly focusing on **how this event could affect the company through industry policy or competitive pressure**.

🧠 **Key Instructions**:
- Keep the explanation concise (1-2 sentences per dimension).
- **Do not generalize or speculate beyond the content of the event.**
- **Only assign strong scores (≥ |0.6|)** when the impact is **clear, material, and direct**.
- Evaluate each dimension **separately**, even if the event has no effect on one of them.

🎯 **Scoring Standards**:

**Industry Policy Impact (Regulatory / Subsidy / Macroeconomic)**  
- `+0.6 to +1.0`: Major favorable policy (e.g., heavy national investment, strategic alignment)  
- `+0.3 to +0.5`: Mildly favorable policy or macro tailwind  
- `-0.2 to +0.2`: Neutral / negligible / indirect  
- `-0.3 to -0.5`: Policy headwind (e.g., regulation, reduced support)  
- `-0.6 to -1.0`: Hostile or damaging policy (e.g., sanctions, exclusion, trade war)

**Peer Competition Impact (Market Positioning / Rival Actions)**  
- `+0.6 to +1.0`: Significant competitive gain (e.g., monopoly, rivals fail)  
- `+0.3 to +0.5`: Moderate gain (e.g., rivals delay product, company expands)  
- `-0.2 to +0.2`: Neutral / status quo  
- `-0.3 to -0.5`: Moderate loss (e.g., new entrant, rival product launch)  
- `-0.6 to -1.0`: Major loss (e.g., competitor dominance, pricing war)

---

List of industry-related news:  
{news}
"""

    # 获取公司新闻要点
    def get_company_news(self, stock_code: str, year: int, month: int) -> str:
        prompt = self.create_news_prompt(stock_code, year, month)

        try:
            # 启用 Google 搜索工具
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            # 配置生成设置，包括联网搜索
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                top_p=0.7,
                tools=[grounding_tool],
                thinking_config=types.ThinkingConfig(thinking_budget=128),
            )

            # 发送请求
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            return response.text

        except Exception as e:
            print(f"API 调用失败: {e}")
        return None

    # 评估新闻并输出结构化数据的字符串
    def evaluate_news(self, stock_code: str, year: int, news: str) -> str:
        if not news:
            raise ValueError("新闻为空，无法解析。")

        prompt = self.create_scoring_prompt(stock_code, year, news)

        try:
            # 配置生成设置
            config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048, top_p=0.3,
                                                 thinking_config=types.ThinkingConfig(thinking_budget=256),
                                                 response_mime_type="application/json",
                                                 response_schema=list[Evaluation])

            # 发送请求
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=config)
            return response.text

        except Exception as e:
            print(f"API 调用失败: {e}")
        return None

    # 加载得到结构化数据
    def deserialize_evaluations(self, evaluations: str) -> List[Evaluation]:
        try:
            adapter = TypeAdapter(List[Evaluation])
            return adapter.validate_json(evaluations)  # 直接传入原始字符串
        except Exception as e:
            print(f"反序列化失败: {e}")
            return []


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = GeminiFinanceAnalyzer()
    # 获取新闻评分
    news = analyzer.get_company_news('NVDA.O', 2025, 4)
    print('线上大模型回复：', news)
    # _evaluations = analyzer.evaluate_news('NVDA.O', 2025, news)
    # print('分数：', _evaluations)
    # evaluations = analyzer.deserialize_evaluations(_evaluations)
    # print('反序列化：', evaluations)
