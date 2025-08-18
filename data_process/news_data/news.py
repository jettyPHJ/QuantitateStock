import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
from pydantic import TypeAdapter
from data_process.finance_data.wind import get_pct_chg
import calendar


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
        last_day = calendar.monthrange(year, month)[1]
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{last_day:02d}"
        price_changes = get_pct_chg(stock_code, start_date, end_date)
        # 生成每日涨跌幅表格
        daily_changes_str = "\n".join(f"{date}: {change:+.2f}%" for date, change in price_changes)
        return f"""You are a top-tier financial analyst. Your task is to identify and analyze the top 3 most influential news from {month}_{year} that significantly impacted the stock price of "{stock_code}".

📈 HISTORICAL PRICE MOVES:
Here is the actual daily % change in closing price for {stock_code} in {month}_{year}:
{daily_changes_str}

📌 OBJECTIVE:
From all news items published during {month}_{year}, select exactly 3 that had a **clear and significant impact** on the stock’s price. Use the daily price changes above to identify and validate potential causal links between news and price movement.

🏗️ IMPACT DIMENSIONS (for reference, not strict sorting):
1. Industry & Policy  
   - Industry-level disruptions (e.g., supply chain crisis, competitive threats)  
   - Regulatory actions, subsidies, investigations, antitrust moves

2. Peer Competition  
   - Product launches or failures by the company or its key competitors  
   - Major strategic moves by the company or competitors (M&A, price wars, leadership changes)

3. Market & Sentiment  
   - Influential analyst rating changes or price targets  
   - Short-seller reports or major media investigations

4. Macro & Geopolitics  
   - Events like CPI shocks, rate changes, global conflict (only if clearly linked to the company)

🧠 GUIDANCE FOR SELECTION:
- Prioritize events that clearly explain large price swings (same or next trading day).
- Do not strictly follow the impact dimension order — use them as reference categories.
- Merge duplicate or ongoing news threads into a single summarized item.
- If fewer than 3 strong-impact events exist, still output 3 by including the most relevant remaining ones.

📝 OUTPUT FORMAT:
Return exactly 3 items using this structure:

---
**Title:** [Headline of the event]  
**Date:** [YYYY-MM-DD]  
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
                temperature=0.0,
                max_output_tokens=2048,
                top_p=0.8,
                tools=[grounding_tool],
                thinking_config=types.ThinkingConfig(thinking_budget=128),
            )

            # 发送请求
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
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
    news = analyzer.get_company_news('NVDA.O', 2025, 3)
    print('线上大模型回复：', news)
    # _evaluations = analyzer.evaluate_news('NVDA.O', 2025, news)
    # print('分数：', _evaluations)
    # evaluations = analyzer.deserialize_evaluations(_evaluations)
    # print('反序列化：', evaluations)
