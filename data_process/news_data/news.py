import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
from pydantic import TypeAdapter
from data_process.finance_data.wind import get_pct_chg
import calendar


class Evaluation(BaseModel):
    title: str
    date: str
    industry_policy_score: float
    peer_competition_score: float
    market_sentiment_score: float
    macro_geopolitics_score: float
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
        return f"""You are a top-tier financial analyst. Your task is to identify and analyze the top 2 most influential news from {month}_{year} that significantly impacted the stock price of "{stock_code}".

📈 HISTORICAL PRICE MOVES:
Here is the actual daily % change in closing price for {stock_code} in {month}_{year}:
{daily_changes_str}

📌 OBJECTIVE:
From all news items published during {month}_{year}, select exactly 2 that had a **clear and significant impact** on the stock’s price. Use the daily price changes above to identify and validate potential causal links between news and price movement.

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
- Prioritize events that clearly explain large price swings (Match each news item to the largest same-day or next-day price change that logically aligns).
- Do not strictly follow the impact dimension order — use them as reference categories.
- Merge duplicate or ongoing news threads into a single summarized item.
- If fewer than 2 strong-impact events exist, still output 2 by including the most relevant remaining ones.

📝 OUTPUT FORMAT:
Return exactly 2 items using this structure:

---
**Title:** [Headline of the event]  
**Date:** [The exact news publication date in format YYYY-MM-DD only. Example: 2024-03-12]  
**Summary:** [Concise and factual summary of what happened]  
**Impact Dimension:** [Choose one from: Industry & Policy / Peer Competition / Market & Sentiment / Macro & Geopolitics] 
**Observed Price Move:** [% price change] 
**Impact Analysis:** [Describe clearly how this caused stock price movement. Example: “Industry-wide chip shortage worsened → Raised ASPs across peers → Investors revised growth outlook upward → Stock rose.”]
---

🔒 BOUNDARY CONDITIONS:
- Only use news published in {month}_{year} (publication date, not actual event occurrence date).
- Avoid vague, speculative, or unverified information.
- Precision and causality are more important than coverage
"""

    def create_scoring_prompt(self, stock_code: str, year: int, month: int, news: str) -> str:
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

    # 回话检查函数
    def check_title_count(self, text: str, context: str) -> None:
        count = text.lower().count("title")
        if count < 2:
            raise ValueError(f"[{context}] 检测到的 'title' 数量不足 2（实际数量为 {count}）")

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
                top_p=0.6,
                tools=[grounding_tool],
                thinking_config=types.ThinkingConfig(thinking_budget=128),
            )

            # 发送请求
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )

            self.check_title_count(response.text, f"{stock_code}-{year}_{month} 新闻检索")
            return response.text

        except Exception as e:
            print(f"API 调用失败: {e}")
        return None

    # 评估新闻并输出结构化数据的字符串
    def evaluate_news(self, stock_code: str, year: int, month: int, news: str) -> Optional[str]:
        """
        对新闻文本执行评分并输出结构化数据（字符串格式）
        """

        if not news:
            raise ValueError("新闻为空，无法解析。")

        self.check_title_count(news, f"{stock_code}-{year}_{month} 原始新闻")

        prompt = self.create_scoring_prompt(stock_code, year, month, news)

        try:
            config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048, top_p=0.8,
                                                 thinking_config=types.ThinkingConfig(thinking_budget=256),
                                                 response_mime_type="application/json",
                                                 response_schema=list[Evaluation])

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )

            self.check_title_count(response.text, f"{stock_code}-{year}_{month} 模型响应")

            return response.text

        except Exception as e:
            print(f"[ERROR] API 调用失败: {e}")
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
    _evaluations = analyzer.evaluate_news('NVDA.O', 2025, 3, news)
    print('分数：', _evaluations)
    # evaluations = analyzer.deserialize_evaluations(_evaluations)
    # print('反序列化：', evaluations)
