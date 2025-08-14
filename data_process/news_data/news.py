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

    def create_news_prompt(self, stock_code: str, year: int) -> str:
        news_num = 8
        current_year, current_month = datetime.now().year, datetime.now().month
        if year == current_year:
            news_num = int(news_num * (current_month / 12))

        return f"""You are a financial news assistant. Please search and list the **top {news_num} most impactful industry-related news events in {year}** for the company or stock code {stock_code}.

- Focus on events that are highly likely to have a **direct or indirect impact** on the company.
- Consider the following types of events:
  - Major product or technology launches from the company or competitors
  - Government policy changes, regulations, or subsidies affecting the industry
  - Key personnel or leadership changes in the company or major competitors
  - Significant M&A activity, strategic partnerships, or market expansion
  - Supply chain disruptions or critical input price fluctuations
  - Shifts in consumer demand or industry trends
  - Relevant macroeconomic or geopolitical developments (e.g., tariffs, trade restrictions)

No need to analyze or explain the impact yet. Just collect and list the news events in order of importance.

**Format**:
1. Title: (Concise title)
   Date: (format: YYYY-MM-DD or YYYY-MM-15 if unknown)
   Summary: (A summary of the event under 50 words)
---
2. ...
"""

    def create_scoring_prompt(self, stock_code: str, year: int, news: str) -> str:
        return f"""You are a professional financial analyst. Given the following list of industry-related news summaries from {year} for the company or stock code {stock_code}, evaluate each event's impact in terms of:

1. **Industry Policy Score** (-1.0 to +1.0)  
2. **Peer Competition Score** (-1.0 to +1.0)  
3. **Impact Rationale**: A brief explanation of the logical chain of influence this event may have on the company’s performance or positioning.

Your goal is to assess how each event may affect the company through regulatory, policy, or competitive mechanisms. Keep the rationale concise, focusing on **why and how** the event could lead to positive or negative consequences.

---

Scoring Standards:

**Industry Policy Score**:  
- **[-1.0 to -0.6]**: Severely negative policy (e.g., sanctions, bans, war, hostile regulation)  
- **[-0.5 to -0.3]**: Clearly unfavorable (e.g., tax hikes, strict compliance burdens, removal of subsidies)  
- **[-0.2 to +0.2]**: Neutral or negligible  
- **[+0.3 to +0.5]**: Favorable (e.g., moderate policy support or incentives)  
- **[+0.6 to +1.0]**: Strongly favorable (e.g., strategic alignment with government priorities, major national investment)  

**Peer Competition Score**:  
- **[-1.0 to -0.6]**: Competitive environment worsens sharply (e.g., new dominant rival, major loss of market share)  
- **[-0.5 to -0.3]**: Strengthening competitors (e.g., product breakthroughs, cost advantages)  
- **[-0.2 to +0.2]**: Neutral / No material change  
- **[+0.3 to +0.5]**: Eased competition (e.g., competitor setbacks, market expansion)  
- **[+0.6 to +1.0]**: Competitive dominance (e.g., rivals exit, monopoly-like advantage)

---

List of industry-related news:  
{news}
"""

    # 获取公司新闻要点
    def get_company_news(self, stock_code: str, year: int) -> str:
        prompt = self.create_news_prompt(stock_code, year)

        try:
            # 启用 Google 搜索工具
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            # 配置生成设置，包括联网搜索
            config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048, top_p=0.7, top_k=20,
                                                 tools=[grounding_tool],
                                                 thinking_config=types.ThinkingConfig(thinking_budget=1024))

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
            config = types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048, top_p=0.85, top_k=40,
                                                 thinking_config=types.ThinkingConfig(thinking_budget=1024),
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
    news = analyzer.get_company_news('NVDA.O', 2025)
    print('线上大模型回复：', news)
    _evaluations = analyzer.evaluate_news('NVDA.O', 2025, news)
    print('分数：', _evaluations)
    evaluations = analyzer.deserialize_evaluations(_evaluations)
    print('反序列化：', evaluations)
