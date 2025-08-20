import os
from google import genai
from google.genai import types
from typing import List, Optional
from pydantic import TypeAdapter
from data_process.finance_data.wind import get_price_change_records
from utils.prompt import news_prompt, scoring_prompt, get_analyse_records, Evaluation, AttributionRecord


class GeminiFinanceAnalyzer:

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("环境变量 'GEMINI_API_KEY' 未设置")
        self.client = genai.Client()

    def create_news_prompt(self, stock_code: str, record: AttributionRecord) -> str:
        return news_prompt(stock_code, record)

    def create_scoring_prompt(self, stock_code: str, year: int, month: int, news: str) -> str:
        return scoring_prompt(stock_code, year, month, news)

    # 回话检查函数
    def check_title_count(self, text: str, context: str) -> None:
        count = text.lower().count("title")
        if count < 1:
            raise ValueError(f"[{context}] 未检测到 'title'，模型回复内容异常")

    # 获取公司新闻要点
    def get_company_news(self, block_code: str, stock_code: str, year: int) -> str:
        price_changes = get_price_change_records(stock_code, block_code, f"{year}-08-01", f"{year}-12-31")
        analyse_records = get_analyse_records(price_changes)

        for record in analyse_records:

            prompt = self.create_news_prompt(stock_code, record)

            try:
                # 启用 Google 搜索工具
                grounding_tool = types.Tool(google_search=types.GoogleSearch())

                # 配置生成设置，包括联网搜索
                config = types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                    tools=[grounding_tool],
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                )

                # 发送请求
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config,
                )

                self.check_title_count(response.text, f"{stock_code}-{year} 新闻检索")
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
            config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048,
                                                 thinking_config=types.ThinkingConfig(thinking_budget=512),
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
    news = analyzer.get_company_news('1000041891000000', 'NVDA.O', 2024)
    print('线上大模型回复：', news)
    # _evaluations = analyzer.evaluate_news('NVDA.O', 2025, 3, news)
    # print('分数：', _evaluations)
    # evaluations = analyzer.deserialize_evaluations(_evaluations)
    # print('反序列化：', evaluations)
