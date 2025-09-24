from google import genai
from google.genai import types
from utils.analyzer import NewsAnalyzer, retry, format_response, ResKind
from data_source.finance.script.wind import get_price_change_records
from data_source.finance.script.block_map import block_cache
from utils.prompt import get_analyse_records


class GeminiAnalyzer(NewsAnalyzer):
    """基于 Google Gemini 的财经新闻分析器"""

    MODEL_NAME: str = "Gemini"

    def __init__(self):
        super().__init__()
        self.client = self.create_client()

    # --------- 子类实现的抽象方法 ---------

    def create_client(self):
        """创建 Gemini 客户端"""
        if not self.api_key:
            raise ValueError("未找到 Gemini 的 API Key")
        return genai.Client(api_key=self.api_key)

    @retry()
    def request_important_news(self, prompt: str) -> str:
        """请求 Gemini 获取重要新闻"""
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            tools=[grounding_tool],
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        return response.text

    @retry()
    def request_related_news(self, prompt: str) -> str:
        """请求 Gemini 获取相关新闻"""
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            tools=[grounding_tool],
            thinking_config=types.ThinkingConfig(thinking_budget=128),
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        return response.text

    @retry()
    def request_news_quantization(self, prompt: str) -> str:
        """请求 Gemini 对新闻进行评分"""
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            tools=[grounding_tool],
            thinking_config=types.ThinkingConfig(thinking_budget=6144),
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        return response.text


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    analyzer = GeminiAnalyzer()

    # ------生成重要新闻------
    stock_code = "NVDA.O"
    block_code = "1000015222000000"
    start_year, end_year = 2023, 2023

    price_changes = get_price_change_records(stock_code, block_code, f"{start_year}-04-01", f"{end_year}-6-30")
    analyse_records = get_analyse_records(price_changes)
    print("重点分析目录：", analyse_records)

    news = analyzer.get_important_news(stock_code, analyse_records[0])
    print("Gemini 重要新闻：", news)

    # -------生成相关新闻-------
    # record = RelatedNewsRecord(
    #     year=2025,
    #     month=8,
    #     sector_name="Semiconductor products",
    #     core_stock_tickers=block_cache.get_stock_codes(Block.get("半导体产品").code).codes,
    # )
    # news = analyzer.get_related_news(record)
    # print("Gemini 相关新闻：", news)

    # -----生成新闻量化结果-----
    important_news = "Wall Street analysts and Nvidia's CEO, Jensen Huang, were dismissing the threat posed by the Chinese AI startup DeepSeek."
    date = "2025-01-28"
    response_text = analyzer.get_news_quantization("NVDA.O", important_news, date)
    print("Gemini 重要新闻分析结果：", response_text)
    format_res = format_response(response_text, ResKind.QUANT)
    print("Gemini 重要新闻量化结果：", format_res)

    # related_news = "TSMC Reports Strong January Revenue, Up 35.9% Year-over-Year, Despite Earthquake Impacting Q1 Outlook."
    # date = "2025-01-10"
    # response_text = analyzer.get_news_quantization("NVDA.O", related_news, date)
    # print("Gemini 相关新闻分析结果：", response_text)
    # format_res = format_response(response_text, ResKind.QUANT)
    # print("Gemini 相关新闻量化结果：", format_res)
