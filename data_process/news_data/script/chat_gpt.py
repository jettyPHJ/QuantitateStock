from openai import OpenAI
from data_process.finance_data.script.wind import get_price_change_records
from utils.prompt import get_analyse_records
from utils.analyzer import NewsAnalyzer


class ChatGPTAnalyzer(NewsAnalyzer):
    """基于 OpenAI ChatGPT 的财经新闻分析器"""

    MODEL_NAME: str = "ChatGPT"

    def __init__(self):
        super().__init__()
        self.client = self.create_client()

    # --------- 子类实现的抽象方法 ---------

    def create_client(self):
        """创建 OpenAI 客户端"""
        if not self.api_key:
            raise ValueError("未找到 OpenAI 的 API Key")
        return OpenAI(api_key=self.api_key)

    def request_important_news(self, prompt: str) -> str:
        """请求 ChatGPT 获取新闻要点"""
        # 注意：像 gpt-4o 这样的模型内置了网页浏览能力，无需像 Gemini 那样显式配置 grounding_tool。
        # 模型会根据 prompt 的内容决定是否需要联网搜索。
        response = self.client.chat.completions.create(
            model="gpt-4o",  # 推荐使用支持联网和强大理解能力的模型
            messages=[{"role": "system", "content": "You are a helpful financial news assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def request_news_quantization(self, prompt: str) -> str:
        """请求 ChatGPT 对新闻进行评分 (JSON格式)"""
        # 为了获得稳定的 JSON 输出，需要在 prompt 中明确指示 ChatGPT 生成 JSON，并启用 JSON 模式。
        # prompt 中应包含类似 "Please respond in JSON format that conforms to the specified schema." 的指令。
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system", "content": "You are an assistant that provides financial analysis in JSON format."
            }, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}  # 启用 JSON 模式
        )
        return response.choices[0].message.content


# --------------------- 测试入口---------------------
if __name__ == "__main__":
    analyzer = ChatGPTAnalyzer()

    stock_code = "9988.HK"
    block_code = "1000069991000000"
    year = 2025

    price_changes = get_price_change_records(stock_code, block_code, f"{year}-08-25", f"{year}-12-31")
    analyse_records = get_analyse_records(price_changes)

    # 假设 analyse_records 不为空
    if analyse_records:
        news = analyzer.get_important_news(stock_code, analyse_records[0])
        print("ChatGPT 新闻：", news)

        # evaluations_str = analyzer.evaluate_news(stock_code, 2025, 3, news)
        # print("ChatGPT 评分：", evaluations_str)

        # evaluations = analyzer.deserialize_evaluations(evaluations_str)
        # print("反序列化结果：", evaluations)
    else:
        print("未找到价格变动记录，无法进行分析。")
