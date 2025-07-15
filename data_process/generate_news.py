import os
from typing import Dict, List
from google import genai
from google.genai import types

API_KEY = os.getenv('GEMINI_API_KEY')  # 要提前配置 GOOGLE_API_KEY  密钥

class GeminiFinanceAnalyzer:
    def __init__(self, api_key: str):
        """
        初始化 Gemini API 客户端     
        Args:
            api_key: Google Gemini API 密钥
        """
        # 设置 API 密钥
        os.environ['GEMINI_API_KEY'] = api_key
        self.client = genai.Client()
        
    def create_prompt(self, stock_code: str, quarter: str, year: int) -> str:
        """
        创建分析提示词   
        Args:
            stock_code: 股票代码 (如 NVDA.O)
            quarter: 季度 (如 Q2)
            year: 年份 (如 2021) 
        Returns:
            格式化的提示词
        """
        # 根据季度计算月份范围
        quarter_months = {
            "Q1": "January–March",
            "Q2": "April–June", 
            "Q3": "July–September",
            "Q4": "October–December"
        }
        
        months = quarter_months.get(quarter, "April–June")
        
        prompt = f"""You are a professional financial news reporter. Your task is to identify and summarize significant public news events related to a specific company during a given period. Focus strictly on reporting factual occurrences based on verifiable news. Avoid making direct conclusions about the impact on stock price or using speculative language.

Stock code: {stock_code}
Period: {year} {months}

Requirements:
- Output must be in English and written as plain text in a clear key-value format.
- Must contain one positive news event and one negative news event.
- Each event should be written in one sentence under 100 words.
- Ignore trivial, repetitive, or irrelevant information.
- Descriptions must be strictly factual, objective, and free from emotional, promotional, or speculative language. Do not infer direct stock price causality.
- The output should be suitable for semantic embedding as an input to a stock price prediction model.
Example Format:
  "Positive": "(description...under 100 words)",
  "Negative": "(description...under 100 words)"
Please begin the summary."""
        
        return prompt
    
    def call_gemini_api(self, prompt: str) -> str:
  
        try:
            # 启用 Google 搜索工具
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            # 配置生成设置，包括联网搜索
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                top_p=0.8,
                top_k=40,
                tools=[grounding_tool],
                thinking_config=types.ThinkingConfig(thinking_budget=5)
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

    def get_embedding(self, text: str) :
        """
        使用 Gemini 模型获取文本的语义向量（embedding）
        """
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            return result.embeddings
        except Exception as e:
            print(f"嵌入调用失败: {e}")
            return []

    def get_company_news(self, stock_code: str, quarter: str, year: int) -> List[Dict[str, str]]:
        """
        获取公司新闻要点
        Args:
            stock_code: 股票代码
            quarter: 季度
            year: 年份 
        Returns:
            模型回应文本
        """
        prompt = self.create_prompt(stock_code, quarter, year)
        response_text = self.call_gemini_api(prompt)
        return response_text


# 创建分析器实例
analyzer = GeminiFinanceAnalyzer(API_KEY)
# 获取新闻
# response_text = analyzer.get_company_news('APPL.O', 'Q2', 2025)
# embedding = analyzer.get_embedding(response_text)
# print('特征向量：',embedding)
