import re
import json
import os
from typing import Dict, List
from google import genai
from google.genai import types


# 配置参数
STOCK_CODE = "AAPL.O"
QUARTER = "Q2"
YEAR = 2025

API_KEY = os.getenv('GEMINI_API_KEY')  # 要提前配置 Gemini API 密钥

class GeminiFinanceAnalyzer:
    def __init__(self, api_key: str):
        """
        初始化 Gemini API 客户端     
        Args:
            api_key: Google Gemini API 密钥
        """
        # 设置 API 密钥
        os.environ['GOOGLE_API_KEY'] = api_key
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
        
        prompt = f"""You are a professional financial news reporter. Your task is to identify and summarize significant public news events related to a specific company during a given quarter. Focus strictly on reporting factual occurrences based on verifiable news. Avoid making direct conclusions about the impact on stock price or using speculative language.

Stock code: {stock_code}
Quarter: {quarter} {year} ({months})

Requirements:
- Output must be in English and in JSON format.
- Identify one distinct publicly reported event generally perceived as positive news for the company, and one distinct publicly reported event generally perceived as negative news for the company. (Total 2 events).
- Each item must contain:
    - event_type: "Positive" or "Negative"
    - event_description: A clear, objective summary of the news event itself, in under 100 words. Focus on what happened, not its presumed stock price effect.
- Ignore trivial, repetitive, or irrelevant information.
- Descriptions must be strictly factual, objective, and free from emotional, promotional, or speculative language. Do not infer direct stock price causality.
- The output should be suitable for semantic embedding as an input to a stock price prediction model.

Example Format:
[
  {{
    "event_type": "Positive",
    "event_description": "The company announced the launch of its new flagship product, receiving positive initial reviews from technology critics."
  }},
  {{
    "event_type": "Negative",
    "event_description": "Regulatory authorities in Country X initiated an investigation into the company's past accounting practices."
  }}
]

Please begin."""
        
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
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40,
                tools=[grounding_tool],
                thinking_config=types.ThinkingConfig(thinking_budget=0)
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
    
    def extract_json_from_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        从 API 响应中提取 JSON 数据
        Args:
            response_text: Gemini API 响应文本
        Returns:
            提取的事件列表
        """
        if not response_text:
            print("无效的 API 响应")
            return []
        
        try:
            # 尝试解析 JSON（可能包含在代码块中）
            match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if not match:
                raise ValueError("未找到 JSON 内容")

            json_str = match.group(1).strip()
            
            # 解析 JSON
            events = json.loads(json_str)

            # 清除每个 event_description 中的末尾引用标签
            for e in events:
                e["event_description"] = re.sub(r"\[\d+(,\s*\d+)*\]\s*$", "", e["event_description"]).strip()

            return events
            
        except json.JSONDecodeError as e:
            print(f"解析响应失败: {e}")
            print(f"原始响应: {response_text}")
            return []
    
    def analyze_company_news(self, stock_code: str, quarter: str, year: int) -> List[Dict[str, str]]:
        """
        分析公司新闻要点
        
        Args:
            stock_code: 股票代码
            quarter: 季度
            year: 年份
            
        Returns:
            事件列表
        """
        print(f"正在分析 {stock_code} 在 {quarter} {year} 的新闻要点...")
        
        # 创建提示词
        prompt = self.create_prompt(stock_code, quarter, year)
        
        # 调用 API
        response_text = self.call_gemini_api(prompt)
        
        if not response_text:
            return []
        
        # 提取结果
        events = self.extract_json_from_response(response_text)
        
        print(f"成功提取到 {len(events)} 个事件")
        return events
    
    def save_results(self, events: List[Dict[str, str]], filename: str):
        """
        保存结果到文件
        
        Args:
            events: 事件列表
            filename: 保存的文件名
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {filename}")


# 创建分析器实例
analyzer = GeminiFinanceAnalyzer(API_KEY)

# 分析新闻
events = analyzer.analyze_company_news(STOCK_CODE, QUARTER, YEAR)

if events:
    # 打印结果
    print("\n=== 分析结果 ===")
    for i, event in enumerate(events, 1):
        print(f"{i}. 类型: {event.get('event_type', 'Unknown')}")
        print(f"   描述: {event.get('event_description', 'No description')}")
        print()
    
    # 保存结果
    filename = f"{STOCK_CODE}_{QUARTER}_{YEAR}_news_summary.json"
    analyzer.save_results(events, filename)
else:
    print("未能获取到有效的分析结果")
