import os
import yaml
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import TypeAdapter
from utils.prompt import important_news_prompt, quantization_prompt, Evaluation, AttributionRecord

# 配置路径
base_dir = os.path.dirname(__file__)
feature_map_path = os.path.join(base_dir, "analyzer.yaml")


class ModelAnalyzer(ABC):
    """通用的模型分析器抽象基类"""

    MODEL_NAME: str = None  # 子类必须指定

    def __init__(self):
        if not self.MODEL_NAME:
            raise ValueError("子类必须指定 MODEL_NAME")

        # 读取 api_key.yaml
        if not os.path.exists(feature_map_path):
            raise FileNotFoundError(f"未找到配置文件: {feature_map_path}")

        with open(feature_map_path, "r", encoding="utf-8") as f:
            key_map = yaml.safe_load(f)

        api_key = key_map.get(self.MODEL_NAME)
        if not api_key:
            raise ValueError(f"未在 {feature_map_path} 中找到 {self.MODEL_NAME} 对应的 api key")

        self.api_key = api_key

    # -------- 子类必须实现的方法 --------

    @abstractmethod
    def create_client(self):
        """创建模型客户端"""
        pass

    @abstractmethod
    def request_important_news(self, prompt: str) -> str:
        pass

    @abstractmethod
    def request_news_quantization(self, prompt: str) -> str:
        pass

    # -------- 公共逻辑 --------
    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def check_title_count(self, text: str, context: str) -> None:
        """检查回复中是否包含至少一个 'title'"""
        count = text.lower().count("title")
        if count < 1:
            raise ValueError(f"[{context}] 未检测到 'title'，模型回复内容异常")

    def get_important_news(self, stock_code: str, record: AttributionRecord) -> Optional[str]:
        prompt = important_news_prompt(stock_code, record)
        try:
            response = self.request_important_news(prompt)
            self.check_title_count(response, f"{stock_code} 新闻检索")
            return response
        except Exception as e:
            print(f"API 调用失败: {e}")
            return None

    def get_news_quantization(self, stock_code: str, news: str) -> Optional[str]:
        if not news:
            raise ValueError("新闻为空，无法解析。")

        prompt = quantization_prompt(stock_code, news)
        try:
            response = self.request_news_quantization(prompt)
            return response
        except Exception as e:
            print(f"[ERROR] API 调用失败: {e}")
            return None

    def deserialize_evaluations(self, evaluations: str) -> List[Evaluation]:
        """把 JSON 字符串转为结构化对象"""
        try:
            adapter = TypeAdapter(List[Evaluation])
            return adapter.validate_json(evaluations)
        except Exception as e:
            print(f"反序列化失败: {e}")
            return []
