import os
import yaml
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import TypeAdapter
import utils.prompt as pt
from google import genai
from google.genai import types

# 配置路径
base_dir = os.path.dirname(__file__)
feature_map_path = os.path.join(base_dir, "analyzer.yaml")
# 读取配置文件
if not os.path.exists(feature_map_path):
    raise FileNotFoundError(f"未找到配置文件: {feature_map_path}")
with open(feature_map_path, "r", encoding="utf-8") as f:
    key_map: Dict[str, Any] = yaml.safe_load(f)


class AssistantAnalyzer:

    def __init__(self):
        self.api_key = key_map.get("Gmini-lite")
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _safe_analyze(func):
        """类内装饰器：统一异常处理 + 强制 JSON 解析"""

        def wrapper(self, *args, **kwargs):
            response = func(self, *args, **kwargs)
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(f"Model output is not valid JSON: {response}") from e
            except Exception as e:
                raise RuntimeError(f"Gemini analysis failed: {e}") from e

        return wrapper

    @_safe_analyze
    def _analyze(self, text: str, response_schema) -> str:
        """
        通用 Gemini 分析调用
        """
        prompt = f"""
        You are an intelligent assistant.
        Please analyze the following text and provide your response strictly in given format.

        Text:
        {text}
        """

        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=config,
        )

        return response.text

    def format_important_news(self, text: str) -> str:
        return self._analyze(text, pt.ImportantNews)

    def format_related_news(self, text: str) -> str:
        return self._analyze(text, list[pt.RelatedNews])

    def format_quantization(self, text: str) -> str:
        return self._analyze(text, pt.QuantitativeScores)


# 实例化一个助手，全局使用，专门负责格式化文本
Assistant = AssistantAnalyzer()


class ModelAnalyzer(ABC):
    """通用的模型分析器抽象基类"""

    MODEL_NAME: str = None  # 子类必须指定

    def __init__(self):
        if not self.MODEL_NAME:
            raise ValueError("子类必须指定 MODEL_NAME")

        api_key = key_map.get(self.MODEL_NAME)
        if not api_key:
            raise ValueError(f"未在 {feature_map_path} 中找到 {self.MODEL_NAME} 对应的 api key")

        self.api_key = api_key

    # ----------------- 子类必须实现的方法 ------------------

    @abstractmethod
    def create_client(self):
        """创建模型客户端"""
        pass

    @abstractmethod
    def request_important_news(self, prompt: str) -> str:
        pass

    @abstractmethod
    def request_related_news(self, prompt: str) -> str:
        pass

    @abstractmethod
    def request_news_quantization(self, prompt: str) -> str:
        pass

    # -------------------- 公共逻辑 --------------------------

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def get_important_news(self, stock_code: str, record: pt.AttributionRecord) -> Optional[str]:
        prompt = pt.important_news_prompt(stock_code, record)
        try:
            response = self.request_important_news(prompt)
            return response
        except Exception as e:
            print(f"API 调用失败: {e}")
            return None

    def get_related_news(self, record: pt.RelatedNewsRecord) -> Optional[str]:
        prompt = pt.related_news_prompt(record)
        try:
            response = self.request_important_news(prompt)
            return response
        except Exception as e:
            print(f"API 调用失败: {e}")
            return None

    def get_news_quantization(self, stock_code: str, news: str) -> Optional[str]:
        if not news:
            raise ValueError("新闻为空，无法解析。")

        prompt = pt.quantization_prompt(stock_code, news)
        try:
            response = self.request_news_quantization(prompt)
            return response
        except Exception as e:
            print(f"[ERROR] API 调用失败: {e}")
            return None

    def deserialize_evaluations(self, evaluations: str) -> List[pt.Evaluation]:
        """把 JSON 字符串转为结构化对象"""
        try:
            adapter = TypeAdapter(List[pt.Evaluation])
            return adapter.validate_json(evaluations)
        except Exception as e:
            print(f"反序列化失败: {e}")
            return []
