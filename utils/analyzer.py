import os
import yaml
import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Callable
import utils.prompt as pt
from google import genai
from google.genai import types
from functools import wraps


def retry(retries=3, delay=2):
    """
    通用重试装饰器
    1. 如果捕获到异常信息包含 "overloaded"，延迟时间加倍
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay  # 局部延迟变量
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries:
                        # 如果异常信息包含 "overloaded"，延迟翻倍
                        if "overloaded" in str(e).lower():
                            current_delay *= 2
                        time.sleep(current_delay)
                    else:
                        raise

        return wrapper

    return decorator


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
        """类内装饰器：统一异常处理 + 强制 JSON 格式校验"""

        def wrapper(self, *args, **kwargs):
            response = func(self, *args, **kwargs)
            # 强制检查是否为合法 JSON，但返回原始字符串
            try:
                json.loads(response)  # 仅校验
                return response  # 返回原始字符串
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
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=config,
        )

        return response.text

    @retry()
    def format_important_news(self, text: str) -> str:
        return self._analyze(text, pt.ImportantNews)

    @retry()
    def format_related_news(self, text: str) -> str:
        return self._analyze(text, list[pt.RelatedNews])

    @retry()
    def format_quantization(self, text: str) -> str:
        return self._analyze(text, pt.QuantitativeScores)


# 实例化一个助手，全局使用，专门负责格式化文本
Assistant = AssistantAnalyzer()


class ResKind(Enum):
    IMP = "important-news"
    REL = "related-news"
    QUANT = "quantization-news"


# --- 模型回复文本格式化函数 ---
def format_response(response_text: str, kind: ResKind) -> str:
    """
    统一格式化函数
    """
    format_map: Dict[ResKind, Callable[[str], str]] = {
        ResKind.IMP: Assistant.format_important_news,
        ResKind.REL: Assistant.format_related_news,
        ResKind.QUANT: Assistant.format_quantization,
    }

    try:
        return format_map[kind](response_text)
    except Exception as e:
        fname = getattr(format_map[kind], "__name__", str(format_map[kind]))
        raise ValueError(f"[WARN] {fname} 调用失败: {e}") from e


class NewsAnalyzer(ABC):
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

    # ---------------------- 公共逻辑 ---------------------

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    # --- 通用请求 ---
    def _request(self, prompt: str, request_func: Callable[[str], str]) -> str:
        try:
            return request_func(prompt)
        except Exception as e:
            fname = getattr(request_func, "__name__", str(request_func))
            raise ValueError(f"[WARN] {fname} 调用失败: {e}") from e

    # --- 获取模型回复文本 ---
    def get_important_news(self, stock_code: str, record: pt.AttributionRecord) -> str:
        prompt = pt.important_news_prompt(stock_code, record)
        return self._request(prompt, self.request_important_news)

    def get_related_news(self, record: pt.RelatedNewsRecord) -> str:
        prompt = pt.related_news_prompt(record)
        return self._request(prompt, self.request_related_news)

    def get_news_quantization(self, stock_code: str, news_title: str, date: str) -> str:
        if not news_title:
            raise ValueError("新闻标题为空，无法解析。")
        prompt = pt.quantization_prompt(stock_code, news_title, date)
        return self._request(prompt, self.request_news_quantization)


class CompanyAnalyzer(ABC):
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
