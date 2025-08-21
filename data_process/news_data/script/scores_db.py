import os
import sqlite3
from typing import List, Dict
from data_process.news_data.script.news import GeminiFinanceAnalyzer
import time
from datetime import datetime, timedelta
import math
import re
from utils.prompt import Evaluation, AttributionRecord


class ScoresDBManager:
    """
    封装对单个股票新闻评分数据的 sqlite 管理逻辑。
    """

    def __init__(self, block_code: str, stock_code: str, db_dir: str = "db/news"):
        self.news_manager = GeminiFinanceAnalyzer()

        self.block_code = block_code
        self.stock_code = stock_code
        self.db_dir = os.path.join(os.path.dirname(__file__), db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

        db_char = self._extract_db_char(stock_code)
        self.db_file = os.path.join(self.db_dir, f"{db_char}.db")
        self.table_name = self._format_table_name(stock_code)

        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

        self._init_db_settings()
        self.ensure_table_exists()

    def get_evaluations(self, year: int, month: int) -> List[Evaluation]:
        """
            核心方法：先从数据库查询评分，如果没有则通过 Gemini API 生成并存储。
            """
        # 1. 尝试从数据库查询，直接将逻辑内联
        try:
            self.cursor.execute(
                f'''SELECT evaluations FROM {self.table_name}
                        WHERE year = ? AND month = ? LIMIT 1''', (year, month))
            row = self.cursor.fetchone()
            if row:
                print(f"[INFO] 从数据库获取 {self.stock_code}_{year}_{month} 的评分数据")
                evaluations = self.news_manager.deserialize_evaluations(row[0])
                return evaluations
        except Exception as e:
            print(f"[ERROR] 查询数据库失败: {e}")

        # 2. 数据库中没有，调用 Gemini API 生成
        print(f"[INFO] 数据库中未找到 {self.stock_code}_{year}_{month} 的评分数据，开始调用大模型...")
        max_attempts = 3  # 总共尝试3次
        for attempt in range(max_attempts):
            try:
                print(f"[INFO] 正在调用大模型抓取新闻... (第 {attempt + 1}/{max_attempts} 次尝试)")

                # 2a. 调用大模型获取原始文本
                response_text = self.news_manager.get_company_news(self.stock_code, year, month)
                if not response_text:
                    # 这种情况是API调用成功，但返回空内容，也应视为一种“失败”
                    raise ValueError("大模型未返回任何新闻内容。")

                # 2b. 调用大模型获取评估
                print("[INFO] 模型返回新闻成功，进行评估中...")
                evaluations_json = self.news_manager.evaluate_news(self.stock_code, year, month, response_text)

                # 2c. 将新生成的数据存储到数据库
                print(f"[INFO] 评分完成，正在将 {self.stock_code} 在 {year}_{month} 的新数据存入数据库...")
                self.save_news_with_evaluations(year, month, response_text, evaluations_json)

                # 如果所有步骤都成功，返回结果并跳出循环
                evaluations = self.news_manager.deserialize_evaluations(evaluations_json)
                return evaluations

            except Exception as e:
                print(f"[ERROR] 第 {attempt + 1} 次尝试失败: {e}")
                # 如果不是最后一次尝试，则等待一小段时间后重试
                if attempt < max_attempts - 1:
                    wait_time = 2
                    print(f"[INFO] 等待 {wait_time} 秒后再次尝试...")
                    time.sleep(wait_time)
                else:
                    print(f"[CRITICAL] 所有 {max_attempts} 次尝试均告失败，无法为 {year} 年生成数据。")
                    return None

    def __del__(self):
        """析构时自动释放资源"""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass


def exponential_decay(days_passed: int, half_life: int = 7) -> float:
    """
    指数衰减函数：影响随时间呈指数衰减，half_life 表示半衰期（多少天影响减半）
    """
    if days_passed < 0:
        return 0.0
    λ = math.log(2) / half_life  # 控制衰减速度
    return math.exp(-λ * days_passed)


def compute_scores(news_items: List[Evaluation], start_date: str, end_date: str, decay_days: int = 7,
                   score_fields: List[str] = None) -> Dict[str, float]:
    """
    计算[start_date, end_date]内，结合news_items中新闻
    （考虑前decay_days天内的新闻事件）对区间综合评分。
    每条新闻对其发生后decay_days天内有影响，影响逐日递减。
    支持任意评分字段，如["industry_policy_score", "peer_competition_score", ...]
    """
    if not news_items:
        return {}

    if score_fields is None:
        score_fields = [
            "industry_policy_score",
            "peer_competition_score",
            "market_sentiment_score",
            "macro_geopolitics_score",
        ]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    window_start = start - timedelta(days=decay_days)
    total_days = (end - start).days + 1

    # 初始化每天的累计权重和评分
    daily_scores = {field: [0.0] * total_days for field in score_fields}
    daily_weights = [0.0] * total_days

    for item in news_items:
        news_date = datetime.strptime(item.date, "%Y-%m-%d")
        if not (window_start <= news_date <= end):
            continue

        for day_offset in range(total_days):
            current_day = start + timedelta(days=day_offset)
            if current_day < news_date or current_day > news_date + timedelta(days=decay_days):
                continue
            days_since_news = (current_day - news_date).days
            weight = exponential_decay(days_since_news)
            if weight <= 0:
                continue

            for field in score_fields:
                value = getattr(item, field, 0.0)
                daily_scores[field][day_offset] += value * weight
            daily_weights[day_offset] += weight

    total_weight = sum(daily_weights)
    if total_weight == 0:
        return {field.replace("_score", ""): 0.0 for field in score_fields}

    # 计算加权平均分
    return {field.replace("_score", ""): sum(daily_scores[field]) / total_weight for field in score_fields}
