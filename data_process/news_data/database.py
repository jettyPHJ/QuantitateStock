import os
import sqlite3
import json
from typing import Optional, List, Dict, Any
import data_process.news_data.news as gn
import time


class NewsDBManager:
    """
    封装对单个股票新闻评分数据的 sqlite 管理逻辑。
    """

    def __init__(self, stock_code: str, db_dir: str = "db"):
        self.news_manager = gn.GeminiFinanceAnalyzer()

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

    def _extract_db_char(self, stock_code: str) -> str:
        """提取股票代码中首个非0的字符（数字或字母）"""
        for c in stock_code:
            if c != '0' and c.isalnum():
                return c.upper()
        return "OTHER"

    def _format_table_name(self, stock_code: str) -> str:
        """将股票代码格式化为合法的 SQLite 表名（去掉 .）"""
        return stock_code.replace(".", "").upper()

    def _init_db_settings(self):
        """配置 SQLite 写入策略"""
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def ensure_table_exists(self):
        """确保该股票代码的评分表存在"""
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER,
                news TEXT,
                scores TEXT
            )
        ''')
        self.conn.commit()

    def save_news_with_scores(self, year: int, news_text: str, scores: list):
        """
        插入大模型输出和结构化评分数据
        """
        try:
            scores_json = json.dumps(scores, ensure_ascii=False)
            self.cursor.execute(
                f'''
                INSERT INTO {self.table_name} (year, news, scores)
                VALUES (?, ?, ?)
                ''', (year, news_text.strip(), scores_json))
            self.conn.commit()
            print(f"[INFO] 已写入 {self.stock_code}_{year} 年的评分数据")
        except Exception as e:
            print(f"[ERROR] 写入数据库失败: {e}")
            self.conn.rollback()  # 写入失败时回滚事务

    def get_or_generate_scores(self, year: int) -> Optional[List[Dict[str, Any]]]:
        """
        核心方法：先从数据库查询评分，如果没有则通过 Gemini API 生成并存储。
        """
        # 1. 尝试从数据库查询，直接将逻辑内联
        try:
            self.cursor.execute(
                f'''SELECT scores FROM {self.table_name}
                                    WHERE year = ? LIMIT 1''', (year,))
            row = self.cursor.fetchone()
            if row:
                print(f"[INFO] 从数据库获取 {self.stock_code}_{year} 年的评分数据")
                return json.loads(row[0])
        except Exception as e:
            print(f"[ERROR] 查询数据库失败: {e}")

        # 2. 数据库中没有，调用 Gemini API 生成
        print(f"[INFO] 数据库中未找到 {self.stock_code}_{year} 年数据，正在调用大模型生成...")
        max_attempts = 3  # 总共尝试2次
        for attempt in range(max_attempts):
            try:
                print(f"[INFO] 正在调用大模型... (第 {attempt + 1}/{max_attempts} 次尝试)")

                # 2a. 调用大模型获取原始文本
                response_text = self.news_manager.get_company_news(self.stock_code, year)
                if not response_text:
                    # 这种情况是API调用成功，但返回空内容，也应视为一种“失败”
                    raise ValueError("大模型未返回任何有效文本内容。")

                # 2b. 解析返回的文本
                print("[INFO] 模型返回数据成功，正在解析分数...")
                generated_scores = self.news_manager.get_scores(response_text)

                # 2c. 将新生成的数据存储到数据库
                print(f"[INFO] 解析成功，正在将 {self.stock_code} 在 {year} 年的新数据存入数据库...")
                self.save_news_with_scores(year, response_text, generated_scores)

                # 如果所有步骤都成功，返回结果并跳出循环
                return generated_scores

            except Exception as e:
                print(f"[ERROR] 第 {attempt + 1} 次尝试失败: {e}")
                # 如果不是最后一次尝试，则等待一小段时间后重试
                if attempt < max_attempts - 1:
                    wait_time = 2
                    print(f"[INFO] 等待 {wait_time} 秒后进行最后一次尝试...")
                    time.sleep(wait_time)
                else:
                    # 所有尝试都失败了
                    print(f"[CRITICAL] 所有 {max_attempts} 次尝试均告失败，无法为 {year} 年生成评分数据。")
                    return None

    def __del__(self):
        """析构时自动释放资源"""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    # 创建数据库实例
    news_db = NewsDBManager(stock_code="NVDA.O")
    results = news_db.get_or_generate_scores(2025)
    print("results: ", results)
