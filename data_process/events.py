import sqlite3
import generate_news
import json
import time

class NewsDBManager:
    def __init__(self, db_path="events.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # 使用默认的 DELETE 模式（非 WAL）
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        # 强同步策略，确保每次写操作都落盘
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def __del__(self):
        self.conn.commit()
        self.conn.close()

    def _get_table_name(self, stock_code):
        return stock_code.replace(".", "")

    def ensure_table_exists(self, stock_code):
        table_name = self._get_table_name(stock_code)
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER,
                quarter TEXT,
                news TEXT,
                gemini_embedding TEXT
            )
        ''')

    def fetch_news_from_db(self, stock_code, year, quarter):
        table_name = self._get_table_name(stock_code)
        try:
            # 检查表是否存在
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not self.cursor.fetchone():
                return None, None

            # 查询数据
            self.cursor.execute(f'''
                SELECT news, gemini_embedding FROM {table_name}
                WHERE year = ? AND quarter = ?
                LIMIT 1
            ''', (year, quarter))
            row = self.cursor.fetchone()
            if row:
                news_text = row[0]
                embedding = json.loads(row[1])
                return news_text, embedding
        except Exception as e:
            print(f"查询数据库失败：{e}")
        return None, None

    def save_news_to_db(self, stock_code, year, quarter, news_text, embedding):
        table_name = self._get_table_name(stock_code)
        self.ensure_table_exists(stock_code)

        try:
            self.cursor.execute(f'''
                INSERT INTO {table_name} (year, quarter, news, gemini_embedding)
                VALUES (?, ?, ?, ?)
            ''', (
                year,
                quarter,
                news_text.strip(),
                json.dumps(embedding)
            ))
            print(f"成功写入 {stock_code} 的事件到数据库")
        except Exception as e:
            print(f"写入数据库失败：{e}")
        self.conn.commit()


def get_news_and_embedding(stock_code: str, quarter: str, year: int, db_manager: NewsDBManager):
    # 尝试从数据库获取
    news_text, embedding = db_manager.fetch_news_from_db(stock_code, year, quarter)
    if news_text and embedding:
        print(f"已在数据库中找到 {stock_code} {quarter} {year} 的记录，直接返回")
        return news_text, embedding

    # 否则调用 API
    print(f"正在获取 {stock_code} 在 {quarter} {year} 的新闻...")
    time.sleep(2)
    try:
        news_text = generate_news.analyzer.get_company_news(stock_code, quarter, year)
        if not news_text:
            print("新闻获取失败")
            return None, None

        raw = generate_news.analyzer.get_embedding(news_text)
        embedding = getattr(raw[0], "values", raw[0]) if raw else None

        if not isinstance(embedding, list):
            print("embedding 获取失败")
            return news_text, None

        db_manager.save_news_to_db(stock_code, year, quarter, news_text, embedding)
        return news_text, embedding

    except Exception as e:
        print(f"调用 API 异常：{e}")
        return None, None

# db = NewsDBManager("events.db")
# text, emb = get_news_and_embedding("AAPL.O", "Q2", 2025, db)