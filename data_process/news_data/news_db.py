import os
import sqlite3
from data_process.news_data.script.news import GeminiFinanceAnalyzer
import re
from utils.prompt import AttributionRecord
from data_process.finance_data.script.wind import get_price_change_records, get_stock_codes
from utils.block import Block
from utils.prompt import get_analyse_records
from datetime import datetime


class NewsDBManager:
    """
    封装对单个股票新闻评分数据的 sqlite 管理逻辑。
    """

    def __init__(self, block_code: str, stock_code: str, db_dir: str = "news_db"):
        self.analyzer = GeminiFinanceAnalyzer()

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
        self.auto_update()

    def _extract_db_char(self, stock_code: str) -> str:
        """提取股票代码中首个非0的字符(数字或字母)"""
        for c in stock_code:
            if c != '0' and c.isalnum():
                return c.upper()
        return "OTHER"

    def _format_table_name(self, stock_code: str) -> str:
        """将股票代码格式化为合法的 SQLite 表名（去掉 .-)"""
        stock_code = stock_code.strip().replace(".", "_").replace("-", "_").upper()
        if not re.match(r"^[A-Z0-9_]+$", stock_code):
            raise ValueError(f"Invalid stock code format: {stock_code}")
        return stock_code

    def _init_db_settings(self):
        """配置 SQLite 写入策略"""
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def ensure_table_exists(self):
        """仅创建基础字段"""
        if not re.match(r'^[a-zA-Z0-9_]+$', self.table_name):
            raise ValueError("Invalid table name")

        query = f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                stock_pct_chg REAL NOT NULL,
                block_pct_chg REAL NOT NULL
            )
        '''
        self.cursor.execute(query)
        self.conn.commit()

    def _ensure_model_field_exists(self, model_name: str):
        """检查是否存在对应的大模型字段，若无则添加"""
        field_name = f"{model_name.lower()}_news"

        # 获取已有字段名
        self.cursor.execute(f"PRAGMA table_info({self.table_name})")
        existing_fields = {row[1] for row in self.cursor.fetchall()}

        if field_name not in existing_fields:
            alter_query = f'''
                ALTER TABLE {self.table_name}
                ADD COLUMN {field_name} TEXT
            '''
            self.cursor.execute(alter_query)
            self.conn.commit()
            print(f"[INFO] 添加新字段 {field_name} 到表 {self.table_name}")

    def save_news(self, record: AttributionRecord, news_text: str, model_name: str):
        """保存大模型输出，如果是首次调用该模型则自动添加字段"""
        try:
            field_name = f"{model_name.lower()}_news"
            self._ensure_model_field_exists(model_name)

            news_text = news_text.strip()
            if not news_text:
                print(f"[WARN] 空新闻内容，跳过写入: {record.date}")
                return

            # 检查记录是否已存在（按唯一 date）
            self.cursor.execute(f"SELECT id FROM {self.table_name} WHERE date = ?", (record.date.isoformat(),))
            row = self.cursor.fetchone()

            if row:
                # 存在就更新对应字段
                self.cursor.execute(
                    f'''
                    UPDATE {self.table_name}
                    SET {field_name} = ?
                    WHERE date = ?
                    ''', (news_text, record.date.isoformat()))
            else:
                # 不存在就插入，其他字段为 NULL
                self.cursor.execute(
                    f'''
                    INSERT INTO {self.table_name} (date, stock_pct_chg, block_pct_chg, {field_name})
                    VALUES (?, ?, ?, ?)
                    ''', (record.date.isoformat(), record.stock_pct_chg, record.block_pct_chg, news_text))

            self.conn.commit()
            print(f"[INFO] 已写入 {self.stock_code}_{record.date} 数据 - {model_name}")
        except Exception as e:
            print(f"[ERROR] 写入数据库失败: {e}")
            self.conn.rollback()

    def get_news(self, date: str, model_name: str) -> str:
        """根据日期和大模型名称获取新闻内容"""
        field_name = f"{model_name.lower()}_news"

        # 检查字段是否存在
        self.cursor.execute(f"PRAGMA table_info({self.table_name})")
        existing_fields = {row[1] for row in self.cursor.fetchall()}

        if field_name not in existing_fields:
            print(f"[WARN] 字段 {field_name} 不存在于表 {self.table_name} 中")
            return ""

        self.cursor.execute(f"SELECT {field_name} FROM {self.table_name} WHERE date = ?", (date,))
        row = self.cursor.fetchone()
        if row and row[0]:
            return row[0]
        else:
            print(f"[INFO] {date} 无 {model_name} 新闻记录")
            return ""

    def auto_update(self):
        """自动更新所有缺失的大模型新闻数据(2005~当前年)"""
        current_year = datetime.now().year
        model_name = self.analyzer.get_model_name()
        field_name = f"{model_name.lower()}_news"

        # 检查字段是否存在，如果没有就添加
        self.cursor.execute(f"PRAGMA table_info({self.table_name})")
        existing_fields = {row[1] for row in self.cursor.fetchall()}
        if field_name not in existing_fields:
            alter_sql = f"ALTER TABLE {self.table_name} ADD COLUMN {field_name} TEXT"
            self.cursor.execute(alter_sql)
            self.conn.commit()
            print(f"[INFO] 已添加新字段: {field_name}")

        for year in range(2005, current_year + 1):
            print(f"[INFO] 正在处理 {year} 年的数据...")
            price_changes = get_price_change_records(self.stock_code, self.block_code, f"{year}-01-01", f"{year}-12-31")
            analyse_records = get_analyse_records(price_changes)
            if len(analyse_records) == 0:
                print(f"[INFO]  {self.stock_code} 在 {year} 年的股价平稳，未达到分析要求...")
                continue
            for record in analyse_records:
                date_str = str(record.date)
                # 检查该日期是否已有该模型字段的数据
                self.cursor.execute(f"SELECT {field_name} FROM {self.table_name} WHERE date = ?", (date_str,))
                row = self.cursor.fetchone()

                if row and row[0]:
                    continue  # 已存在则跳过

                try:
                    news = self.analyzer.get_company_news(self.stock_code, record)
                    self.save_news(record, news, model_name)
                except Exception as e:
                    print(f"[ERROR] {self.stock_code}_{date_str} 生成或写入失败: {e}")
                    self.conn.rollback()

    def __del__(self):
        """析构时自动释放资源"""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass


# 生成板块配置文件的子结构对应股票的新闻数据
def create_news_db(entry_key: str):
    """
    创建一个包含指定板块所有股票新闻数据的 sqlite 数据库。
    """
    sub_items = Block.get_sub_items(entry_key)
    for _, item in sub_items.items():
        print(f"[INFO] 开始搜寻 {item.desc} 板块的股票新闻")
        stock_codes = get_stock_codes(item.code)
        if len(stock_codes) == 0:
            print(f"[INFO] {item.desc} 板块没有股票")
            continue
        for stock_code in stock_codes:
            NewsDBManager(item.code, stock_code)
            print(f"[INFO] {item.desc} 板块_{stock_code} 新闻收集完成")


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    create_news_db("SP500_WIND行业类")
