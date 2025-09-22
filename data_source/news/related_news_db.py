import os
import sqlite3
import re
from datetime import datetime
import time
from typing import List
from data_source.news.script.gemini import GeminiAnalyzer
from data_source.finance.script.wind import get_stock_codes
from utils.prompt import RelatedNewsRecord, related_news_prompt
from utils.block import Block
import utils.prompt as pt
from utils.analyzer import format_response, ResKind

start_year = 2025


class RelatedNewsDBManager:
    """
    封装对板块相关新闻数据的 sqlite 管理逻辑。
    每个板块名称(sector_name)对应一个数据库文件。
    数据库中的每张表对应一个年份。
    """

    def __init__(self, sector_name_cn: str, db_dir: str = "related_news_db"):
        self.analyzer = GeminiAnalyzer()

        self.sector_name_cn = sector_name_cn
        self.sector_name_en = None
        self.sector_description = None
        self.core_stock_tickers = None

        self.db_dir = os.path.join(os.path.dirname(__file__), db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

        # 数据库文件名直接使用板块中文名
        self.db_file = os.path.join(self.db_dir, f"{self._format_db_name(self.sector_name_cn)}.db")

        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

        # 初始化值
        self._init_values()
        # 初始化数据库，如果不存在
        self._init_db_settings()
        # auto_update 会自动处理表创建和数据更新
        self.auto_update(start_year=start_year)

    def _format_db_name(self, name: str) -> str:
        """格式化名称为合法的文件名，移除非法字符"""
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    def _format_year_table_name(self, year: int) -> str:
        """将年份格式化为合法的 SQLite 表名，例如 "Y_2024" """
        return f'"Y_{year}"'

    def _init_values(self):
        """初始化板块描述和核心股票列表"""
        item = Block.get(self.sector_name_cn)
        if item:
            self.sector_name_en = item.name_en
            self.sector_description = item.description_en
            self.core_tickers = get_stock_codes(item.id)
        else:
            raise ValueError(f"板块 {self.sector_name_cn} 不存在")

    def _init_db_settings(self):
        """配置 SQLite 写入策略"""
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def ensure_table_exists(self, year: int):
        """确保对应年份的表存在，表结构以月份为主键"""
        table_name = self._format_year_table_name(year)
        query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                month INTEGER PRIMARY KEY,
                news_data TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        '''
        self.cursor.execute(query)
        self.conn.commit()

    def save_related_news(self, year: int, month: int, news_text: str):
        """将新闻列表（序列化为JSON）保存或更新到指定年份的表中"""
        if not news_text:
            print(f"[WARN] 新闻列表为空，跳过写入: {self.sector_name_cn} {year}-{month:02d}")
            return

        # 先确保表存在
        self.ensure_table_exists(year)
        table_name = self._format_year_table_name(year)

        try:
            last_updated = datetime.now().isoformat()

            query = f'''
                INSERT OR REPLACE INTO {table_name} (month, news_data, last_updated)
                VALUES (?, ?, ?)
            '''
            self.cursor.execute(query, (month, news_text, last_updated))
            self.conn.commit()
            print(f"[INFO] 已写入 {self.sector_name_cn} 板块 {year}-{month:02d} 的相关新闻")

        except Exception as e:
            print(f"[ERROR] 写入数据库失败 ({self.db_file} -> {table_name}): {e}")
            self.conn.rollback()

    def get_related_news(self, year: int, month: int) -> List[pt.RelatedNews]:
        """根据年月获取新闻列表，会自动处理表不存在的情况"""
        table_name = self._format_year_table_name(year)

        try:
            self.cursor.execute(f"SELECT news_data FROM {table_name} WHERE month = ?", (month,))
            row = self.cursor.fetchone()
            if row and row[0]:
                return pt.deserialize(row[0], List[pt.RelatedNews])
            else:
                return []
        except sqlite3.OperationalError:
            # 如果表不存在，会触发此异常，说明肯定没有数据
            return []
        except Exception as e:
            print(f"[ERROR] 读取数据库失败 ({self.db_file} -> {table_name}): {e}")
            return []

    def auto_update(self, start_year: int = 2023):
        """自动更新从指定年份至今所有缺失月份的新闻数据"""
        print(f"[INFO] 开始为板块 [{self.sector_name_cn}] 自动更新新闻数据...")
        now = datetime.now()

        for year in range(start_year, now.year + 1):
            end_month = now.month if year == now.year else 12
            for month in range(1, end_month + 1):
                if self.get_related_news(year, month):
                    # print(f"[DEBUG] 数据已存在，跳过: {self.sector_name_cn} {year}-{month:02d}")
                    continue

                print(f"[INFO] 正在获取 {self.sector_name_cn} 板块 {year}-{month:02d} 的新闻...")
                try:
                    record = RelatedNewsRecord(year, month, self.sector_name_en, self.sector_description,
                                               self.core_stock_tickers)
                    response_text = self.analyzer.get_related_news(record)
                    format_res = format_response(response_text, ResKind.REL)
                    self.save_related_news(year, month, format_res)
                    time.sleep(1)

                except Exception as e:
                    print(f"[ERROR] 在处理 {self.sector_name_cn} {year}-{month:02d} 时发生错误: {e}")
                    self.conn.rollback()

    def __del__(self):
        """析构时自动释放资源"""
        try:
            if self.conn:
                self.conn.commit()
                self.conn.close()
        except Exception:
            pass


def create_related_news_db(parent_name: str):
    """
    为指定板块分类下的所有子板块创建或更新其新闻数据库。
    每个子板块一个独立的 .db 文件。
    """
    sub_items = Block.get_items_by_parent(parent_name)
    if not sub_items:
        print(f"[WARN] 未找到板块分类 '{parent_name}' 下的子板块")
        return

    for _, item in sub_items.items():
        print(f"\n===== [INFO] 开始处理板块: {item.name_cn} =====")
        core_tickers = get_stock_codes(item.id)
        if not core_tickers:
            print(f"[WARN] 板块 {item.name_cn} 未找到核心股票，跳过")
            continue

        # 实例化管理器，它将自动创建库、表并填充数据
        # 注意这里传入的是 item.desc (中文名) 作为数据库标识
        RelatedNewsDBManager(sector_name=item.name_cn)
        print(f"===== [INFO] 板块 {item.name_cn} 新闻更新完成 =====\n")


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    RelatedNewsDBManager("半导体产品")
    # create_related_news_db("SP500_WIND行业类")
