import os
import sqlite3
import re
import json
from datetime import datetime
import time
from typing import List, Dict
from data_process.news_data.script.gemini import GeminiAnalyzer, get_stock_codes
from utils.prompt import RelatedNewsRecord, related_news_prompt
from utils.block import Block


class RelatedNewsDBManager:
    """
    封装对板块相关新闻数据的 sqlite 管理逻辑。
    每个大的板块分类（entry_key）对应一个数据库文件。
    每个子板块（sector）对应数据库中的一张表。
    """

    def __init__(self, entry_key: str, sector_code: str, sector_name: str, core_stock_tickers: List[str],
                 db_dir: str = "related_news_db"):
        self.analyzer = GeminiAnalyzer()

        self.entry_key = entry_key
        self.sector_code = sector_code
        self.sector_name = sector_name
        self.core_stock_tickers = core_stock_tickers

        self.db_dir = os.path.join(os.path.dirname(__file__), db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

        self.db_file = os.path.join(self.db_dir, f"{self._format_db_name(self.entry_key)}.db")
        self.table_name = self._format_table_name(self.sector_code)

        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

        self._init_db_settings()
        self.ensure_table_exists()
        self.auto_update()

    def _format_db_name(self, entry_key: str) -> str:
        """格式化entry_key为合法的文件名"""
        return re.sub(r'[\\/*?:"<>|]', "_", entry_key)

    def _format_table_name(self, code: str) -> str:
        """将板块代码格式化为合法的 SQLite 表名"""
        code = code.strip().replace(".", "_").replace("-", "_").upper()
        if not re.match(r"^[A-Z0-9_]+$", code):
            raise ValueError(f"不合法的板块代码格式: {code}")
        return f'"{code}"'

    def _init_db_settings(self):
        """配置 SQLite 写入策略"""
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")

    def ensure_table_exists(self):
        """确保表存在，以年月为联合唯一键"""
        query = f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                news_data TEXT,
                last_updated TEXT NOT NULL,
                UNIQUE(year, month)
            )
        '''
        self.cursor.execute(query)
        self.conn.commit()

    def _parse_news_from_response(self, response: str) -> List[Dict[str, str]]:
        """从模型的原始文本输出中解析新闻列表"""
        news_list = []
        # 使用正则表达式匹配标题和日期
        pattern = re.compile(r"-\s*Event Title:\s*(.*?)\s*-\s*Date:\s*(\d{4}-\d{2}-\d{2})", re.DOTALL)
        matches = pattern.findall(response)

        for title, date in matches:
            news_list.append({"title": title.strip(), "date": date.strip()})
        return news_list

    def save_related_news(self, year: int, month: int, news_list: List[Dict[str, str]]):
        """将新闻列表（序列化为JSON）保存或更新到数据库"""
        try:
            if not news_list:
                print(f"[WARN] 新闻列表为空，跳过写入: {self.sector_name} {year}-{month:02d}")
                return

            news_json = json.dumps(news_list, ensure_ascii=False, indent=2)
            last_updated = datetime.now().isoformat()

            # 使用 INSERT OR REPLACE 来简化插入或更新逻辑
            query = f'''
                INSERT OR REPLACE INTO {self.table_name} (year, month, news_data, last_updated)
                VALUES (?, ?, ?, ?)
            '''
            self.cursor.execute(query, (year, month, news_json, last_updated))
            self.conn.commit()
            print(f"[INFO] 已写入 {self.sector_name} {year}-{month:02d} 的相关新闻")

        except Exception as e:
            print(f"[ERROR] 写入数据库失败: {e}")
            self.conn.rollback()

    def get_related_news(self, year: int, month: int) -> List[Dict[str, str]]:
        """根据年月获取新闻列表"""
        try:
            self.cursor.execute(f"SELECT news_data FROM {self.table_name} WHERE year = ? AND month = ?", (year, month))
            row = self.cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            else:
                return []
        except Exception as e:
            print(f"[ERROR] 读取数据库失败: {e}")
            return []

    def auto_update(self, start_year: int = 2005):
        """自动更新从指定年份至今所有缺失月份的新闻数据"""
        print(f"[INFO] 开始为板块 [{self.sector_name}] 自动更新新闻数据...")
        now = datetime.now()

        for year in range(start_year, now.year + 1):
            # 月份循环到当前月份
            end_month = now.month if year == now.year else 12
            for month in range(1, end_month + 1):
                # 检查该月份数据是否已存在
                if self.get_related_news(year, month):
                    continue

                print(f"[INFO] 正在为 {self.sector_name} 获取 {year}-{month:02d} 的新闻...")
                try:
                    # 1. 创建记录并生成prompt
                    record = RelatedNewsRecord(year, month, self.sector_name, self.core_stock_tickers)

                    # 2. 调用模型获取新闻
                    response_text = self.analyzer.get_related_news(record)

                    # 3. 解析模型输出
                    news_items = self._parse_news_from_response(response_text)

                    # 4. 保存到数据库
                    self.save_related_news(year, month, news_items)

                    time.sleep(1)  # 避免请求过于频繁

                except Exception as e:
                    print(f"[ERROR] 在处理 {year}-{month:02d} 时发生错误: {e}")
                    self.conn.rollback()

    def __del__(self):
        """析构时自动释放资源"""
        try:
            if self.conn:
                self.conn.commit()
                self.conn.close()
        except Exception as e:
            # 在程序退出时可能会遇到已关闭的连接，忽略错误
            pass


def create_related_news_db(entry_key: str):
    """
    创建一个包含指定板块分类下所有子板块新闻数据的 sqlite 数据库。
    """
    sub_items = Block.get_sub_items(entry_key)
    if not sub_items:
        print(f"[WARN] 未找到板块分类 '{entry_key}' 下的子板块")
        return

    for _, item in sub_items.items():
        print(f"\n===== [INFO] 开始处理板块: {item.desc} ({item.code}) =====")
        # 获取该板块的核心股票作为生态圈定位
        core_tickers = get_stock_codes(item.code)
        if not core_tickers:
            print(f"[WARN] 板块 {item.desc} 未找到核心股票，跳过")
            continue

        # 实例化DB管理器，将自动执行数据更新
        RelatedNewsDBManager(entry_key=entry_key, sector_code=item.code, sector_name=item.name,
                             core_stock_tickers=core_tickers)
        print(f"===== [INFO] 板块 {item.desc} 新闻更新完成 =====\n")


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    # 只需要调用此函数，即可为 "SP500_WIND行业类" 下的所有子板块
    # 创建或更新其相关新闻数据库
    create_related_news_db("SP500_WIND行业类")
