import sqlite3
import os
import wind as wd

path = os.path.join(os.path.dirname(__file__), "finance.db")


class FinanceDBManager:

    def __init__(self, db_path=path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def __del__(self):
        self.conn.commit()
        self.conn.close()

    def _get_table_name(self, stock_code: str):
        return stock_code.replace(".", "")

    def ensure_table_exists(self, stock_code: str, sample_data: dict):
        table_name = self._get_table_name(stock_code)

        # 构造动态字段
        field_defs = """
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            报告期 TEXT,
            发布日期 TEXT,
            统计开始日 TEXT,
            统计结束日 TEXT
        """
        for key in sample_data:
            if key not in ("报告期", "发布日期", "统计开始日", "统计结束日"):
                field_defs += f', "{key}" TEXT'  # 用 TEXT 存储，兼容各种格式（数值或空）

        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                {field_defs}
            )
        ''')

    def save_financial_record(self, stock_code: str, record: dict):
        table_name = self._get_table_name(stock_code)
        self.ensure_table_exists(stock_code, record)

        fields = ', '.join(f'"{k}"' for k in record)
        placeholders = ', '.join('?' for _ in record)
        values = [str(v) if v is not None else "" for v in record.values()]  # 保证写入不报错

        try:
            self.cursor.execute(f'INSERT INTO {table_name} ({fields}) VALUES ({placeholders})', values)
            print(f"[写入成功] {stock_code} - {record.get('报告期')}")
        except Exception as e:
            print(f"[写入失败] {stock_code} - {record.get('报告期')}，错误：{e}")
        self.conn.commit()


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    fetcher = wd.WindFinancialDataFetcher(stock_code="NVDA.O", block_code=wd.BlockCode.US_CHIP)
    data_list = fetcher.get_data()

    db = FinanceDBManager()
    for record in data_list:
        db.save_financial_record("NVDA.O", record)
