import sqlite3
import os
import wind as wd


class FinanceDBManager:

    def __init__(self, block_code: wd.BlockCode, db_dir="db"):
        self.block_code = block_code
        db_file = f"{self.block_code.name}.db"
        self.db_path = os.path.join(os.path.dirname(__file__), db_dir, db_file)

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def __del__(self):
        self.conn.commit()
        self.conn.close()

    def _get_table_name(self, stock_code: str):
        return stock_code.replace(".", "").upper()

    def ensure_table_exists(self, stock_code: str, sample_data: dict):
        table_name = self._get_table_name(stock_code)

        field_defs = """
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            报告期 TEXT,
            发布日期 TEXT,
            统计开始日 TEXT,
            统计结束日 TEXT
        """
        for key in sample_data:
            if key not in ("报告期", "发布日期", "统计开始日", "统计结束日"):
                field_defs += f', "{key}" TEXT'

        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                {field_defs}
            )
        ''')

    def save_financial_record(self, stock_code: str, record: dict):
        table_name = self._get_table_name(stock_code)
        self.ensure_table_exists(stock_code, record)

        fields = ', '.join(f'"{k}"' for k in record)
        placeholders = ', '.join('?' for _ in record)
        values = [
            f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v) if v is not None else ""
            for v in record.values()
        ]

        try:
            self.cursor.execute(
                f'''
                INSERT INTO "{table_name}" ({fields}) VALUES ({placeholders})
            ''', values)
            print(f"[写入成功] {stock_code} ({self.block_code.name}) - {record.get('报告期')}")
        except Exception as e:
            print(f"[写入失败] {stock_code} ({self.block_code.name}) - {record.get('报告期')}，错误：{e}")
        self.conn.commit()

    def load_or_fetch_data(self, stock_code: str):
        table_name = self._get_table_name(stock_code)

        # 检查表是否存在
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name, ))
        exists = self.cursor.fetchone() is not None

        if not exists:
            # 表不存在：调用 fetcher 抓取数据并存入数据库
            fetcher = wd.WindFinancialDataFetcher(stock_code=stock_code, block_code=self.block_code)
            data_list = fetcher.get_data()

            for record in data_list:
                self.save_financial_record(stock_code, record)

        # 从数据库中读取数据
        self.cursor.execute(f'SELECT * FROM "{table_name}"')
        rows = self.cursor.fetchall()

        # 获取列名
        col_names = [desc[0] for desc in self.cursor.description]

        # 转为 dict 列表
        data = [dict(zip(col_names, row)) for row in rows]
        return data


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    stock_code = "TSLA.O"
    block_code = wd.BlockCode.US_AUTO

    db = FinanceDBManager(block_code=block_code)
    data = db.load_or_fetch_data(stock_code)

    print(f"共获取 {len(data)} 条数据：")
    for row in data[:2]:
        print(row)
