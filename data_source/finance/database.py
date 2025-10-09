import sqlite3
import os
import data_source.finance.script.wind as wd
from data_source.finance.script.block import Block, BlockItem


class FinanceDBManager:

    def __init__(self, block: BlockItem, db_dir="clean_data"):
        self.block = block
        db_file = f"{self.block.name_cn}.db"
        self.db_path = os.path.join(os.path.dirname(__file__), db_dir, db_file)

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=DELETE;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.commit()
            self.conn.close()

    def _get_table_name(self, stock_code: str):
        return stock_code.replace(".", "_").upper()

    def _table_name_to_stock_code(self, table_name: str):
        return table_name.replace("_", ".")

    def ensure_table_exists(self, stock_code: str, sample_data: dict):
        table_name = self._get_table_name(stock_code)
        field_defs = ['id INTEGER PRIMARY KEY AUTOINCREMENT']
        for key, value in sample_data.items():
            # 推断 SQLite 字段类型
            if isinstance(value, int):
                sqlite_type = "INTEGER"
            elif isinstance(value, float):
                sqlite_type = "REAL"
            else:
                sqlite_type = "TEXT"

            field_defs.append(f'"{key}" {sqlite_type}')

        # 构造完整 SQL 并执行
        field_def_sql = ', '.join(field_defs)
        self.cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({field_def_sql})')

    def save_financial_record(self, stock_code: str, record: dict):
        table_name = self._get_table_name(stock_code)
        self.ensure_table_exists(stock_code, record)

        fields = ', '.join(f'"{k}"' for k in record)
        placeholders = ', '.join('?' for _ in record)
        values = list(record.values())

        try:
            self.cursor.execute(
                f'''
                INSERT INTO "{table_name}" ({fields}) VALUES ({placeholders})
                ''', values)
            print(f"[写入成功] {stock_code} ({self.block.name_cn}) - {record.get('报告期')}")
        except Exception as e:
            print(f"[写入失败] {stock_code} ({self.block.name_cn}) - {record.get('报告期')}，错误：{e}")
        self.conn.commit()

    # 抓取单股数据
    def fetch_stock_data(self, stock_code: str):
        """
        获取单个股票的数据，并以 [record_dict, ...] 的形式返回。
        """
        table_name = self._get_table_name(stock_code)

        # 检查表是否存在
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = self.cursor.fetchone() is not None

        if not exists:
            # 表不存在：调用 fetcher 抓取数据并存入数据库
            fetcher = wd.WindFinancialDataFetcher(stock_code=stock_code, block_code=self.block.id)
            data_list = fetcher.get_data()

            if not data_list:
                print(f"[Info] {stock_code} 无可用财报数据，跳过")
                return []

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

    # 抓取同一板块的股票数据
    def fetch_block_data(self, update=False):
        """
        获取整个板块下所有股票的财务数据，并以 {stock_code: [record_dict, ...]} 的形式返回。
        当update为false时只从数据库中获取
        """
        result = {}
        stock_codes = Block.get_stock_codes(self.block.id)
        print(f"[Info] 开始处理板块（{self.block.name_cn}）下共 {len(stock_codes)} 支股票")

        for stock_code in stock_codes:
            try:
                if update:
                    # 强制更新数据，直接调用 fetch_stock_data
                    data = self.fetch_stock_data(stock_code)
                else:
                    # 从数据库读取数据
                    table_name = self._get_table_name(stock_code)

                    # 尝试查询数据，如果表不存在或无数据，则会得到空列表
                    self.cursor.execute(f'SELECT * FROM "{table_name}"')
                    rows = self.cursor.fetchall()

                    if not rows:
                        print(f"[Info] 跳过 {stock_code}（表无数据）")
                        continue

                    col_names = [desc[0] for desc in self.cursor.description]
                    data = [dict(zip(col_names, row)) for row in rows]

                # 将有效数据添加到结果中
                if data:
                    result[stock_code] = data

            except Exception as e:
                # print(f"[错误] 获取 {stock_code} 财务数据失败：{e}")
                continue

        print(f"[完成] 板块处理完成，共获取 {len(result)} 支股票的财务数据")
        return result


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    parent_block = "SP500_WIND行业类"
    db_dir = "db/SP500_WIND行业类"
