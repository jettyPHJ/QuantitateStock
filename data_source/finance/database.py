import sqlite3
import os
import glob
import data_source.finance.script.wind as wd
from utils.block import Block, BlockItem
from data_source.finance.script.block_map import block_cache


class FinanceDBManager:

    def __init__(self, block: BlockItem, db_dir="db"):
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
        return stock_code.replace(".", "").upper()

    def ensure_table_exists(self, stock_code: str, sample_data: dict):
        table_name = self._get_table_name(stock_code)
        field_defs = ['id INTEGER PRIMARY KEY AUTOINCREMENT']
        for key, value in sample_data.items():
            if isinstance(value, int):
                sqlite_type = "INTEGER"
            elif isinstance(value, float):
                sqlite_type = "REAL"
            else:
                sqlite_type = "TEXT"
            field_defs.append(f'"{key}" {sqlite_type}')
        field_def_sql = ', '.join(field_defs)
        self.cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({field_def_sql})')

    def save_financial_record(self, stock_code: str, record: dict):
        table_name = self._get_table_name(stock_code)
        self.ensure_table_exists(stock_code, record)
        fields = ', '.join(f'"{k}"' for k in record)
        placeholders = ', '.join('?' for _ in record)
        values = list(record.values())
        try:
            self.cursor.execute(f'INSERT INTO "{table_name}" ({fields}) VALUES ({placeholders})', values)
            print(f"[写入成功] {stock_code} ({self.block.name_cn}) - {record.get('报告期')}")
        except Exception as e:
            print(f"[写入失败] {stock_code} ({self.block.name_cn}) - {record.get('报告期')}，错误：{e}")
        self.conn.commit()

    def fetch_stock_data(self, stock_code: str):
        table_name = self._get_table_name(stock_code)
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = self.cursor.fetchone() is not None
        if not exists:
            fetcher = wd.WindFinancialDataFetcher(stock_code=stock_code, block_code=self.block.id)
            data_list = fetcher.get_data()
            if not data_list:
                print(f"[Info] {stock_code} 无可用财报数据，跳过")
                return []
            for record in data_list:
                self.save_financial_record(stock_code, record)
        self.cursor.execute(f'SELECT * FROM "{table_name}"')
        rows = self.cursor.fetchall()
        col_names = [desc[0] for desc in self.cursor.description]
        data = [dict(zip(col_names, row)) for row in rows]
        return data

    def fetch_block_data(self, update=False):
        result = {}
        stock_codes = block_cache.get_stock_codes(self.block.id).codes
        print(f"[Info] 开始处理板块（{self.block.name_cn}）下共 {len(stock_codes)} 支股票")
        for stock_code in stock_codes:
            try:
                if update:
                    data = self.fetch_stock_data(stock_code)
                else:
                    table_name = self._get_table_name(stock_code)
                    self.cursor.execute(f'SELECT * FROM "{table_name}"')
                    rows = self.cursor.fetchall()
                    if not rows:
                        print(f"[Info] 跳过 {stock_code}（表无数据）")
                        continue
                    col_names = [desc[0] for desc in self.cursor.description]
                    data = [dict(zip(col_names, row)) for row in rows]
                if data:
                    result[stock_code] = data
            except Exception as e:
                continue
        print(f"[完成] 板块处理完成，共获取 {len(result)} 支股票的财务数据")
        return result

    def add_columns(self, stock_code: str, new_columns: dict, default_values: dict = None):
        """
        新增列，并可批量设置默认值
        :param stock_code: 股票代码
        :param new_columns: dict，key=列名，value=类型('TEXT', 'INTEGER', 'REAL')
        :param default_values: 可选 dict，key=列名，value=默认值，用于更新已有行
        """
        table_name = self._get_table_name(stock_code)
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if self.cursor.fetchone() is None:
            print(f"[跳过] 表 {table_name} 不存在，无法添加列")
            return

        added_cols = []
        for col_name, col_type in new_columns.items():
            try:
                self.cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col_name}" {col_type}')
                print(f"  [新增列] {table_name} 添加列 {col_name} {col_type}")
                added_cols.append(col_name)
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"  [跳过] {table_name} 已存在列 {col_name}")
                else:
                    print(f"  [错误] 新增列 {col_name} 失败: {e}")

        if default_values:
            update_cols = {k: v for k, v in default_values.items() if k in new_columns}
            set_expr = ', '.join(f'"{col}"=?' for col in update_cols)
            if set_expr:
                values = list(update_cols.values())
                self.cursor.execute(f'UPDATE "{table_name}" SET {set_expr}', values)
                print(f"  [更新值] {table_name} 已批量赋默认值 {update_cols}")

        self.conn.commit()

    def drop_columns(self, stock_code: str, columns_to_drop: list):
        """
        删除列
        :param stock_code: 股票代码
        :param columns_to_drop: 列名列表
        """
        table_name = self._get_table_name(stock_code)
        for col_name in columns_to_drop:
            try:
                self.cursor.execute(f'ALTER TABLE "{table_name}" DROP COLUMN "{col_name}"')
                print(f"  [删除列] {table_name} 删除列 {col_name}")
            except sqlite3.OperationalError as e:
                print(f"  [跳过/错误] {table_name} 删除列 {col_name} 失败: {e}")
        self.conn.commit()

    # --- 批量操作方法 ---
    def batch_add_columns_for_block(self, new_columns: dict, default_values: dict):
        """
        通过循环调用 add_columns，为板块下所有股票的表批量添加列。
        """
        print(f"\n--- 正在为板块 '{self.block.name_cn}' 批量添加列 ---")
        stock_codes = block_cache.get_stock_codes(self.block.id).codes
        for stock_code in stock_codes:
            self.add_columns(stock_code, new_columns, default_values)
        print(f"--- 板块 '{self.block.name_cn}' 批量添加列完成 ---\n")

    def batch_drop_columns_for_block(self, columns_to_drop: list):
        """
        通过循环调用 drop_columns，为板块下所有股票的表批量删除列。
        """
        print(f"\n--- 正在为板块 '{self.block.name_cn}' 批量删除列 ---")
        stock_codes = block_cache.get_stock_codes(self.block.id).codes
        for stock_code in stock_codes:
            self.drop_columns(stock_code, columns_to_drop)
        print(f"--- 板块 '{self.block.name_cn}' 批量删除列完成 ---\n")


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":

    target_db_dir = os.path.join(os.path.dirname(__file__), "db/SP500_WIND行业类")

    # =======================================================================
    #  阶段一：准备测试数据 (如果已有数据库，可注释掉此部分)
    # =======================================================================
    print("=" * 20 + " 阶段一：准备测试数据库和表 " + "=" * 20)
    parent_block_setup = "SP500_WIND行业类"
    db_dir_setup = f"db/{parent_block_setup}"
    sub_items_setup = Block.get_items_by_parent(parent_block_setup)

    if sub_items_setup:
        for name, item in sub_items_setup.items():
            # 确保 block_cache 中有数据
            stocks = block_cache.get_stock_codes(item.id).codes
            if not stocks:
                print(f"板块 '{name}' 无成分股，跳过数据准备。")
                continue
            db_setup = FinanceDBManager(block=item, db_dir=db_dir_setup)
            db_setup.fetch_block_data(update=True)
    else:
        print("没有找到任何子项目用于数据准备")
    print("=" * 20 + " 阶段一：数据准备完毕 " + "=" * 20 + "\n")

    # =======================================================================
    #  阶段二：执行批量添加测试列
    # =======================================================================
    # print("=" * 20 + " 阶段二：开始批量添加测试列 " + "=" * 20)
    # db_files_add = glob.glob(os.path.join(target_db_dir, '*.db'))

    # if not db_files_add:
    #     print(f"在目录 '{target_db_dir}' 中未找到 .db 文件。")
    # else:
    #     for db_path in db_files_add:
    #         block_name = os.path.basename(db_path).replace('.db', '')
    #         block_item = Block.get(block_name)
    #         if block_item:
    #             relative_dir = os.path.relpath(os.path.dirname(db_path), os.path.dirname(__file__))
    #             db_modifier = FinanceDBManager(block=block_item, db_dir=relative_dir)
    #             db_modifier.batch_add_columns_for_block(new_columns={'测试': 'INTEGER'}, default_values={'测试': 1})
    #         else:
    #             print(f"[警告] 找不到与数据库 '{os.path.basename(db_path)}' 对应的板块信息，已跳过。")
    # print("=" * 20 + " 阶段二：批量添加测试列完成 " + "=" * 20 + "\n")

    # =======================================================================
    #  阶段三：执行批量删除测试列 (清理阶段)
    # =======================================================================
    # input("按 Enter键 继续执行删除测试列的清理操作...")
    # print("\n" + "=" * 20 + " 阶段三：开始批量刪除测试列 " + "=" * 20)
    # db_files_drop = glob.glob(os.path.join(target_db_dir, '*.db'))

    # if not db_files_drop:
    #     print(f"在目录 '{target_db_dir}' 中未找到 .db 文件。")
    # else:
    #     for db_path in db_files_drop:
    #         block_name = os.path.basename(db_path).replace('.db', '')
    #         block_item = Block.get(block_name)
    #         if block_item:
    #             relative_dir = os.path.relpath(os.path.dirname(db_path), os.path.dirname(__file__))
    #             db_cleaner = FinanceDBManager(block=block_item, db_dir=relative_dir)
    #             db_cleaner.batch_drop_columns_for_block(columns_to_drop=['测试'])
    #         else:
    #             print(f"[警告] 找不到与数据库 '{os.path.basename(db_path)}' 对应的板块信息，已跳过。")
    # print("=" * 20 + " 阶段三：批量删除测试列完成 " + "=" * 20)
