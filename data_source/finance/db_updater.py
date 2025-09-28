import sqlite3
import os
import glob
from typing import Dict, List
import data_source.finance.script.wind as wd
from utils.block import Block, BlockItem
import utils.feature as ft
from data_source.finance.script.block_map import block_cache
from data_source.finance.database import FinanceDBManager
# ------------------------------------


class FinanceDBUpdater:
    """
    管理对一个已存在的金融数据库的**修改和更新**操作。
    职责:
    1. 修改表结构（添加/删除列）。
    2. 从外部源(Wind)获取增量数据，丰富（Enrich）数据库中的已有记录。
    """

    def __init__(self, db_path: str, block: BlockItem):
        """
        为指定的数据库文件初始化更新器。

        :param db_path: SQLite数据库文件的完整路径。
        :param block: 与此数据库关联的 BlockItem 对象。
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        self.db_path = db_path
        self.block = block
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=OFF;")
        self.cursor.execute("PRAGMA synchronous=FULL;")

    def __del__(self):
        """确保在对象销毁时，提交更改并关闭数据库连接。"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.commit()
            self.conn.close()

    def _get_table_name(self, stock_code: str) -> str:
        """
        将股票代码转换为表名，此方法必须与 FinanceDBManager 中的完全一致。
        例如: 'NVDA.O' -> 'NVDA_O'
        """
        return stock_code.replace(".", "_").upper()

    def get_all_stock_codes_in_db(self) -> List[str]:
        """从数据库中获取所有表的名称，并将其转换回股票代码列表。"""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        table_names = [row[0] for row in self.cursor.fetchall()]

        # 为了从 'NVDAO' 准确地还原回 'NVDA.O'，我们依赖 block_cache
        all_stocks_in_block = block_cache.get_stock_codes(self.block.id).codes
        stock_map = {self._get_table_name(code): code for code in all_stocks_in_block}

        return [stock_map[tn] for tn in table_names if tn in stock_map]

    def add_columns_to_stock(self, stock_code: str, new_columns: Dict[str, str]) -> List[str]:
        """为单个股票表安全地添加新列，返回实际新增的列名"""
        table_name = self._get_table_name(stock_code)
        self.cursor.execute(f'PRAGMA table_info("{table_name}")')
        existing_cols = {row[1] for row in self.cursor.fetchall()}

        added_cols = []
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                self.cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col_name}" {col_type}')
                added_cols.append(col_name)
                print(f"[新增列] {table_name} -> {col_name} {col_type}")
            else:
                print(f"[跳过] {table_name} 已存在列 {col_name}")
        self.conn.commit()
        return added_cols

    def enrich_stock_with_features(self, stock_code: str, indicators: List[str]):
        """
        [核心] 通过为每条已有记录抓取新特征数据，来丰富单个股票的数据。
        """
        table_name = self._get_table_name(stock_code)
        # 1. 确保所有列都存在（新增的列初始化为NULL）
        self.add_columns_to_stock(stock_code, {name: 'REAL' for name in indicators})
        # 2. 获取所有需要的列
        cols = ["id", '"统计开始日"', '"统计结束日"'] + indicators
        col_expr = ", ".join(cols)
        try:
            self.cursor.execute(f'SELECT {col_expr} FROM "{table_name}"')
            rows = self.cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"[错误] 无法读取 {table_name}：{e}")
            return
        print(f"[INFO] 当前处理 {stock_code} 中 {len(rows)} 条记录")
        # 3. 遍历行，只补充缺失的列值
        for row in rows:
            row_dict = dict(zip([c.strip('"') for c in cols], row))
            row_id, start_day, end_day = row_dict["id"], row_dict["统计开始日"], row_dict["统计结束日"]
            # 找出该行缺失的特征列
            missing_cols = [col for col in indicators if row_dict[col] is None]
            if not missing_cols:
                continue  # 这一行特征已经全有了，跳过
            try:
                # 调用外部API获取缺失的指标
                update_data = wd.fetch_data_for_period(stock_code, start_day, end_day, missing_cols, self.block.id)
                if not update_data:
                    continue
                # 只更新缺失列
                set_expr = ", ".join(f'"{k}"=?' for k in update_data)
                values = list(update_data.values())
                self.cursor.execute(f'UPDATE "{table_name}" SET {set_expr} WHERE id=?', values + [row_id])
            except Exception as e:
                print(f"    [更新失败] {stock_code} 行ID {row_id}: {e}")
        self.conn.commit()

    def enrich_block_with_features(self, indicators: List[str]):
        """[工作流] 为此数据库中的所有股票表丰富新特征。"""
        print(f"\n===== 开始丰富数据库特征: {os.path.basename(self.db_path)} =====")
        stock_codes = self.get_all_stock_codes_in_db()
        for stock_code in stock_codes:
            self.enrich_stock_with_features(stock_code, indicators)
        print(f"===== 数据库特征丰富完毕: {os.path.basename(self.db_path)} =====")

    def drop_columns_from_stock(self, stock_code: str, drop_columns: List[str]):
        """
        为单个股票的表删除指定列。
        要求 SQLite 版本 >= 3.35 才支持 ALTER TABLE DROP COLUMN。
        """
        table_name = self._get_table_name(stock_code)
        for col_name in drop_columns:
            try:
                self.cursor.execute(f'ALTER TABLE "{table_name}" DROP COLUMN "{col_name}"')
                print(f"  [删除列] 表 {table_name} 删除列 {col_name}")
            except sqlite3.OperationalError as e:
                # 如果列不存在，跳过
                if "no such column" in str(e).lower():
                    print(f"  [跳过] 表 {table_name} 不存在列 {col_name}")
                else:
                    print(f"  [错误] 删除列 {col_name} 失败: {e}")
        self.conn.commit()

    def drop_columns_from_block(self, drop_columns: List[str]):
        """[工作流] 为此数据库中的所有股票表删除指定列。"""
        print(f"\n===== 开始删除数据库列: {os.path.basename(self.db_path)} =====")
        stock_codes = self.get_all_stock_codes_in_db()
        for stock_code in stock_codes:
            self.drop_columns_from_stock(stock_code, drop_columns)
        print(f"===== 数据库列删除完毕: {os.path.basename(self.db_path)} =====")


# ==============================================================================
# =====                    高级工作流函数 (业务逻辑层)                       =====
# ==============================================================================


def run_initial_data_population(parent_block_name: str, db_root_dir: str):
    """高级工作流：执行指定父板块下所有子板块的初始数据抓取和填充。"""
    print("=" * 20 + " 工作流: [1] 初始数据填充 " + "=" * 20)
    sub_items = Block.get_items_by_parent(parent_block_name)
    if not sub_items:
        print(f"父板块 '{parent_block_name}' 下未找到任何子板块。")
        return

    for _, item in sub_items.items():
        # 使用 FinanceDBManager 来执行数据填充
        manager = FinanceDBManager(block=item, db_dir=db_root_dir)
        manager.fetch_block_data(update=True)  # update=True会强制从API抓取
    print("=" * 20 + " 工作流: [1] 初始数据填充完成 " + "=" * 20 + "\n")


def run_add_new_features(db_root_dir: str, new_features: List[str]):
    """高级工作流：扫描指定目录下的所有数据库文件，并用新特征丰富它们。"""
    print("=" * 20 + " 工作流: [2] 数据库特征丰富 " + "=" * 20)
    db_files = glob.glob(os.path.join(os.path.dirname(__file__), db_root_dir, '*.db'))
    if not db_files:
        print(f"在目录 '{db_root_dir}' 中未找到任何.db数据库文件。")
        return

    if not new_features:
        print("未定义任何要添加的特征，流程中止。")
        return

    for db_path in db_files:
        block_name = os.path.basename(db_path).replace('.db', '')
        block_item = Block.get(block_name)
        if block_item:
            # 使用 FinanceDBUpdater 来执行数据更新和丰富
            updater = FinanceDBUpdater(db_path=db_path, block=block_item)
            updater.enrich_block_with_features(indicators=new_features)
        else:
            print(f"[警告] 跳过 '{db_path}', 因为找不到名为 '{block_name}' 的板块定义。")
    print("=" * 20 + " 工作流: [2] 数据库特征丰富完成 " + "=" * 20 + "\n")


def run_drop_features(db_root_dir: str, drop_columns: List[str]):
    """高级工作流：扫描指定目录下的所有数据库文件，并删除指定的列。"""
    print("=" * 20 + " 工作流: [3] 删除列 " + "=" * 20)
    db_files = glob.glob(os.path.join(os.path.dirname(__file__), db_root_dir, '*.db'))
    if not db_files:
        print(f"在目录 '{db_root_dir}' 中未找到任何.db数据库文件。")
        return

    for db_path in db_files:
        block_name = os.path.basename(db_path).replace('.db', '')
        block_item = Block.get(block_name)
        if block_item:
            updater = FinanceDBUpdater(db_path=db_path, block=block_item)
            updater.drop_columns_from_block(drop_columns=drop_columns)
        else:
            print(f"[警告] 跳过 '{db_path}', 因为找不到名为 '{block_name}' 的板块定义。")
    print("=" * 20 + " 工作流: [3] 删除列完成 " + "=" * 20 + "\n")


def run_statistics(db_root_dir: str):
    """
    高级工作流：统计目录下所有数据库文件中，各列总数据量与缺失率（空字符串和NULL都算缺失）
    汇总同名列 across 所有表
    """
    print("=" * 20 + " 工作流: [4] 数据库统计汇总 " + "=" * 20)

    db_files = glob.glob(os.path.join(os.path.dirname(__file__), db_root_dir, '*.db'))
    if not db_files:
        print(f"在目录 '{db_root_dir}' 中未找到任何.db数据库文件。")
        return

    # 用于统计每列总行数和非缺失行数
    col_total_count: Dict[str, int] = {}
    col_valid_count: Dict[str, int] = {}

    for db_path in db_files:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取所有用户表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        table_names = [row[0] for row in cursor.fetchall()]
        for table_name in table_names:
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            total_rows = cursor.fetchone()[0]
            if total_rows == 0:
                continue
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = [row[1] for row in cursor.fetchall()]
            for col in columns:
                # 统计非空且非空字符串的数量
                cursor.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL AND TRIM("{col}") <> ""')
                valid_count = cursor.fetchone()[0]

                # 累加总行数和非缺失行数
                col_total_count[col] = col_total_count.get(col, 0) + total_rows
                col_valid_count[col] = col_valid_count.get(col, 0) + valid_count
        conn.close()

    # 输出汇总
    print(f"\n===== 数据库统计汇总 =====")
    if col_total_count:
        total_rows = next(iter(col_total_count.values()))
        print(f"总行数: {total_rows}\n")
        print(f"{'列名':<20} {'缺失率':>10}")
        print("-" * 47)
        for col, valid in col_valid_count.items():
            missing_rate = (total_rows - valid) / total_rows
            print(f"{col:<20} {missing_rate:>9.2%}")
    else:
        print("未统计到任何列数据")

    print("=" * 20 + " 工作流: [4] 数据库统计汇总完成 " + "=" * 20 + "\n")


# ==============================================================================
# =====                         程序执行入口                               =====
# ==============================================================================

if __name__ == "__main__":

    # --- 配置区 ---
    # 选择要运行的工作流:
    # 1: 从wind里抓取数据 2: 新增列 3: 删除列 4: 统计
    WORKFLOW_TO_RUN = 4

    PARENT_BLOCK = "SP500_WIND行业类"
    DB_DIRECTORY = f"db/测试"

    # --- 执行区 ---
    if WORKFLOW_TO_RUN == 1:
        run_initial_data_population(parent_block_name=PARENT_BLOCK, db_root_dir=DB_DIRECTORY)

    elif WORKFLOW_TO_RUN == 2:
        features_to_add = ft.get_feature_names_by_source("板块")
        run_add_new_features(db_root_dir=DB_DIRECTORY, new_features=features_to_add)

    elif WORKFLOW_TO_RUN == 3:
        # 示例: 删除之前添加的测试列
        features_to_drop = ft.get_feature_names_by_source("板块")
        run_drop_features(db_root_dir=DB_DIRECTORY, drop_columns=features_to_drop)

    elif WORKFLOW_TO_RUN == 4:
        run_statistics(db_root_dir=DB_DIRECTORY)

    else:
        print("无效的 WORKFLOW_TO_RUN 值。请输入 1, 2, 3 或 4。")
