import sqlite3
import json
import os
import re
from datetime import datetime


class ExpectationDBManager:
    """
    封装对行业和公司量化预期数据的 SQLite 管理逻辑。
    - 'industry_expectations.db' 存储所有行业的预期。
    - 'company_expectations.db' 存储所有公司的预期。
    - 每个数据库中的表对应一个行业或公司。
    - 表中的数据通过 as_of_date 进行版本控制。
    """

    def __init__(self, db_dir: str = "quant_expectations_db"):
        self.db_dir = os.path.join(os.path.dirname(__file__), db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

        # 分别为行业和公司建立数据库连接
        self.industry_db_file = os.path.join(self.db_dir, "industry_expectations.db")
        self.company_db_file = os.path.join(self.db_dir, "company_expectations.db")

        self.ind_conn = sqlite3.connect(self.industry_db_file)
        self.ind_cursor = self.ind_conn.cursor()
        self.comp_conn = sqlite3.connect(self.company_db_file)
        self.comp_cursor = self.comp_conn.cursor()

        self._init_db_settings(self.ind_cursor)
        self._init_db_settings(self.comp_cursor)

    def _format_table_name(self, name: str) -> str:
        """格式化名称为合法的 SQLite 表名"""
        safe_name = re.sub(r'[\W_]+', '_', name)
        return f'"{safe_name}"'

    def _init_db_settings(self, cursor):
        cursor.execute("PRAGMA journal_mode=DELETE;")
        cursor.execute("PRAGMA synchronous=FULL;")

    def _ensure_industry_table_exists(self, cursor, conn, table_name: str):
        query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                as_of_date TEXT PRIMARY KEY,
                full_expectation_json TEXT NOT NULL,
                core_thesis TEXT
            )
        '''
        cursor.execute(query)
        conn.commit()

    def _ensure_company_table_exists(self, cursor, conn, table_name: str):
        query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                as_of_date TEXT PRIMARY KEY,
                full_expectation_json TEXT NOT NULL,
                core_thesis TEXT,
                industry_dependencies TEXT
            )
        '''
        cursor.execute(query)
        conn.commit()

    def save_industry_expectation(self, industry_name: str, data: dict):
        table_name = self._format_table_name(industry_name)
        self._ensure_industry_table_exists(self.ind_cursor, self.ind_conn, table_name)

        as_of_date = data.get("as_of_date")
        core_thesis = data.get("core_thesis")

        query = f'INSERT OR REPLACE INTO {table_name} VALUES (?, ?, ?)'
        self.ind_cursor.execute(query, (as_of_date, json.dumps(data), core_thesis))
        self.ind_conn.commit()
        # ... (print log)

    # load_latest_industry_expectation 不变, 它返回完整的 JSON，我们可以从中提取任何字段

    def save_company_expectation(self, stock_code: str, data: dict, dependencies: dict):
        table_name = self._format_table_name(stock_code)
        self._ensure_company_table_exists(self.comp_cursor, self.comp_conn, table_name)

        as_of_date = data.get("as_of_date")
        core_thesis = data.get("core_thesis")
        dependencies_json = json.dumps(dependencies)

        query = f'INSERT OR REPLACE INTO {table_name} VALUES (?, ?, ?, ?)'
        self.comp_cursor.execute(query, (as_of_date, json.dumps(data), core_thesis, dependencies_json))
        self.comp_conn.commit()

    def load_latest_industry_expectation(self, industry_name: str) -> dict | None:
        table_name = self._format_table_name(industry_name)
        try:
            query = f"SELECT expectation_data FROM {table_name} ORDER BY as_of_date DESC LIMIT 1"
            self.ind_cursor.execute(query)
            row = self.ind_cursor.fetchone()
            return json.loads(row[0]) if row else None
        except sqlite3.OperationalError:
            return None  # 表不存在

    def save_company_expectation(self, stock_code: str, data: dict):
        table_name = self._format_table_name(stock_code)
        self._ensure_table_exists(self.comp_cursor, self.comp_conn, table_name)

        as_of_date = data.get("as_of_date", datetime.now().strftime('%Y-%m-%d'))
        time_horizon = data.get("time_horizon", "1 Year")

        query = f'''
            INSERT OR REPLACE INTO {table_name} (as_of_date, expectation_data, time_horizon)
            VALUES (?, ?, ?)
        '''
        self.comp_cursor.execute(query, (as_of_date, json.dumps(data), time_horizon))
        self.comp_conn.commit()
        print(f"✅ Saved company expectation for '{stock_code}' as of {as_of_date}")

    def load_latest_company_expectation(self, stock_code: str) -> dict | None:
        table_name = self._format_table_name(stock_code)
        try:
            query = f"SELECT expectation_data FROM {table_name} ORDER BY as_of_date DESC LIMIT 1"
            self.comp_cursor.execute(query)
            row = self.comp_cursor.fetchone()
            return json.loads(row[0]) if row else None
        except sqlite3.OperationalError:
            return None  # 表不存在

    def __del__(self):
        """析构时自动释放资源"""
        try:
            if self.ind_conn:
                self.ind_conn.commit()
                self.ind_conn.close()
            if self.comp_conn:
                self.comp_conn.commit()
                self.comp_conn.close()
        except Exception:
            pass
