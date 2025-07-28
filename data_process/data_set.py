import numpy as np
import random
import pandas as pd
from collections import defaultdict
from finance_data.database import FinanceDBManager, BlockCode
from news_data.database import NewsDBManager


class FinancialDataModule:
    """封装财务数据加载、预处理和样本构造"""

    def __init__(self, block_code=BlockCode.US_CHIP, use_news=False, min_sample_size=8):
        self.exclude_columns = ['报告期', '发布日期', '统计开始日', '统计结束日']
        self.target_column = '区间日均收盘价'
        self.min_sample_size = min_sample_size
        self.block_code = block_code
        self.use_news = use_news

        self.finance_db = FinanceDBManager(block_code)
        self.news_db = NewsDBManager(block_code) if use_news else None
        self.stock_data = self.finance_db.fetch_block_data()

        self.data_map = {}
        self.news_map = defaultdict(dict)
        self.feature_columns = []

    def build(self):
        """构建数据映射"""
        for stock_code, records in self.stock_data.items():
            if len(records) < self.min_sample_size:
                continue

            df = pd.DataFrame(records)
            if df.empty or self.target_column not in df.columns:
                continue

            df['发布日期'] = pd.to_datetime(df['发布日期'], errors='coerce')
            df = df.sort_values('发布日期')
            df['时间'] = df['发布日期'].dt.to_period('Q').astype(str)
            df['股票代码'] = stock_code

            company_data = {col: df[col].to_numpy() for col in df.columns}
            self.data_map[stock_code] = company_data

            if self.use_news:
                for t in company_data['时间']:
                    if isinstance(t, str) and len(t) >= 5:
                        year, quarter = int(t[:4]), t[4:]
                        _, embedding = self.news_db.get_news_and_embedding(stock_code, quarter, year, self.news_db)
                        if embedding:
                            self.news_map[stock_code][t] = embedding
                        else:
                            print(f"[缺失新闻] {stock_code} - {t}")

        print(f"[构建完成] 有效公司：{len(self.data_map)}")

    def get_feature_columns(self):
        if not self.feature_columns:
            sample_company = next(iter(self.data_map.values()))
            all_columns = list(sample_company.keys())
            self.feature_columns = [col for col in all_columns if col not in self.exclude_columns]
        return self.feature_columns

    def sigmoid_normalize(self, arr, scale=10.0):
        arr = np.array(arr, dtype=float)
        min_val, max_val = np.min(arr), np.max(arr)
        scaled = (arr - min_val) / (max_val - min_val + 1e-8)
        centered = (scaled - 0.5) * scale
        return 1 / (1 + np.exp(-centered))

    def generate_samples(self):
        samples = []
        feature_columns = self.get_feature_columns()

        for stock_code, company_data in self.data_map.items():
            row_count = len(next(iter(company_data.values())))
            if row_count < self.min_sample_size:
                continue

            sample_size = int(row_count // 1.5)

            for _ in range(sample_size):
                seq_len = random.randint(self.min_sample_size - 1, row_count - 1)
                max_start = row_count - seq_len - 1
                start_idx = random.randint(0, max_start)

                raw_data, sample = [], []
                target = 0.0

                for col in feature_columns:
                    col_data = company_data[col][start_idx:start_idx + seq_len]
                    col_arr = np.array(col_data, dtype=float)
                    if col == self.target_column:
                        target = float(company_data[col][start_idx + seq_len])
                    raw_data.append(col_data)
                    sample.append(self.sigmoid_normalize(col_arr))

                origin = np.stack(raw_data, axis=1)
                features = np.stack(sample, axis=1)

                item = {'origin': origin, 'features': features, 'target': target}

                if self.use_news:
                    times = company_data['时间'][start_idx:start_idx + seq_len + 1]
                    try:
                        news_features = [self.news_map[stock_code][t] for t in times]
                        item['news_features'] = news_features
                    except KeyError:
                        continue

                samples.append(item)

        print(f"[完成] 共生成 {len(samples)} 条训练样本")
        return samples
