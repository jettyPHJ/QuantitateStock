import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.feature import FEATURE_META, FeatureConfig, ScalingMethod, zscore_normalize, log_zscore_normalize, clip_normalize, get_trainable_feature_names
from data_source.finance.database import FinanceDBManager, Block


# --------------------- 基类 ---------------------
class BaseFinancialDataset:
    sample_len = 8
    target_column = '区间日均收盘价'
    sort_column = '报告期'

    def _infer_feature_columns(self) -> list[str]:
        return get_trainable_feature_names()

    def _scaling(self, arr: np.ndarray, config: FeatureConfig) -> np.ndarray:
        """
        Args:
            arr: numpy 数组，一维，表示滑动窗口的特征数据。
            method: ScalingMethod 枚举值，指定归一化方法。
        Returns:
            归一化后的 numpy 数组。
        """
        arr = np.array(arr, dtype=float)
        try:
            if config.norm == ScalingMethod.NONE:
                return arr
            elif config.norm == ScalingMethod.ZSCORE:
                return zscore_normalize(arr)
            elif config.norm == ScalingMethod.LOG_ZSCORE:
                return log_zscore_normalize(arr, offset=1)
            elif config.norm == ScalingMethod.CLIP:
                scale = config.clip_scale or 100
                return clip_normalize(arr, min_val=-scale, max_val=scale)
            else:
                return arr
        except Exception as e:
            print(f"Warning: scaling failed with error: {e}")
            return arr

    def _build_samples_from_df(self, df: pd.DataFrame, feature_columns: list[str],
                               allow_last_nan: bool = False) -> list:
        """
        构建滑动窗口样本，归一化采用历史累计数据。
        allow_last_nan用于预测脚本，允许最后一行数据中存在 NaN 值。
        """
        df = df.sort_values(by=self.sort_column).reset_index(drop=True)
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        samples = []

        def _build_one_sample(i):
            hist_df = df.iloc[:i]
            window_df = df.iloc[i - self.sample_len:i]

            norm_features, origin_features = [], []

            for col in feature_columns:
                hist_series = hist_df[col].values
                window_series = window_df[col].values
                normed = self._scaling(hist_series, FEATURE_META[col])[-self.sample_len:]

                if len(normed) != self.sample_len:
                    normed = np.full(self.sample_len, np.nan)
                if len(window_series) != self.sample_len:
                    window_series = np.full(self.sample_len, np.nan)

                norm_features.append(normed)
                origin_features.append(window_series)

            feature_matrix = np.stack(norm_features, axis=-1)
            origin_matrix = np.stack(origin_features, axis=-1)

            return origin_matrix, feature_matrix

        # 主循环构造正常样本
        for i in range(self.sample_len, len(df)):
            origin_matrix, feature_matrix = _build_one_sample(i)

            target_val = (df.iloc[i][self.target_column] -
                          df.iloc[i - 1][self.target_column]) / df.iloc[i - 1][self.target_column]
            if pd.isna(target_val):
                continue

            samples.append((origin_matrix, feature_matrix, target_val))

        # 最后一个样本（预测用，无 target）
        if allow_last_nan and len(df) >= self.sample_len:
            origin_matrix, feature_matrix = _build_one_sample(len(df))
            samples.append((origin_matrix, feature_matrix, None))

        return samples


# --------------------- 用于训练集 ---------------------
class FinancialDataset(BaseFinancialDataset, Dataset):

    def __init__(self, block_codes, use_news=False, exclude_stocks=[], update=False):
        if not isinstance(block_codes, list):
            block_codes = [block_codes]
        self.block_codes = block_codes
        self.use_news = use_news
        self.exclude_stocks = exclude_stocks
        self.update = update
        self.finance_dbs = [FinanceDBManager(code) for code in block_codes]
        self.company_data = {}
        self.feature_columns = []
        self.samples_by_company = {}

        self._load_data()
        self._build_samples()

    def _load_data(self):
        """加载每家公司数据并自动识别可训练字段,空字符串设置为nan"""
        for finance_db in self.finance_dbs:
            raw_data = finance_db.fetch_block_data(self.update)
            for stock_code, records in raw_data.items():
                if stock_code in self.exclude_stocks:
                    continue
                df = pd.DataFrame(records).replace(r'^\s*$', pd.NA, regex=True)
                if df.empty or self.target_column not in df.columns:
                    continue
                if df[self.sort_column].isna().any():
                    continue
                df = df.reset_index(drop=True)
                self.company_data[stock_code] = df
                if not self.feature_columns:
                    self.feature_columns = self._infer_feature_columns()

    def _build_samples(self):
        for stock_code, df in self.company_data.items():
            samples = self._build_samples_from_df(df, self.feature_columns, False)
            if samples:
                self.samples_by_company[stock_code] = samples

    def build_datasets(self, train_ratio=0.85, split_method='chronological'):
        """
        构建训练和验证数据集。

        Args:
            train_ratio (float): 训练集比例。
            split_method (str): 'chronological' 或 'random'。
                - 'chronological': 对每支股票按时间划分，用于真实回测。
                - 'random': 将所有样本混合后随机划分，用于诊断数据分布问题。
        """
        if split_method not in ['chronological', 'random']:
            raise ValueError("split_method must be 'chronological' or 'random'")

        if split_method == 'chronological':
            train_samples = []
            val_samples = []
            # 遍历每支股票，对其时间序列进行划分
            for _, samples in self.samples_by_company.items():
                if not samples:
                    continue
                split_index = int(len(samples) * train_ratio)
                if split_index > 0:
                    train_samples.extend(samples[:split_index])
                if split_index < len(samples):
                    val_samples.extend(samples[split_index:])

        else:  # split_method == 'random'
            # 先将所有公司的所有样本汇集到一个大列表中
            all_samples = []
            for _, samples in self.samples_by_company.items():
                all_samples.extend(samples)

            # 彻底打乱所有样本
            np.random.seed(27)
            np.random.shuffle(all_samples)

            # 然后再进行划分
            split_index = int(len(all_samples) * train_ratio)
            train_samples = all_samples[:split_index]
            val_samples = all_samples[split_index:]

        # （可选）可以打乱划分好的训练集，这有助于训练
        np.random.seed(27)
        np.random.shuffle(train_samples)

        return TorchFinancialDataset(train_samples), TorchFinancialDataset(val_samples)


# --------------------- 用于单支股票推理 ---------------------
class SingleStockDataset(BaseFinancialDataset, Dataset):

    def __init__(self, stock_code: str, block_code: str):
        self.stock_code = stock_code
        self.block_code = block_code
        self.finance_db = FinanceDBManager(block_code)
        df = pd.DataFrame(self.finance_db.fetch_stock_data(stock_code))

        if df.empty or self.target_column not in df.columns:
            raise ValueError(f"{stock_code} 财务数据无效")
        if df[self.sort_column].isna().any():
            raise ValueError(f"{stock_code} 报告期字段为空")

        self.feature_columns = self._infer_feature_columns()
        self.samples = self._build_samples_from_df(df, self.feature_columns, True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        origin, features = self.samples[index][:2]

        dummy_target = 0.0  # 随便给个 target 值
        return (
            torch.tensor(origin, dtype=torch.float32),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor([dummy_target], dtype=torch.float32),
        )


# --------------------- PyTorch 数据集封装 ---------------------
class TorchFinancialDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        origin, features, target = self.samples[index]
        return (
            torch.tensor(origin, dtype=torch.float32),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor([target], dtype=torch.float32),
        )


def collate_fn(batch):
    """
    用于 DataLoader 的批处理函数（定长版本）
    返回: origins, features, targets
    """
    batch_origins = [item[0] for item in batch]
    batch_features = torch.stack([item[1] for item in batch])
    batch_targets = torch.stack([item[2] for item in batch]).squeeze(-1)
    return batch_origins, batch_features, batch_targets


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    dataset = FinancialDataset(block_codes=[Block.get("标普500指数").id], update=True)
    train_set, val_set = dataset.build_datasets()
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
