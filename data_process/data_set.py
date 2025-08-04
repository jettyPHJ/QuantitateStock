import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_process.finance_data.feature import FEATURE_META, FeatureConfig, ScalingMethod
from data_process.finance_data.database import BlockCode, FinanceDBManager
from data_process.news_data.database import NewsDBManager


class FinancialDataset(Dataset):
    """基于数据库构造的财务+新闻数据集"""

    def __init__(self, block_code=BlockCode.US_CHIP, use_news=False, exclude_stocks=[]):
        self.block_code = block_code
        self.use_news = use_news
        self.sample_len = 8
        self.target_column = '区间日均收盘价'
        self.sort_column = '报告期'
        self.exclude_stocks = exclude_stocks

        self.finance_db = FinanceDBManager(block_code)
        self.news_db = NewsDBManager(block_code) if use_news else None

        self.company_data = {}
        self.feature_columns = []

        self.samples = []  # 用于训练的数据集，缺失值保留，设为Nan

        self._load_data()
        self._build_samples()

    def _load_data(self):
        """加载每家公司数据并自动识别可训练字段,空字符串设置为nan"""
        raw_data = self.finance_db.fetch_block_data()
        for stock_code, records in raw_data.items():
            if stock_code in self.exclude_stocks:
                continue
            df = pd.DataFrame(records).replace(r'^\s*$', pd.NA, regex=True)

            if df[self.target_column].isna().any():
                print(f"Warning: {stock_code} has NaN in target column '{self.target_column}'.")
                continue
            if df[self.sort_column].isna().any():
                print(f"Warning: {stock_code} has NaN in sort column '{self.sort_column}'.")
                continue
            if df.empty or self.target_column not in df.columns:
                print(f"Warning: {stock_code} is missing the target column !")
                continue
            if self.use_news and '哈希' not in df.columns:
                print(f"Warning: {stock_code} is missing the hash column !")
                continue

            df = df.reset_index(drop=True)
            self.company_data[stock_code] = df

            if not self.feature_columns:
                self.feature_columns = self._infer_feature_columns(df)

    def _infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        自动识别用于训练的数值列，排除日期、布尔、非数值和全空列。
        Args:输入的 Pandas DataFrame。
        Returns:一个包含数值列名的列表。
        """
        numeric_cols = []

        for col in df.columns:
            series = df[col]

            # 检查1: 跳过日期时间类型的列
            if pd.api.types.is_datetime64_any_dtype(series):
                continue

            # 检查2: 跳过布尔类型的列
            if pd.api.types.is_bool_dtype(series):
                continue

            # 检查3: 跳过所有值都为缺失值的列
            # 因为数据已在_load_data中清洗，所以这里直接检查isna()即可
            if series.isna().all():
                continue

            # 检查4: 尝试将列转换为数值类型，如果转换后所有值都变成NaN，则说明该列不是数值列
            # (例如，一个包含['a', 'b', 'c']的列)
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.isna().all():
                continue

            # 检查5：是否在FEATURE_META中能找到列名
            if col not in FEATURE_META:
                continue

            # 如果通过所有检查，则认为是有效的特征列
            numeric_cols.append(col)

        return numeric_cols

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
                scale = config.clip_scale or 100  # 默认 100
                return clip_normalize(arr, min_val=-scale, max_val=scale)
            else:
                # 未知方法返回原始数据
                return arr
        except Exception as e:
            print(f"Warning: scaling failed with error: {e}")
            return arr

    def _build_samples(self):
        """
        构建滑动窗口样本，归一化采用历史累计数据。
        """
        for _, df in self.company_data.items():
            df: pd.DataFrame = df.sort_values(by=self.sort_column).reset_index(drop=True)

            # 所有特征列变成数值型，便于后续处理
            for col in self.feature_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            for i in range(self.sample_len, len(df)):
                hist_df = df.iloc[:i]  # 包含所有历史数据
                window_df = hist_df.iloc[-self.sample_len:]  # 取当前窗口

                # 构造每个特征的归一化值（用所有历史数据统计参数）
                norm_features = []
                origin_features = []

                for col in self.feature_columns:
                    hist_series = hist_df[col].values  # 历史全部数据
                    window_series = window_df[col].values  # 当前窗口

                    # 保留原始值和归一化值，即使是 NaN
                    normed = self._scaling(hist_series, FEATURE_META[col])[-self.sample_len:]

                    # 确保维度对齐
                    if len(normed) != self.sample_len or len(window_series) != self.sample_len:
                        print(f"Warning: feature {col} at i={i} has mismatched length.")
                        normed = np.full(self.sample_len, np.nan)
                        window_series = np.full(self.sample_len, np.nan)

                    norm_features.append(normed)
                    origin_features.append(window_series)

                feature_matrix = np.stack(norm_features, axis=-1)  # (seq_len, num_features)
                origin_matrix = np.stack(origin_features, axis=-1)

                target_val = (df.iloc[i][self.target_column] -
                              df.iloc[i - 1][self.target_column]) / df.iloc[i - 1][self.target_column]
                if pd.isna(target_val):
                    continue

                self.samples.append((origin_matrix, feature_matrix, target_val))

    def build_datasets(self, train_ratio=0.8, seed=27):
        np.random.seed(seed)
        all_samples = self.samples
        np.random.shuffle(all_samples)
        train_size = int(len(all_samples) * train_ratio)
        train_set = TorchFinancialDataset(all_samples[:train_size])
        val_set = TorchFinancialDataset(all_samples[train_size:])
        return train_set, val_set


class TorchFinancialDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        origin, features, target = self.samples[index]
        return (torch.tensor(origin, dtype=torch.float32), torch.tensor(features, dtype=torch.float32),
                torch.tensor([target], dtype=torch.float32))


def collate_fn(batch):
    """
    用于 DataLoader 的批处理函数（定长版本）
    返回: origins, features, targets
    """
    batch_origins = [item[0] for item in batch]
    batch_features = torch.stack([item[1] for item in batch])  # shape: (B, T, F)
    batch_targets = torch.stack([item[2] for item in batch]).squeeze(-1)  # shape: (B,)

    return batch_origins, batch_features, batch_targets


# ---------------------归一化函数-----------------------
# 输入的是滑动窗口，归一化采用全部历史数据


def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """
    对数组进行 Z-score 归一化，忽略 NaN。保留原 NaN。
    """
    arr = np.array(arr, dtype=float)
    mask = ~np.isnan(arr)

    if mask.sum() == 0:
        return np.full_like(arr, np.nan)  # 全是空

    mean_val = np.mean(arr[mask])
    std_val = np.std(arr[mask])

    if std_val == 0:
        normed = np.zeros_like(arr)
        normed[~mask] = np.nan
        return normed

    normed = np.empty_like(arr)
    normed[mask] = (arr[mask] - mean_val) / std_val
    normed[~mask] = np.nan
    return normed


def log_zscore_normalize(arr: np.ndarray, offset=1) -> np.ndarray:
    """
    对非 NaN 元素进行 log + z-score 归一化。
    原始值中小于 0 的元素直接设为 NaN。
    """
    arr = np.array(arr, dtype=float)
    safe_arr = np.where(arr < 0, np.nan, arr)  # 负值也视为 nan

    with np.errstate(invalid='ignore'):
        log_arr = np.log(safe_arr + offset)

    return zscore_normalize(log_arr)


def clip_normalize(arr, min_val=0.0, max_val=1.0):
    """
    裁剪归一化
    限制在指定的最小值和最大值之间。
    """
    arr = np.array(arr / max_val, dtype=float)
    return np.clip(arr, min_val, max_val)


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    data = FinancialDataset(block_code=BlockCode.NASDAQ_Computer_Index, use_news=False)
