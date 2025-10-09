import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.feature import FEATURE_META, FeatureConfig, ScalingMethod, zscore_normalize, log_zscore_normalize, clip_normalize
from data_source.finance.database import FinanceDBManager, Block


# --------------------- 基类 ---------------------
class BaseFinancialDataset:
    sample_len = 8
    target_column = '区间日均收盘价'
    sort_column = '报告期'

    def _infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
        numeric_cols = []
        for col in df.columns:
            if col not in FEATURE_META:
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            if series.isna().all():
                continue
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
        构建滑动窗口样本。
        修正后的逻辑：先对每支股票的整个时间序列特征进行归一化，然后再切分窗口。
        这保证了数据处理的一致性，并避免了前视偏差。
        """
        df = df.sort_values(by=self.sort_column).reset_index(drop=True)

        # 强制转换所有特征列为数值类型，处理非数值数据
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 步骤1: 先对整个DataFrame的特征列进行归一化
        norm_df = df.copy()
        for col in feature_columns:
            series = df[col]  # 此处 series 已经是数值类型或NaN
            # 如果一整列都是NaN，就保持原样
            if series.isna().all():
                norm_df[col] = series.values
            else:
                # 对整列（完整的历史）进行归一化
                norm_df[col] = self._scaling(series.values, FEATURE_META[col])

        samples = []

        # 步骤2: 主循环，直接从归一化好的 norm_df 中切分窗口
        for i in range(self.sample_len, len(df)):

            # 从预先归一化好的DataFrame中切片，作为模型的输入特征
            feature_matrix = norm_df.iloc[i - self.sample_len:i][feature_columns].values

            # 从已经清洗过的原始DataFrame中切片
            origin_matrix = df.iloc[i - self.sample_len:i][feature_columns].values

            # 计算目标值 (Target)
            target_val = (df.iloc[i][self.target_column] -
                          df.iloc[i - 1][self.target_column]) / df.iloc[i - 1][self.target_column]

            # 检查目标值是否有效
            if pd.isna(target_val) or np.isinf(target_val):
                continue

            samples.append((origin_matrix, feature_matrix, target_val))

        # 步骤3: 处理最后一个样本用于预测
        if allow_last_nan and len(df) >= self.sample_len:
            feature_matrix = norm_df.iloc[len(df) - self.sample_len:len(df)][feature_columns].values
            origin_matrix = df.iloc[len(df) - self.sample_len:len(df)][feature_columns].values
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
                    self.feature_columns = self._infer_feature_columns(df)

    def _build_samples(self):
        for stock_code, df in self.company_data.items():
            samples = self._build_samples_from_df(df, self.feature_columns, False)
            if samples:
                self.samples_by_company[stock_code] = samples

    def build_datasets(self, train_ratio=0.85):
        train_samples = []
        val_samples = []

        # 遍历每支股票，对其时间序列进行划分
        for _, samples in self.samples_by_company.items():
            if not samples:
                continue

            # 按时间顺序划分样本
            split_index = int(len(samples) * train_ratio)
            if split_index > 0:
                train_samples.extend(samples[:split_index])
            if split_index < len(samples):
                val_samples.extend(samples[split_index:])

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

        self.feature_columns = self._infer_feature_columns(df)
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
