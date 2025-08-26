import os
import yaml
from enum import Enum
from dataclasses import dataclass
import numpy as np
import math
from datetime import datetime, date
# ------------------------- 1. 枚举定义 -------------------------


class ScalingMethod(Enum):
    NONE = "none"
    ZSCORE = "z-score"
    LOG_ZSCORE = "log+z-score"
    CLIP = "clip"


# ------------------------- 2. 特征结构体 -------------------------


@dataclass
class FeatureConfig:
    train: bool
    norm: ScalingMethod
    source: str
    clip_scale: int | None = None  # 仅当 norm == CLIP 时使用


# ------------------------- 3. 特征元数据 -------------------------

FEATURE_META: dict[str, FeatureConfig] = {
    # 财报数值
    "营业收入(单季)": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "财报"),
    "营业收入(TTM)": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "财报"),
    "经营活动现金流(TTM)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报"),

    # 财报比例
    "毛利率(单季)": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "毛利率(TTM)": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "净利率(单季)": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "净利率(TTM)": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "总资产收益率(单季)": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "总资产收益率(TTM)": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "资产负债率": FeatureConfig(True, ScalingMethod.CLIP, "财报"),
    "总资产周转率": FeatureConfig(True, ScalingMethod.CLIP, "财报"),

    # 股市数值
    "区间收盘价": FeatureConfig(True, ScalingMethod.ZSCORE, "股市"),
    "区间日均收盘价": FeatureConfig(True, ScalingMethod.ZSCORE, "股市"),
    "区间最高收盘价": FeatureConfig(True, ScalingMethod.ZSCORE, "股市"),
    "区间最低收盘价": FeatureConfig(True, ScalingMethod.ZSCORE, "股市"),
    "区间最高收盘价日期": FeatureConfig(False, ScalingMethod.NONE, "股市"),
    "区间最低收盘价日期": FeatureConfig(False, ScalingMethod.NONE, "股市"),

    # 股市比例
    "区间振幅": FeatureConfig(True, ScalingMethod.CLIP, "股市"),
    "区间日均换手率": FeatureConfig(True, ScalingMethod.CLIP, "股市"),
    "区间平均PE": FeatureConfig(False, ScalingMethod.CLIP, "股市"),
    "区间平均PB": FeatureConfig(False, ScalingMethod.CLIP, "股市"),
    "区间平均PS": FeatureConfig(False, ScalingMethod.CLIP, "股市"),
    "市现率PCF": FeatureConfig(False, ScalingMethod.CLIP, "股市", 1000),

    # 板块比例
    "板块涨跌幅": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
    "板块日均换手率": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
    "板块PE": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
    "板块PB": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
    "板块PS": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
    "板块PCF": FeatureConfig(True, ScalingMethod.CLIP, "板块", 1000),
}

# ------------------------- 4. 特征映射文件加载 -------------------------

base_dir = os.path.dirname(__file__)
feature_map_path = os.path.join(base_dir, "feature.yaml")

with open(feature_map_path, "r", encoding="utf-8") as f:
    FEATURE_NAME_MAP = yaml.safe_load(f)  # 中文名 → Wind 字段
    REVERSE_MAP = {v.lower(): k for k, v in FEATURE_NAME_MAP.items()}  # Wind → 中文

# ------------------------- 5. 工具函数 -------------------------


# 中文名 → Wind 字段
def translate_to_wind_fields(feature_list_cn: list[str]) -> list[str]:
    return [FEATURE_NAME_MAP.get(f, f) for f in feature_list_cn]


# Wind 字段 → 中文名
def translate_to_chinese_fields(wind_field_list: list[str]) -> list[str]:
    return [REVERSE_MAP.get(f.lower(), f) for f in wind_field_list]


# 获取某类数据来源（如“股市”）的特征列名（中文）
def get_feature_names_by_source(source: str) -> list[str]:
    return [name for name, cfg in FEATURE_META.items() if cfg.source == source]


# 获取可参与训练的特征列名（中文）
def get_trainable_feature_names() -> list[str]:
    return [name for name, cfg in FEATURE_META.items() if cfg.train]


def build_translated_data_map(wind_fields: list[str], values: list[list]) -> dict:
    """
    将 Wind 字段名及对应数据转换为 中文字段 → 值 的映射。
    如果值是 datetime/date 类型，则转为 'yyyy-mm-dd' 字符串。
    空值和 NaN 将被转换为空字符串。
    """
    chinese_fields = translate_to_chinese_fields(wind_fields)
    result = {}

    for ch_name, val in zip(chinese_fields, values):
        v = val[0] if isinstance(val, list) and len(val) == 1 else val

        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.strftime("%Y-%m-%d")

        result[ch_name] = v

    return result


# ------------------------- 6. Wind 接口字段拼接 -------------------------

features_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("财报")))
stock_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("股市")))
block_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("板块")))


def features_wind_opt(date: int) -> str:
    return f"unit=1;rptDate={date};rptType=1;currencyType="


def stock_wind_opt(trade_days: int, end_day: int, start_day: int) -> str:
    return f"ndays=-{trade_days};tradeDate={end_day};startDate={start_day};endDate={end_day};priceAdj=F"


def block_wind_opt(start_day: int, end_day: int, year: int) -> str:
    return f"startDate={start_day};endDate={end_day};tradeDate={end_day};DynamicTime=1;excludeRule=2;year={year}"


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
