import os
import yaml
from enum import Enum
from dataclasses import dataclass

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

    # 板块比例
    "板块涨跌幅": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
    "板块日均换手率": FeatureConfig(True, ScalingMethod.CLIP, "板块"),
}

# ------------------------- 4. 特征映射文件加载 -------------------------

base_dir = os.path.dirname(__file__)
feature_map_path = os.path.join(base_dir, "feature_map.yaml")

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


# ------------------------- 6. Wind 接口字段拼接 -------------------------

features_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("财报")))
stock_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("股市")))
block_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("板块")))
