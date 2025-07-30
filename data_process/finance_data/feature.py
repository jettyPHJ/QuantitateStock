from enum import Enum
import yaml
import os


class Normalization(Enum):
    NONE = "none"
    ZSCORE = "z-score"
    LOG_ZSCORE = "log+z-score"
    CLIP = "clip"


FEATURE_META = {
    # 财报数值
    "营业收入(单季)": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "财报"},
    "营业收入(TTM)": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "财报"},
    "经营活动现金流(TTM)": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "财报"},

    # 财报比例
    "毛利率(单季)": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "毛利率(TTM)": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "净利率(单季)": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "净利率(TTM)": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "总资产收益率(单季)": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "总资产收益率(TTM)": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "资产负债率": {"train": True, "norm": Normalization.CLIP, "source": "财报"},
    "总资产周转率": {"train": True, "norm": Normalization.CLIP, "source": "财报"},

    # 股市数值
    "区间收盘价": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "股市"},
    "区间日均收盘价": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "股市"},
    "区间最高收盘价": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "股市"},
    "区间最低收盘价": {"train": True, "norm": Normalization.LOG_ZSCORE, "source": "股市"},
    "区间最高收盘价日期": {"train": False, "norm": Normalization.NONE, "source": "股市"},
    "区间最低收盘价日期": {"train": False, "norm": Normalization.NONE, "source": "股市"},

    # 股市比例
    "区间振幅": {"train": True, "norm": Normalization.CLIP, "source": "股市"},
    "区间日均换手率": {"train": True, "norm": Normalization.CLIP, "source": "股市"},

    # 板块比例
    "板块涨跌幅": {"train": True, "norm": Normalization.CLIP, "source": "板块"},
    "板块日均换手率": {"train": True, "norm": Normalization.CLIP, "source": "板块"},
}

base_dir = os.path.dirname(__file__)
feature_map_path = os.path.join(base_dir, "feature_map.yaml")
with open(feature_map_path, "r", encoding="utf-8") as f:
    FEATURE_NAME_MAP = yaml.safe_load(f)
    REVERSE_MAP = {v: k for k, v in FEATURE_NAME_MAP.items()}


# 中文名 → Wind字段
def translate_to_wind_fields(feature_list_cn: list[str]) -> list[str]:
    return [FEATURE_NAME_MAP.get(f, f) for f in feature_list_cn]


# Wind字段 → 中文名
def translate_to_chinese_fields(wind_field_list: list[str]) -> list[str]:
    return [REVERSE_MAP.get(f.lower(), f) for f in wind_field_list]


# 中文列名 → Wind字段名 映射函数你已有
def get_feature_names_by_source(source: str) -> list[str]:
    return [name for name, cfg in FEATURE_META.items() if cfg["source"] == source]


# 自动拼接用于万得接口 wss/wsd 的字段字符串
features_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("财报")))
stock_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("股市")))
block_wind = ",".join(translate_to_wind_fields(get_feature_names_by_source("板块")))
