import os
import yaml
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import math
from datetime import datetime, date

# ------------------------- 1. 枚举定义 -------------------------


class ScalingMethod(Enum):
    """数据归一化方法枚举"""
    NONE = "none"
    ZSCORE = "z-score"
    LOG_ZSCORE = "log+z-score"
    CLIP = "clip"


# ------------------------- 2. 特征结构体 -------------------------


@dataclass
class FeatureConfig:
    """
    单个特征的完整配置信息。
    """
    train: bool  # 是否作为训练特征
    norm: ScalingMethod  # 归一化方法
    source: str  # 数据来源分类（财报、股市、板块等）
    clip_scale: Optional[float] = None  # 当 norm == CLIP 时，用于缩放和裁剪的最大值

    # ------ Wind 相关配置（用于每个特征单独控制查询） ------
    wind_api: Optional[str] = None  # e.g., "wsd", "wss", "wset", "wsi"
    wind_field: Optional[str] = None  # 如果希望覆盖 feature.yaml 的映射，在此处指定

    # wind_params 是一个 key->value 映射，最终被格式化成 "key=val;key2=val2" 的 options 字符串
    wind_params: Dict[str, Any] = field(default_factory=dict)


# ------------------------- 3. 特征元数据 -------------------------

# 核心配置区域：定义所有特征的元数据
# 注意：wind_params 中的 key 必须是 Wind API 认可的参数名
FEATURE_META: Dict[str, FeatureConfig] = {

    # ------------------------------财报数值-------------------------------
    "营业收入(单季)": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "财报", wind_api="wss", wind_field="wgsd_qfa_sales_oper",
                              wind_params={"unit": "1", "rptDate": "", "rptType": "1", "currencyType": " "}),
    "营业收入(TTM)": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "财报", wind_api="wss", wind_field="or_ttm3",
                               wind_params={"unit": "1", "rptDate": "", "currencyType": " "}),
    "经营活动现金流(TTM)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="operatecashflow_ttm3",
                                  wind_params={"unit": "1", "rptDate": "", "currencyType": " "}),

    # ------------------------------------------财报比例------------------------------------
    "毛利率(单季)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="wgsd_qfa_grossprofitmargin",
                             wind_params={"rptDate": "", "rptType": "1"}),
    "毛利率(TTM)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="grossprofitmargin_ttm3",
                              wind_params={"rptDate": ""}),
    "净利率(单季)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="wgsd_qfa_netprofitmargin",
                             wind_params={"unit": 1, "rptDate": "", "rptType": 1}),
    "净利率(TTM)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="netprofitmargin_ttm3",
                              wind_params={"rptDate": ""}),
    "总资产收益率(单季)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="wgsd_qfa_roa",
                                wind_params={"rptDate": "", "rptType": "1"}),
    "总资产收益率(TTM)": FeatureConfig(True, ScalingMethod.ZSCORE, "财报", wind_api="wss", wind_field="roa_ttm2",
                                 wind_params={"rptDate": ""}),
    "资产负债率": FeatureConfig(True, ScalingMethod.CLIP, "财报", wind_api="wss", wind_field="wgsd_debttoassets",
                           wind_params={"rptDate": ""}),
    "总资产周转率": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "财报", wind_api="wss", wind_field="wgsd_assetsturn",
                            wind_params={"rptDate": ""}),

    # -----------------------------------股市数值 (wss - 获取区间统计)--------------------------
    "区间收盘价": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "股市", wind_api="wss", wind_field="close_per",
                           wind_params={"startDate": "", "endDate": "", "priceAdj": "F"}),
    "区间日均收盘价": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "股市", wind_api="wss", wind_field="avgclose_per",
                             wind_params={"ndays": "-1", "tradeDate": "", "priceAdj": "F"}),

    # ---------------------------------股市比例------------------------------------
    "区间振幅": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "股市", wind_api="wss", wind_field="swing_per",
                          wind_params={"startDate": "", "endDate": ""}),
    "区间日均换手率": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "股市", wind_api="wss", wind_field="avg_turn_per",
                             wind_params={"startDate": "", "endDate": ""}),
    "区间平均PE": FeatureConfig(True, ScalingMethod.ZSCORE, "股市", wind_api="wss", wind_field="val_pettm_avg",
                            wind_params={"startDate": "", "endDate": ""}),
    "区间平均PB": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "股市", wind_api="wss", wind_field="val_pb_avg",
                            wind_params={"startDate": "", "endDate": ""}),
    "区间平均PS": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "股市", wind_api="wss", wind_field="val_psttm_avg",
                            wind_params={"startDate": "", "endDate": ""}),
    "市现率PCF": FeatureConfig(True, ScalingMethod.ZSCORE, "股市", wind_api="wss", wind_field="pcf_ocf_ttm",
                            wind_params={"tradeDate": ""}),

    # ----------------------板块比例 (wsee - 获取板块历史估值)--------------------------------
    "板块涨跌幅": FeatureConfig(True, ScalingMethod.ZSCORE, "板块", wind_api="wsee", wind_field="sec_pq_pct_chg_tmc_wavg",
                           wind_params={"startDate": "", "endDate": "", "DynamicTime": "0"}),
    "板块日均换手率": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "板块", wind_api="wsee", wind_field="sec_pq_avgturn_avg",
                             wind_params={"startDate": "", "endDate": "", "DynamicTime": "0"}),
    "板块PE": FeatureConfig(True, ScalingMethod.ZSCORE, "板块", wind_api="wsee", wind_field="sec_pe_media_chn",
                          wind_params={"tradeDate": "", "ruleType": "10", "excludeRule": "2", "DynamicTime": "0"}),
    "板块PB": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "板块", wind_api="wsee", wind_field="sec_pb_media_chn",
                          wind_params={"tradeDate": "", "ruleType": "10", "excludeRule": "2", "DynamicTime": "0"}),
    "板块PS": FeatureConfig(True, ScalingMethod.LOG_ZSCORE, "板块", wind_api="wsee", wind_field="sec_ps_media_chn",
                          wind_params={"tradeDate": "", "ruleType": "10", "excludeRule": "2", "DynamicTime": "0"}),
    "板块PCF": FeatureConfig(True, ScalingMethod.ZSCORE, "板块", wind_api="wsee", wind_field="sec_pcf_media_chn",
                           wind_params={"tradeDate": "", "ruleType": "10", "excludeRule": "2", "DynamicTime": "0"}),
}

# ------------------------- 4. 核心工具函数 -------------------------


def translate_to_wind_fields(feature_list_cn: List[str]) -> List[str]:
    wind_fields = []
    for name in feature_list_cn:
        config = FEATURE_META.get(name)
        if config and config.wind_field:
            wind_fields.append(config.wind_field)
        else:
            # 没有 wind_field 就直接用中文名
            wind_fields.append(name)
    return wind_fields


# 在 FEATURE_META 定义后加一个反查表
WIND_TO_CN = {(cfg.wind_field or name).lower(): name for name, cfg in FEATURE_META.items()}


def translate_to_chinese_fields(wind_field_list: List[str]) -> List[str]:
    return [WIND_TO_CN.get(f.lower(), f) for f in wind_field_list]


def get_feature_names_by_source(source: str) -> List[str]:
    """获取某类数据来源（如“股市”）的特征列名（中文）"""
    return [name for name, cfg in FEATURE_META.items() if cfg.source == source]


def get_trainable_feature_names() -> List[str]:
    """获取可参与训练的特征列名（中文）"""
    return [name for name, cfg in FEATURE_META.items() if cfg.train]


def build_translated_data_map(wind_fields: List[str], values: List[list]) -> Dict[str, Any]:
    """
    将 Wind 返回的字段名和数据，转换为 {中文名: 值} 的字典。
    """
    chinese_fields = translate_to_chinese_fields(wind_fields)
    result = {}
    for ch_name, val in zip(chinese_fields, values):
        v = val[0] if isinstance(val, list) and len(val) == 1 else val
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = None  # 统一使用 None 代表缺失值
        elif isinstance(v, (datetime, date)):
            v = v.strftime("%Y-%m-%d")
        result[ch_name] = v
    return result


# ------------------------- 5. Wind 接口动态参数构建 -------------------------


def build_wind_options(config: FeatureConfig, context: Dict[str, Any]) -> str:
    """
    根据 FeatureConfig 和上下文构建 Wind API 参数字符串。

    wind_params 约定：
    - 空字符串 "" → 必须从 context 提供对应 key
    - 非空字符串 → 直接使用静态值

    Args:
        config: 特征配置
        context: 动态上下文字典

    Returns:
        str: Wind API 参数字符串，例如 "tradeDate=20250101;unit=1"
    """
    param_list = []
    for key, value in config.wind_params.items():
        try:
            if value == "":  # 动态值
                final_value = context[key]
            else:  # 静态值
                final_value = value
        except KeyError:
            print(f"[错误] (特征: {config.wind_field}) 构建 Wind 参数失败，缺少上下文 key: {key}")
            final_value = "<MISSING>"

        param_list.append(f"{key}={final_value}")

    return ";".join(param_list)


def group_features_for_api_call(features_cn: List[str], context: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """
    [核心] 将一批特征根据其所需的API和参数进行分组，以便批量调用。

    :param features_cn: 中文特征名列表。
    :param context: 包含动态值的字典，用于构建 options。
    :return: 一个字典，键为 (wind_api, options_string)，值为该组的中文特征名列表。
    """
    groups: Dict[Tuple[str, str], List[str]] = {}
    for name in features_cn:
        config = FEATURE_META.get(name)
        if not config or not config.wind_api:
            print(f"[警告] 特征 '{name}' 缺少 'wind_api' 配置，已跳过。")
            continue

        # 为当前特征生成唯一的 options 字符串
        options_str = build_wind_options(config, context)

        # 使用 (api, options) 作为分组的键
        group_key = (config.wind_api, options_str)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(name)

    return groups


# ------------------------- 6. 归一化函数 -------------------------
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


def clip_normalize(arr, min_val=-100.0, max_val=100.0):
    """
    裁剪归一化
    限制在指定的最小值和最大值之间。
    """
    arr = np.array(arr / max_val, dtype=float)
    return np.clip(arr, min_val, max_val)


# ==============================================================================
# =====                         程序执行入口 (示例)                          =====
# ==============================================================================

if __name__ == "__main__":

    print("=============== 1. 演示特征分组功能 ===============")

    # 模拟需要查询的特征列表
    features_to_query = [
        "营业收入(单季)",
        "毛利率(单季)",  # 财报特征
        "区间日均收盘价",
        "区间振幅",  # 股市特征
        "板块PE"  # 板块特征
    ]

    # 模拟动态上下文信息
    runtime_context = {"rptDate": "20240331", "startDate": "2024-01-01", "endDate": "2024-03-31", "trade_days": 60}

    # 调用分组函数
    api_call_groups = group_features_for_api_call(features_to_query, runtime_context)

    print(f"待查询特征: {features_to_query}")
    print(f"运行时上下文: {runtime_context}\n")
    print("智能分组后的API调用计划:")

    for (api, options), features in api_call_groups.items():
        wind_fields = translate_to_wind_fields(features)
        print("-" * 40)
        print(f"API类型: {api}")
        print(f"API参数: {options}")
        print(f"中文特征: {features}")
        print(f"Wind字段: {wind_fields}")
