from WindPy import w
from datetime import datetime, date, timedelta
import yaml
import os

# 财报数据
value_features_cn = ["营业收入(单季)", "营业收入(TTM)", "EBITDA(TTM)", "经营活动现金流(TTM)"]
ratio_features_cn = [
    "毛利率(单季)", "毛利率(TTM)", "净利率(单季)", "净利率(TTM)", "净资产收益率(单季)", "净资产收益率(TTM)", "总资产收益率(单季)", "总资产收益率(TTM)", "资产负债率", "总资产周转率"
]
# 股市数据
stock_feature_cn = ["区间成交均价", "区间日均换手率", "区间最高价", "区间最低价"]

# 板块数据
block_feature_cn = ["板块涨跌幅", "板块日均换手率"]

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


# 财报数据默认配置
features_wind = ",".join(translate_to_wind_fields(value_features_cn + ratio_features_cn))
financial_options = "unit=1;rptType=1;currencyType=;Period=Q;Days=Alldays;PriceAdj=F"

# 股价数据默认配置
stock_wind = ",".join(translate_to_wind_fields(stock_feature_cn))

# 板块数据默认配置
block_wind = ",".join(translate_to_wind_fields(block_feature_cn))


def check_wind_data(wind_data, context=""):
    if wind_data.ErrorCode != 0:
        raise RuntimeError(f"[Wind ERROR] {context} 请求失败，错误码：{wind_data.ErrorCode}, fields: {wind_data.Fields}")
    return wind_data


def build_translated_data_map(wind_fields: list[str], values: list[list]) -> dict:
    """
    将 Wind 字段名及对应数据转换为中文字段 → 值 的映射。
    """
    chinese_fields = translate_to_chinese_fields(wind_fields)
    return {ch_name: val[0] if isinstance(val, list) and len(val) == 1 else val for ch_name, val in zip(chinese_fields, values)}


class WindFinancialDataFetcher:

    def __init__(self, stock_code: str, block_code: str):
        w.start()
        self.stock_code = stock_code
        self.block_code = block_code
        self._report_dates = None

    def get_report_dates(self):
        if self._report_dates is not None:
            return self._report_dates

        query_end_date = datetime.now().date() + timedelta(days=500)
        outdata = check_wind_data(w.wsd(self.stock_code, "stm_issuingdate", "2005-01-01", query_end_date, self.options))
        pub_dates_raw = outdata.Data[0]
        report_dates_raw = outdata.Times

        if not pub_dates_raw or not report_dates_raw:
            print("[Warning] 没有获取到有效的发布日期数据")
            return [], []

        # 提取最后一段连续非 None 的数据区间
        n = len(pub_dates_raw)
        end_index = next((i for i in reversed(range(n)) if isinstance(pub_dates_raw[i], (datetime, date))), None)
        if end_index is None:
            print("[Warning] 未找到任何有效发布日期")
            return [], []

        start_index = end_index
        for i in range(end_index, -1, -1):
            if pub_dates_raw[i] is None:
                start_index = i + 1
                break
            start_index = i

        report_dates, pub_dates = [], []
        for i in range(start_index, end_index + 1):
            report_date = report_dates_raw[i]
            pub_date = pub_dates_raw[i]

            report_str = report_date.strftime("%Y-%m-%d") if isinstance(report_date, (datetime, date)) else "None"
            pub_str = pub_date.strftime("%Y-%m-%d") if isinstance(pub_date, (datetime, date)) else "None"

            report_dates.append(report_str)
            pub_dates.append(pub_str)

        print(f"提取最近连续财报记录：共 {len(report_dates)} 条，起止：{report_dates[0]} 到 {report_dates[-1]}")
        self._report_dates = (report_dates, pub_dates)
        return self._report_dates

    # 获取上市公司各季度财报数据
    def get_finance_data(self, rpt_date: str):
        date = int(rpt_date.replace("-", ""))

        wss_result = check_wind_data(w.wss(self.stock_code, features_wind, f"unit=1;rptDate={date};rptType=1"))

        finance_data_map = build_translated_data_map(wss_result.Fields, wss_result.Data)

        return finance_data_map

    # 获取区间内的股价相关统计信息
    def get_stock_data(self, start_day: str, end_day: str):
        start_day_int, end_day_int = int(start_day.replace("-", "")), int(end_day.replace("-", ""))

        wss_result = check_wind_data(w.wss(self.stock_code, stock_wind,
                                           f"startDate={start_day_int};endDate={end_day_int};priceAdj=F"),
                                     context="get_stock_data")

        stock_data_map = build_translated_data_map(wss_result.Fields, wss_result.Data)

        return stock_data_map

    # 获取板块相关统计信息
    def get_block_data(self, start_day: str, end_day: str):
        start_day_int, end_day_int = int(start_day.replace("-", "")), int(end_day.replace("-", ""))

        wsee_result = check_wind_data(w.wsee(self.block_code, block_wind,
                                             f"startDate={start_day_int};endDate={end_day_int};DynamicTime=1"),
                                      context="get_block_data")

        block_data_map = build_translated_data_map(wsee_result.Fields, wsee_result.Data)

        return block_data_map


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    fetcher = WindFinancialDataFetcher(stock_code="NVDA.O", block_code="1000041891000000")

    # fetcher.get_finance_data("2025-03-31")

    fetcher.get_block_data("2025-06-24", "2025-07-20")
