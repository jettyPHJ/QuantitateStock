from WindPy import w
from datetime import datetime, date, timedelta
import yaml
import os
from enum import Enum
import time
import math
import data_process.finance_data.feature as ft

w.start()

# 数据提取起始时间
start_point = "2005-01-01"


# 板块枚举值
class BlockCode(Enum):
    US_CHIP = ("[US]芯片", "1000041891000000")
    US_SEMI = ("[US]半导体", "1000041892000000")
    US_AUTO = ("[US]无人驾驶", "1000041895000000")
    NASDAQ_Computer_Index = ("纳斯达克计算机指数", "1000010326000000")

    def __init__(self, desc: str, code: str):
        self.desc = desc
        self.code = code


base_dir = os.path.dirname(__file__)
feature_map_path = os.path.join(base_dir, "feature_map.yaml")
with open(feature_map_path, "r", encoding="utf-8") as f:
    FEATURE_NAME_MAP = yaml.safe_load(f)
    REVERSE_MAP = {v: k for k, v in FEATURE_NAME_MAP.items()}


def check_wind_data(wind_data, context=""):
    if wind_data.ErrorCode != 0:
        raise RuntimeError(f"[Wind ERROR] {context} 请求失败，错误码：{wind_data.ErrorCode}, fields: {wind_data.Fields}")
    return wind_data


def build_translated_data_map(wind_fields: list[str], values: list[list]) -> dict:
    """
    将 Wind 字段名及对应数据转换为 中文字段 → 值 的映射。
    如果值是 datetime/date 类型，则转为 'yyyy-mm-dd' 字符串。
    空值和 NaN 将被转换为空字符串。
    """
    chinese_fields = ft.translate_to_chinese_fields(wind_fields)
    result = {}

    for ch_name, val in zip(chinese_fields, values):
        v = val[0] if isinstance(val, list) and len(val) == 1 else val

        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = ""
        elif isinstance(v, (datetime, date)):
            v = v.strftime("%Y-%m-%d")

        result[ch_name] = v

    return result


class WindFinancialDataFetcher:

    def __init__(self, stock_code: str, block_code: BlockCode):
        self.stock_code = stock_code
        self.block_code = block_code.code
        self._report_dates = None

    def get_report_dates(self):
        if self._report_dates is not None:
            return self._report_dates

        query_end_date = datetime.now().date() + timedelta(days=500)
        outdata = check_wind_data(
            w.wsd(self.stock_code, "stm_issuingdate", start_point, query_end_date, "Period=Q;Days=Alldays"),
            context=f"stock_code:{self.stock_code},获取日期序列")
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

        print(f"提取{self.stock_code}最近连续财报记录：共 {len(report_dates)} 条，起止：{report_dates[0]} 到 {report_dates[-1]}")
        self._report_dates = (report_dates, pub_dates)
        return self._report_dates

    # 获取上市公司各季度财报数据
    def get_finance_data(self, rpt_date: str):
        date = int(rpt_date.replace("-", ""))

        wss_result = check_wind_data(w.wss(self.stock_code, ft.features_wind, ft.features_wind_opt(date)),
                                     context=f"stock_code:{self.stock_code},获取财报数据")

        finance_data_map = build_translated_data_map(wss_result.Fields, wss_result.Data)

        return finance_data_map

    # 获取区间内的股价相关统计信息
    def get_stock_data(self, start_day: str, end_day: str):
        start_day_int, end_day_int = int(start_day.replace("-", "")), int(end_day.replace("-", ""))

        # 提取股票交易天数
        wss_result = check_wind_data(
            w.wss(self.stock_code, "trade_days_per", f"startDate={start_day_int};endDate={end_day_int}"),
            context=f"stock_code:{self.stock_code},获取交易天数")
        [[trade_days]] = wss_result.Data

        wss_result = check_wind_data(
            w.wss(self.stock_code, ft.stock_wind, ft.stock_wind_opt(trade_days, end_day_int, start_day_int)),
            context=f"stock_code:{self.stock_code},获取区间股价统计信息")

        stock_data_map = build_translated_data_map(wss_result.Fields, wss_result.Data)

        return stock_data_map

    # 获取板块相关统计信息
    def get_block_data(self, start_day: str, end_day: str):
        start_day_int, end_day_int = int(start_day.replace("-", "")), int(end_day.replace("-", ""))
        year = int(start_day.split('-')[0])
        wsee_result = check_wind_data(
            w.wsee(self.block_code, ft.block_wind, ft.block_wind_opt(start_day_int, end_day_int, year)),
            context=f"stock_code:{self.stock_code},获取板块数据")

        block_data_map = build_translated_data_map(wsee_result.Fields, wsee_result.Data)

        return block_data_map

    # 获取单支股票所有信息
    def get_data(self):
        report_dates, pub_dates = self.get_report_dates()

        all_data = []

        for i in range(1, len(report_dates)):
            report_date = report_dates[i]
            pub_date = pub_dates[i]
            prev_pub_date = pub_dates[i - 1]  # 上一期发布日期

            start_day = prev_pub_date
            end_day = pub_date

            try:
                finance_data = self.get_finance_data(report_date)
                stock_data = self.get_stock_data(start_day, end_day)
                block_data = self.get_block_data(start_day, end_day)

                merged = {
                    "报告期": report_date, "发布日期": pub_date, "统计开始日": start_day, "统计结束日": end_day, **finance_data,
                    **stock_data, **block_data
                }

                all_data.append(merged)

            except Exception as e:
                print(f"[Warning] 数据抓取失败：报告期 {report_date} -> {e}")

            finally:
                # 成功或失败都 sleep
                time.sleep(0.05)

        return all_data


# 获取指定板块所有的股票代码
def get_stock_codes(block_code: BlockCode):
    wset_result = check_wind_data(
        w.wset("sectorconstituent", f"date={datetime.now().date()};sectorid={block_code.code}"),
        context=f"获取 {block_code.desc} 板块股票")

    result_list = build_translated_data_map(wset_result.Fields, wset_result.Data)["wind_code"]

    return result_list


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    fetcher = WindFinancialDataFetcher(stock_code="NVDA.O", block_code=BlockCode.US_CHIP)
    data = fetcher.get_data()
    print(data)
    print(get_stock_codes(BlockCode.US_CHIP))
