from WindPy import w
from datetime import datetime, date, timedelta
from utils.block import Block
import time
import math
import utils.feature as ft
from typing import List
from utils.prompt import PriceChangeRecord

w.start()

# 数据提取起始时间
start_point = "2005-01-01"


def check_wind_data(wind_data, context=""):
    if wind_data.ErrorCode != 0:
        raise RuntimeError(f"[Wind ERROR] {context} 请求失败，错误码：{wind_data.ErrorCode}, fields: {wind_data.Fields}")
    return wind_data


class WindFinancialDataFetcher:

    def __init__(self, stock_code: str, block_code: str):
        self.stock_code = stock_code
        self.block_code = block_code
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

        finance_data_map = ft.build_translated_data_map(wss_result.Fields, wss_result.Data)

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

        stock_data_map = ft.build_translated_data_map(wss_result.Fields, wss_result.Data)

        return stock_data_map

    # 获取板块相关统计信息
    def get_block_data(self, start_day: str, end_day: str):
        start_day_int, end_day_int = int(start_day.replace("-", "")), int(end_day.replace("-", ""))
        year = int(start_day.split('-')[0])
        wsee_result = check_wind_data(
            w.wsee(self.block_code, ft.block_wind, ft.block_wind_opt(start_day_int, end_day_int, year)),
            context=f"stock_code:{self.stock_code},获取板块数据")

        block_data_map = ft.build_translated_data_map(wsee_result.Fields, wsee_result.Data)

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
def get_stock_codes(block_code: str):
    try:
        wset_result = check_wind_data(
            w.wset("sectorconstituent", f"date={datetime.now().date()};sectorid={block_code}"),
            context=f"获取 {block_code} 板块股票")

        result_list = []
        if wset_result.ErrorCode == 0 and len(wset_result.Data) > 0:
            result_list = ft.build_translated_data_map(wset_result.Fields, wset_result.Data)["wind_code"]
        return result_list
    except Exception as e:
        print(f"[Error] 获取板块 {block_code} 股票列表失败: {e}")
        return []


def get_price_change_records(
    stock_code: str,
    block_code: str = None,
    start_date: str = "",
    end_date: str = "",
) -> List[PriceChangeRecord]:
    """
    获取指定日期范围内某股票和/或板块的涨跌幅序列（含日期），合并成统一结构返回。
    自动跳过值为 None 或 nan 的数据。
    """
    if not stock_code and not block_code:
        raise ValueError("必须提供至少一个 stock_code 或 block_code")

    result_dict: dict[datetime.date, PriceChangeRecord] = {}

    # 处理个股涨跌幅
    if stock_code:
        options = f"Days=Alldays;PriceAdj=F"
        wsd_result = check_wind_data(w.wsd(stock_code, "pct_chg", start_date, end_date, options),
                                     context=f"获取 {stock_code} 涨跌幅")
        for d, val in zip(wsd_result.Times, wsd_result.Data[0]):
            if val is not None and not math.isnan(val):
                result_dict.setdefault(d, PriceChangeRecord(date=d))
                result_dict[d].stock_pct_chg = round(val, 2)
    # 处理板块涨跌幅
    if block_code:
        buffer_days = 7
        query_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        options = "Days=Alldays;DynamicTime=0"
        ws_result = w.wses(block_code, "sec_close_avg", query_start, end_date, options)

        if ws_result.ErrorCode != 0:
            raise RuntimeError(f"Wind请求失败，错误码 {ws_result.ErrorCode}")

        dates, prices = [], []
        for d, val in zip(ws_result.Times, ws_result.Data[0]):
            if val is not None and not math.isnan(val):
                dates.append(d)
                prices.append(val)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        index = next((i for i, d in enumerate(dates) if d >= start_dt), None)

        if index is None or index == 0:
            raise ValueError("无法定位 start_date 或缺少其前一个交易日数据")

        for i in range(index, len(dates)):
            prev_price = prices[i - 1]
            curr_price = prices[i]
            if prev_price is None or curr_price is None:
                continue
            pct_chg = round((curr_price - prev_price) / prev_price * 100, 2)
            date = dates[i]
            result_dict.setdefault(date, PriceChangeRecord(date=date))
            result_dict[date].block_pct_chg = pct_chg

    # 返回按日期排序后的列表
    return [result_dict[k] for k in sorted(result_dict.keys())]


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    block_code = Block.get("US_芯片").code
    fetcher = WindFinancialDataFetcher("NVDA.O", block_code)
    data = fetcher.get_data()
    print(data)
    print(get_stock_codes(block_code))
    res = get_price_change_records("NVDA.O", block_code, "2025-08-01", "2025-08-20")
    print("res:", res)
