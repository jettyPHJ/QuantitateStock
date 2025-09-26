from WindPy import w
from datetime import datetime, date, timedelta
from utils.block import Block
import time
import math
import utils.feature as ft
from typing import List, Dict, Any
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

    def get_update_data(self, row_data: dict, modules: List[str] = None) -> Dict:
        """
        根据已有行数据（统计开始日/结束日）抓取指定模块的区间统计信息，并返回更新字典。
        
        :param row_data: 包含 '统计开始日' 和 '统计结束日' 的行数据
        :param modules: 要更新的模块列表，可选值 'stock' 和 'block'。默认 None（不抓取任何模块）
        :return: 字典，包含抓取的数据
        """
        allowed_modules = {"stock", "block"}

        if modules is None:
            # 默认不抓取任何模块
            modules = []
        else:
            # 检查输入合法性
            invalid = set(modules) - allowed_modules
            if invalid:
                raise ValueError(f"[Error] modules 参数包含非法值: {invalid}. 可选值: {allowed_modules}")

        start_day = row_data.get("统计开始日")
        end_day = row_data.get("统计结束日")
        if not start_day or not end_day:
            return {}

        result = {}

        if "stock" in modules:
            try:
                stock_stats = self.get_stock_data(start_day, end_day)
                result.update(stock_stats)
            except Exception as e:
                print(f"[Warning] 股票区间数据抓取失败: {row_data} -> {e}")

        if "block" in modules:
            try:
                block_stats = self.get_block_data(start_day, end_day)
                result.update(block_stats)
            except Exception as e:
                print(f"[Warning] 板块区间数据抓取失败: {row_data} -> {e}")

        return result


# TODO：待进一步优化到单个指标，节约api调用
def fetch_data_for_period(stock_code: str, start_day: str, end_day: str, indicators: List[str],
                          block_code: str = None) -> Dict[str, Any]:
    """
    为指定股票和时间段，抓取一系列指定的指标数据。
    该函数会自动区分指标是属于个股还是板块，并调用相应的Wind接口。

    :param stock_code: 股票代码 (e.g., "NVDA.O")
    :param start_day: 统计开始日期 (e.g., "2023-01-01")
    :param end_day: 统计结束日期 (e.g., "2023-03-31")
    :param indicators: 需要抓取的指标名称列表 (e.g., ['换手率', '市盈率', '板块换手率'])
    :param block_code: 板块代码，如果需要抓取板块指标则必须提供
    :return: 一个包含所有成功抓取到的指标和其值的字典
    """
    if not indicators:
        return {}

    merged_data_map = {}
    start_day_int, end_day_int = int(start_day.replace("-", "")), int(end_day.replace("-", ""))

    # 1. 区分个股指标和板块指标 (依赖 ft 模块中的定义)
    stock_indicators_to_fetch = [name for name in indicators if name in ft.get_feature_names_by_source("股市")]
    block_indicators_to_fetch = [name for name in indicators if name in ft.get_feature_names_by_source("板块")]

    # 2. 抓取个股指标
    if stock_indicators_to_fetch:
        try:
            wss_trade_days = check_wind_data(
                w.wss(stock_code, "trade_days_per", f"startDate={start_day_int};endDate={end_day_int}"),
                context=f"stock_code:{stock_code},获取交易天数")
            [[trade_days]] = wss_trade_days.Data

            if trade_days and trade_days > 0:
                wss_stock_data = check_wind_data(
                    w.wss(stock_code, ft.stock_wind, ft.stock_wind_opt(trade_days, end_day_int, start_day_int)),
                    context=f"stock_code:{stock_code},获取区间股价统计信息")
                stock_data_map = ft.build_translated_data_map(wss_stock_data.Fields, wss_stock_data.Data)
                merged_data_map.update(stock_data_map)
        except Exception as e:
            print(f"[Warning] 抓取个股指标 for {stock_code} ({start_day} to {end_day}) 失败: {e}")

    # 3. 抓取板块指标
    if block_indicators_to_fetch:
        if not block_code:
            print(f"[Warning] 需要抓取板块指标 {block_indicators_to_fetch} 但未提供 block_code，已跳过。")
        else:
            try:
                year = int(start_day.split('-')[0])
                wsee_result = check_wind_data(
                    w.wsee(block_code, ft.block_wind, ft.block_wind_opt(start_day_int, end_day_int, year)),
                    context=f"block_code:{block_code},获取板块数据")
                block_data_map = ft.build_translated_data_map(wsee_result.Fields, wsee_result.Data)
                merged_data_map.update(block_data_map)
            except Exception as e:
                print(f"[Warning] 抓取板块指标 for {block_code} ({start_day} to {end_day}) 失败: {e}")

    return merged_data_map


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
    res = get_price_change_records("NVDA.O", block_code, "2025-08-01", "2025-08-20")
    print("res:", res)
