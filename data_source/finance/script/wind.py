from WindPy import w
from datetime import datetime, date, timedelta
import time
import math
from typing import List, Dict, Any, Protocol
import utils.feature as ft
from data_source.finance.script.block import Block
from utils.prompt import PriceChangeRecord

w.start()

# 数据提取起始时间
start_point = "2024-06-01"


class WindData(Protocol):
    """
    一个协议类，用于为 Wind API 返回的对象提供类型提示。
    描述了代码中实际使用到的该对象的属性结构。
    """
    ErrorCode: int
    Data: List[List[Any]]
    Fields: List[str]
    Times: List[datetime]


def check_wind_data(wind_data: WindData, context="") -> WindData:
    """检查Wind返回数据，若有错误则抛出异常。"""
    if wind_data.ErrorCode != 0:
        # 打印更详细的错误信息，便于调试
        error_msg = f"[Wind ERROR] {context} 请求失败, Code: {wind_data.ErrorCode}, Fields: {getattr(wind_data, 'Fields', 'N/A')}"
        raise RuntimeError(error_msg)
    return wind_data


def fetch_data_from_wind(stock_code: str, block_code: str, features_cn: List[str],
                         context: Dict[str, Any]) -> Dict[str, Any]:
    """
    [核心函数] 为指定股票和时间段，根据特征配置智能地抓取一系列指标。
    
    该函数利用 ft.group_features_for_api_call 将不同API和参数的特征分组，
    实现最高效的批量调用。

    :param stock_code: 股票代码 (e.g., "NVDA.O")
    :param block_code: 板块代码 (e.g., "1000015221000000")
    :param features_cn: 需要抓取的中文特征名列表
    :param context: 包含所有动态参数的上下文，如 startDate, endDate, rptDate 等
    :return: 包含所有成功抓取到的 {特征名: 值} 的字典
    """
    if not features_cn:
        return {}

    # 1. 根据API和参数要求，对特征进行智能分组
    api_call_groups = ft.group_features_for_api_call(features_cn, context)

    final_data_map = {}

    # 2. 遍历每个组，执行一次API调用
    for (api, options), group_features_cn in api_call_groups.items():
        # 将中文名列表转换为Wind字段列表
        wind_fields = ft.translate_to_wind_fields(group_features_cn)
        fields_str = ",".join(wind_fields)

        # 根据API类型选择合适的Wind函数和代码
        api_func = getattr(w, api, None)
        code = block_code if api in ['wsee', 'wset'] else stock_code

        if not api_func or not code:
            print(f"[警告] API '{api}' 不支持或代码缺失，跳过特征: {group_features_cn}")
            continue

        try:
            # 统一调用接口
            wind_result = check_wind_data(api_func(code, fields_str, options), context=f"API={api}, Code={code}")

            # 将返回结果转换为 {中文名: 值} 的字典并合并
            group_data_map = ft.build_translated_data_map(wind_result.Fields, wind_result.Data)
            final_data_map.update(group_data_map)

        except Exception as e:
            print(f"[警告] 抓取数据失败 for group {group_features_cn}. Error: {e}")

    return final_data_map


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

    # 获取单支股票所有信息
    def get_data(self) -> List[Dict[str, Any]]:
        """
        获取单支股票在所有报告期的全量特征数据。
        """
        report_dates, pub_dates = self.get_report_dates()
        if not report_dates:
            return []

        all_data = []

        features_to_fetch = (ft.get_feature_names_by_source("财报") + ft.get_feature_names_by_source("股市") +
                             ft.get_feature_names_by_source("板块"))

        for i in range(1, len(report_dates)):
            report_date_str: str = report_dates[i]
            pub_date_str: str = pub_dates[i]
            prev_pub_date_str: str = pub_dates[i - 1]

            try:
                # --- 新增逻辑：提前获取交易日数 ---
                start_day_int = int(prev_pub_date_str.replace("-", ""))
                end_day_int = int(pub_date_str.replace("-", ""))

                wss_trade_days = check_wind_data(
                    w.wss(self.stock_code, "trade_days_per", f"startDate={start_day_int};endDate={end_day_int}"),
                    context=f"stock_code:{self.stock_code}, 获取交易天数")
                # 安全取值，拿不到就给 63
                ndays_int = -(wss_trade_days.Data[0][0] if wss_trade_days.Data and wss_trade_days.Data[0] else 63)

                context = {
                    "rptDate": report_date_str.replace("-", ""),
                    "startDate": str(start_day_int),
                    "endDate": str(end_day_int),
                    "ndays": str(ndays_int),
                    "tradeDate": pub_date_str.replace("-", ""),
                    "year": datetime.strptime(report_date_str, "%Y-%m-%d").year,
                }

                fetched_data = fetch_data_from_wind(self.stock_code, self.block_code, features_to_fetch, context)

                record = {
                    "报告期": report_date_str, "发布日期": pub_date_str, "统计开始日": prev_pub_date_str, "统计结束日": pub_date_str,
                    **fetched_data
                }
                all_data.append(record)

            except Exception as e:
                print(f"[Warning] 数据抓取失败：报告期 {report_date_str} -> {e}")

            finally:
                time.sleep(0.05)

        return all_data


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

    block_code = Block.get("US_芯片").id
    stock_code = "NVDA.O"

    choice = 1

    if choice == 1:
        fetcher = WindFinancialDataFetcher(stock_code, block_code)
        data = fetcher.get_data()
        print("get_data:", data)

    elif choice == 2:
        res = get_price_change_records(stock_code, block_code, "2025-08-01", "2025-08-20")
        print("get_price_change_records:", res)

    elif choice == 3:
        data = fetch_data_from_wind(stock_code, block_code, ["营业收入(单季)"], {"rptDate": "2024-12-31"})
        print("fetch_data_for_period:", data)

    else:
        print("无效的 choice 值，请选择 1 / 2 / 3")
