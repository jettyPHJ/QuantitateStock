from workflow_engine import ExpectationWorkflowEngine
from datetime import datetime
from utils.block import Block  # 假设 Block 可以这样导入


def run_for_sector(sector_name_cn: str):
    """为一个板块下的所有核心公司运行工作流"""

    engine = ExpectationWorkflowEngine()

    # 使用你的 Block 工具获取板块信息
    sector_item = Block.get(sector_name_cn)
    if not sector_item:
        print(f"Error: Sector '{sector_name_cn}' not found.")
        return

    # ⚠️ 假设 get_stock_codes 返回一个字典 {'code': 'name'}
    from data_process.news_data.script.gemini import get_stock_codes
    core_tickers = get_stock_codes(sector_item.id)

    if not core_tickers:
        print(f"No core tickers found for sector '{sector_name_cn}'.")
        return

    for stock_code, company_name in core_tickers.items():
        # 假设新闻事件，实际应用中你会从新闻源获取
        news_event = {
            "title": f"{company_name} announces new AI chip 'Archon' to double performance.",
            "summary": f"In a press release, {company_name} unveiled its next-generation AI accelerator, the 'Archon', promising a 100% performance uplift over the previous generation with a 30% reduction in power consumption. The chip is expected to sample in Q1 next year.",
            "date": datetime.now().strftime('%Y-%m-%d')
        }

        # 传入公司和行业信息，处理事件
        engine.handle_news_event(
            stock_code=stock_code,
            company_name=company_name,
            industry_name=sector_name_cn,  # 使用中文名作为行业标识
            news_event=news_event)


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    # 以“半导体产品”板块为例
    target_sector = "半导体产品"
    run_for_sector(target_sector)
