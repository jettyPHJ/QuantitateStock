import json
from datetime import datetime, timedelta
from .expectation_db import ExpectationDBManager
from .prompts import triage_prompts, industry_prompts, company_prompts, update_prompts
from data_process.news_data.script.gemini import GeminiAnalyzer
from utils.block import Block

# 配置：定义数据“陈旧”的阈值（天）
STALE_THRESHOLD_DAYS = 90


class ExpectationWorkflowEngine:

    def __init__(self):
        self.db_manager = ExpectationDBManager()
        self.analyzer = GeminiAnalyzer()

    # --- 新增的私有方法：用于执行“例行更新” ---
    def _perform_proactive_industry_update(self, industry_name: str):
        """对单个行业进行一次主动的、基于当前时间的预期更新"""
        print(f"🔄 Performing proactive refresh for industry '{industry_name}'...")
        as_of_date = datetime.now().strftime('%Y-%m-%d')
        prompt = industry_prompts.industry_future_prospects_prompt(industry_name, as_of_date)
        data = self.analyzer.call_llm(prompt)
        if data:
            self.db_manager.save_industry_expectation(industry_name, data)

    def _perform_proactive_company_update(self, stock_code: str, company_name: str, industry_names: list):
        """对单个公司进行一次主动的、基于当前时间的预期更新"""
        print(f"🔄 Performing proactive refresh for company '{company_name}'...")
        as_of_date = datetime.now().strftime('%Y-%m-%d')
        # 注意：这里我们简单地使用第一个行业作为代表，或者你可以设计更复杂的逻辑
        prompt = company_prompts.company_moat_trajectory_prompt(stock_code, company_name, as_of_date)
        data = self.analyzer.call_llm(prompt)
        if data:
            self.db_manager.save_company_expectation(stock_code, data)

    # --- 新增的私有方法：用于检查和触发“例行更新” ---
    def _check_and_refresh_expectations(self, stock_code: str, company_name: str, industry_names: list):
        """
        在处理新闻前，检查所有相关预期的时效性，如果陈旧则更新。
        """
        print("\n--- STEP 0: Checking data freshness ---")
        now = datetime.now()
        stale_threshold = now - timedelta(days=STALE_THRESHOLD_DAYS)

        # 1. 检查并更新关联的行业
        for industry_name in industry_names:
            latest_exp = self.db_manager.load_latest_industry_expectation(industry_name)
            if not latest_exp:
                print(f"No baseline found for industry '{industry_name}'. Performing initial creation.")
                self._perform_proactive_industry_update(industry_name)
            else:
                last_update_date = datetime.strptime(latest_exp['as_of_date'], '%Y-%m-%d')
                if last_update_date < stale_threshold:
                    print(
                        f"Industry '{industry_name}' expectation is stale (last updated on {latest_exp['as_of_date']})."
                    )
                    self._perform_proactive_industry_update(industry_name)
                else:
                    print(f"Industry '{industry_name}' expectation is fresh.")

        # 2. 检查并更新公司
        latest_comp_exp = self.db_manager.load_latest_company_expectation(stock_code)
        if not latest_comp_exp:
            print(f"No baseline found for company '{company_name}'. Performing initial creation.")
            self._perform_proactive_company_update(stock_code, company_name, industry_names)
        else:
            last_update_date = datetime.strptime(latest_comp_exp['as_of_date'], '%Y-%m-%d')
            if last_update_date < stale_threshold:
                print(
                    f"Company '{company_name}' expectation is stale (last updated on {latest_comp_exp['as_of_date']}).")
                self._perform_proactive_company_update(stock_code, company_name, industry_names)
            else:
                print(f"Company '{company_name}' expectation is fresh.")

    def handle_news_event(self, stock_code: str, company_name: str, industry_names: list, news_event: dict):
        """
        处理单条新闻事件的完整工作流（V5版）。
        """
        print(f"\n🔥🔥🔥 Handling news event for {company_name}: '{news_event['title']}' 🔥🔥🔥")

        # --- 【新】步骤 0: 检查并刷新陈旧数据 ---
        self._check_and_refresh_expectations(stock_code, company_name, industry_names)

        # --- 步骤 1: 分诊 (Triage) ---
        print("\n--- STEP 1: Triaging news event ---")
        # 注意：这里我们只使用第一个或最主要的行业进行分诊，以简化prompt
        primary_industry_name = industry_names[0]
        triage_prompt = triage_prompts.event_triage_and_routing_prompt(news_title=news_event['title'],
                                                                       news_summary=news_event['summary'],
                                                                       company_name=company_name,
                                                                       industry_name=primary_industry_name)
        triage_result = self.analyzer.call_llm(triage_prompt)

        if not triage_result:
            print("Triage failed. Aborting.")
            return

        update_target = triage_result.get("update_target", "None")
        print(
            f"🧠 Triage Decision: Significance={triage_result.get('significance_score')}, Update Target='{update_target}'"
        )

        # --- 步骤 2: 路由与执行 (Routing & Execution) ---
        if update_target == "None":
            print("➡️ Event is not significant. No event-driven update needed. Process finished.")
            return

        # --- 步骤 3: 执行事件驱动的更新 ---
        print(f"\n--- STEP 2: Performing event-driven update ('{update_target}') ---")
        # 此时加载的数据一定是最新的
        current_industry_exp = self.db_manager.load_latest_industry_expectation(primary_industry_name)
        current_company_exp = self.db_manager.load_latest_company_expectation(stock_code)

        prior_expectation = {"industry_prospects": current_industry_exp, "company_moat_trajectory": current_company_exp}

        update_prompt = update_prompts.event_impact_and_expectation_update_prompt(
            prior_expectation_json=json.dumps(prior_expectation, ensure_ascii=False,
                                              indent=2), news_event_title=news_event['title'],
            news_event_summary=news_event['summary'], event_date=news_event['date'])

        update_result = self.analyzer.call_llm(update_prompt)

        if update_result and "updated_expectation" in update_result:
            updated_data = update_result["updated_expectation"]

            # 保存更新后的数据，创建新版本
            if update_target == "Industry_and_Company" and "industry_prospects" in updated_data:
                self.db_manager.save_industry_expectation(primary_industry_name, updated_data["industry_prospects"])

            if "company_moat_trajectory" in updated_data:
                self.db_manager.save_company_expectation(stock_code, updated_data["company_moat_trajectory"])

            print("✅ Event-driven update successful.")
            impact_data = update_result.get("impact_quantification")
            print("📈 Impact Quantification:", impact_data)
        else:
            print("❌ Event-driven update failed.")
