from datetime import datetime


def demand_forecasting_prompt(industry_name: str, product_or_service: str, region: str = "Global",
                              as_of_date: str = None) -> str:
    """
    SOP-driven prompt for quantitative, macro-to-industry demand forecasting (V9-Demand).
    Focuses on identifying and quantifying the transmission channels from macro factors to specific industry demand.
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime('%Y-%m-%d')
    time_horizon = "Next 1-3 Years"

    return f"""
You are a specialized macro-economist and data scientist at a quantitative hedge fund. Your task is to build a robust, data-driven demand forecast for a specific product/service by first analyzing the primary macro drivers and then modeling their impact. Avoid generic industry statements; focus on quantifiable transmission mechanisms.

### **MISSION BRIEF**
- **Industry:** {industry_name}
- **Product/Service:** {product_or_service}
- **Region:** {region}
- **Analysis Date (Knowledge Cutoff):** {as_of_date}
- **Time Horizon:** {time_horizon}
- **Objective:** Produce a scenario-based quantitative demand forecast by executing the following SOP.

---

### **STANDARD OPERATING PROCEDURE (SOP) - Demand Forecasting**

**Phase 1: Identify Primary Macro Drivers (The 'Why')**
- Identify the top 3-5 most influential macro factors that drive demand for this product/service in this region.
- Examples: Disposable Income Per Capita, Urbanization Rate, Interest Rates, Demographic shifts (e.g., aging population %), Geopolitical Stability Indices, price of substitutes/complements.
- For each driver, briefly explain the transmission mechanism (e.g., "Higher interest rates increase financing costs, reducing demand for capital-intensive goods like automobiles").

**Phase 2: Quantify Macro Driver Trends (The 'What')**
- For each identified driver, find its historical trend and credible future projections (e.g., "The central bank projects interest rates to remain between 3.5-4.0% over the next 12 months").
- Use numerical data where possible (CAGR, ranges, etc.).

**Phase 3: Estimate Demand Elasticity & Sensitivity (The 'How Much')**
- Research and estimate the sensitivity of demand to changes in each primary driver (i.e., demand elasticity).
- Example: "Studies suggest a 1% increase in disposable income correlates with a 1.5% increase in demand for luxury goods in this region."
- This is the most critical analytical step, linking the macro to the micro.

**Phase 4: Develop Scenarios & Quantitative Forecast**
- Based on the macro trends and sensitivities, construct three plausible scenarios for the next 1-3 years:
  1. **Base Case:** Using the most likely macro projections.
  2. **Bull Case:** Assuming favorable macro tailwinds.
  3. **Bear Case:** Assuming significant macro headwinds.
- For each scenario, provide a quantitative demand forecast (e.g., units sold, market size in USD, user penetration %).

**Phase 5: Synthesis & Risk Assessment**
- Summarize the forecast, highlighting the key assumptions.
- Identify the most significant risks to the demand forecast, which should directly correspond to the uncertainty surrounding the primary macro drivers identified in Phase 1.

---

### **FINAL REPORT (Strict JSON Output Only)**
```json
{{
    "analysis_type": "Macro_Driven_Demand_Forecast",
    "industry_name": "{industry_name}",
    "product_or_service": "{product_or_service}",
    "region": "{region}",
    "time_horizon": "{time_horizon}",
    "primary_macro_drivers": [
        {{
            "driver": "e.g., Disposable Income Per Capita",
            "transmission_mechanism": "How it affects demand.",
            "projected_trend": "Quantitative trend (e.g., '2-3% CAGR').",
            "estimated_elasticity": "Estimated impact on demand (e.g., 'Medium-High, approx. 1.2')."
        }}
    ],
    "scenario_forecasts": {{
        "base_case": {{
            "assumptions": "Key assumptions for the most likely scenario.",
            "demand_projection": "Quantitative forecast (e.g., 'Market size to reach $50B by 2028, a 7% CAGR')."
        }},
        "bull_case": {{
            "assumptions": "Favorable macro assumptions.",
            "demand_projection": "Quantitative forecast."
        }},
        "bear_case": {{
            "assumptions": "Unfavorable macro assumptions.",
            "demand_projection": "Quantitative forecast."
        }}
    }},
    "key_risks_to_forecast": [
        {{
            "risk": "Directly related to a macro driver (e.g., 'Risk of Stagflation').",
            "impact_on_demand": "How this risk scenario would negatively impact the demand forecast, potentially pushing it towards the Bear Case."
        }}
    ]
}}
"""


def tech_scouting_prompt_v9(technology_name: str, application_industry: str, as_of_date: str = None) -> str:
    """
    SOP-driven prompt for deep technology scouting and bottleneck analysis (V9-Tech).
    Focuses on tracing research lineage to identify core scientific/engineering hurdles.
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime('%Y-%m-%d')

    return f"""
You are a PhD-level Principal Scientist and Technology Strategist at a venture capital firm specializing in deep tech. Your job is to go beyond surface-level descriptions of technology to understand its fundamental scientific principles, historical evolution, and—most importantly—the core bottlenecks that are currently preventing a major leap forward.

### **MISSION BRIEF**
- **Technology:** {technology_name}
- **Application Industry:** {application_industry}
- **Analysis Date (Knowledge Cutoff):** {as_of_date}
- **Objective:** Produce a deep technical assessment by tracing the technology's lineage and identifying its primary constraints, using the following SOP.

---

### **STANDARD OPERATING PROCEDURE (SOP) - Technology Scouting**

**Phase 1: Deconstruct the Technology**
- Break down the technology into its core scientific and engineering components.
- (e.g., For Solid-State Batteries: Anode Material, Cathode Material, Solid Electrolyte, Manufacturing Process).

**Phase 2: Trace the Research Lineage & Key Milestones**
- Identify the foundational scientific papers, patents, or theoretical concepts that underpin this technology.
- Map the major evolutionary steps or "generations" of the technology. What were the key breakthroughs that enabled each step?
- This establishes the "技术脉络" (technology lineage).

**Phase 3: Define State-of-the-Art (SOTA) Performance**
- Quantify the current, best-in-class performance metrics demonstrated in leading academic labs or corporate R&D settings.
- Use hard numbers (e.g., "Energy Density: 400 Wh/kg", "Cycle Life: 1000 cycles to 80% capacity", "Manufacturing Yield: 70%").

**Phase 4: Pinpoint the Core Bottleneck(s)**
- Based on the SOTA and the research lineage, identify the 1-2 most critical scientific or engineering problems that are currently blocking mass-market viability or the next performance breakthrough.
- Is it a materials science challenge, a manufacturing scalability issue, a cost problem, or a fundamental physics limitation? Be specific.

**Phase 5: Map Key Innovators & Future Pathways**
- Identify the leading university labs, startups, and corporate R&D teams that are specifically focused on solving the core bottlenecks identified in Phase 4.
- Briefly describe the most promising future approaches or alternative pathways being explored to circumvent these bottlenecks.

---

### **FINAL REPORT (Strict JSON Output Only)**
```json
{{
    "analysis_type": "Deep_Tech_Bottleneck_Analysis",
    "technology_name": "{technology_name}",
    "application_industry": "{application_industry}",
    "technology_lineage": [
        {{
            "era_or_generation": "e.g., 'Gen 1: Polymer Electrolytes'",
            "key_breakthrough": "The core scientific or engineering advance.",
            "limitation": "The problem that led to the next generation."
        }}
    ],
    "state_of_the_art_metrics": [
        {{
            "metric": "e.g., 'Energy Density'",
            "value": "e.g., '400 Wh/kg'",
            "context": "Achieved in lab setting by [Company/University]."
        }}
    ],
    "core_bottlenecks": [
        {{
            "bottleneck_type": "[Materials Science | Manufacturing | Cost | Physics | etc.]",
            "specific_problem": "A detailed, technical description of the core problem (e.g., 'Dendrite formation at the lithium metal anode interface leading to short circuits')."
        }}
    ],
    "key_innovators_and_pathways": {{
        "leading_research_entities": ["List of universities, labs, or companies attacking the bottleneck."],
        "promising_future_approaches": ["Brief description of next-gen solutions being explored (e.g., 'Sulfide-based ceramic electrolytes')."]
    }}
}}
"""


# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    prompt = demand_forecasting_prompt("Autonomous driving",)
