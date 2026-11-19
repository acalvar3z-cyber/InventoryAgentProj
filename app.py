"""
app.py
------
Interactive Entry Point for the Supply Chain Analytics Inventory Decision Agent.

Supports:
  1. Interactive Command-Line Interface (CLI) for natural-language questions
  2. Automated Benchmark & Demonstration suite across all 5 inventory models:
       - Economic Order Quantity (EOQ)
       - Quantity Discount Model
       - Safety Stock and Reorder Point (SS & ROP)
       - Fixed-Period Review (Order-Up-To Level)
       - Newsvendor Single-Period Model
  3. Interactive Streamlit Web UI (when run via `streamlit run app.py`)
"""

import sys
import os
from typing import Dict, Any

from inventory_agent import InventoryDecisionAgent, InventoryModelType


BENCHMARK_SCENARIOS = [
    {
        "title": "Scenario 1: Economic Order Quantity (EOQ)",
        "question": (
            "Our manufacturing facility has an annual demand of 12,000 units. "
            "Each order incurs a setup cost of $50, and our annual holding cost is $2 per unit per year. "
            "Each unit costs $10. What is our optimal order quantity and how often should we order?"
        ),
    },
    {
        "title": "Scenario 2: Quantity Discount Analysis",
        "question": (
            "Annual demand is 10,000 units and ordering cost is $100 per order. "
            "Carrying cost is 20% of item cost. The supplier offers the following price brackets: "
            "1 to 499 units at $5.00, 500 to 999 units at $4.50, and 1000 and above at $4.00. "
            "Which quantity discount bracket should we choose to minimize total costs?"
        ),
    },
    {
        "title": "Scenario 3: Safety Stock & Reorder Point (Continuous Review)",
        "question": (
            "Daily demand averages 100 units with a standard deviation of 20 units. "
            "Supplier lead time is 4 days. Management targets a 95% cycle service level. "
            "What should our safety stock and reorder point be?"
        ),
    },
    {
        "title": "Scenario 4: Fixed-Period Review (Periodic Order-Up-To Level)",
        "question": (
            "We operate a periodic review inventory policy reviewed every 7 days with a supplier lead time of 3 days. "
            "Average daily demand is 50 units with a standard deviation of 10 units. "
            "With a desired service level of 95%, our current on-hand inventory is 120 units and zero on-order. "
            "What is our target base stock level and how many units should we order today?"
        ),
    },
    {
        "title": "Scenario 5: Newsvendor Model (Single-Period Perishable Inventory)",
        "question": (
            "We are procuring a seasonal winter jacket for a single-period retail season. "
            "The retail selling price is $120, unit wholesale cost is $50, and unsold jackets have a salvage value of $20. "
            "Customer demand is normally distributed with a mean of 500 units and standard deviation of 80 units. "
            "What is the optimal order quantity, critical fractile, and expected profit?"
        ),
    },
]


def print_banner():
    banner = r"""
================================================================================
     SUPPLY CHAIN ANALYTICS INVENTORY DECISION AGENT (BAN 517)
================================================================================
 Pipeline Architecture:
   [Customer Question] 
          ↓ 
   [Identify Model] 
          ↓ 
   [Extract Inputs] 
          ↓ 
   [Select Python Tool] 
          ↓ 
   [Calculate (Python)] 
          ↓ 
   [Verify] 
          ↓ 
   [Explain]
================================================================================
 Supported Models:
   1. Economic Order Quantity (EOQ)
   2. Quantity Discount Model
   3. Safety Stock & Reorder Point (SS & ROP)
   4. Fixed-Period Review (Base Stock / Order-Up-To Level)
   5. Newsvendor (Single-Period Stochastic Critical Fractile)
================================================================================
"""
    print(banner)


def run_benchmark_suite(agent: InventoryDecisionAgent):
    print("\n" + "=" * 80)
    print(" >>> RUNNING AUTOMATED BENCHMARK SUITE ACROSS ALL 5 MODELS <<<")
    print("=" * 80 + "\n")

    for i, scenario in enumerate(BENCHMARK_SCENARIOS, 1):
        print(f"\n{'#' * 80}")
        print(f" {scenario['title']}")
        print(f"{'#' * 80}")
        print(f"\n[Customer Question]:\n\"{scenario['question']}\"\n")

        response = agent.process_question(scenario["question"])

        print("[Agent Pipeline Execution Audit Trace]:")
        for trace in response.traces:
            print(f"  [{trace.stage_name:20}] Status: {trace.status}")

        print(f"\n[Identified Model]: {response.model_type.value}")
        print(f"[Extracted Inputs]: {response.extracted_inputs}")
        print(f"[Verification Status]: {response.verification_status['status']}")

        print("\n" + "-" * 80)
        print(response.explanation)
        print("-" * 80)


def run_interactive_cli(agent: InventoryDecisionAgent):
    print_banner()
    print("Type your natural language inventory question below.")
    print("Commands:")
    print("  'demo' or 'test'  : Run the full 5-scenario textbook benchmark suite")
    print("  'help'            : Show sample questions for each model")
    print("  'quit' or 'exit'  : Exit the application\n")

    while True:
        try:
            user_input = input("\nInventory Agent > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Thank you!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["quit", "exit", "q"]:
            print("Session ended. Goodbye!")
            break

        if cmd in ["demo", "test", "run"]:
            run_benchmark_suite(agent)
            continue

        if cmd == "help":
            print("\nSAMPLE QUESTIONS YOU CAN ASK:")
            for sc in BENCHMARK_SCENARIOS:
                print(f"\n- {sc['title']}:\n  \"{sc['question']}\"")
            continue

        # Process user question through 7-stage pipeline
        response = agent.process_question(user_input)

        print("\n" + "=" * 80)
        print(f" MODEL IDENTIFIED: {response.model_type.value}")
        print(f" PIPELINE VERIFICATION: {response.verification_status['status']}")
        print("=" * 80)
        print(response.explanation)
        print("=" * 80)


def analyze_inventory_dataframe(df: Any) -> Dict[str, Any]:
    """
    Performs deterministic pandas analytics on an uploaded inventory dataset.
    Validates required columns: Product, Category, Units, Unit Cost.
    Calculates:
      1. Total Units
      2. Average Unit Cost
      3. Product with highest Units
      4. Total Units grouped by Category
      5. Inventory Value = Units * Unit Cost
      6. Total Inventory Value across all products
    """
    import pandas as pd

    required_columns = ["Product", "Category", "Units", "Unit Cost"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "found_columns": list(df.columns),
            "rows": len(df),
            "columns": len(df.columns),
        }

    df_calc = df.copy()
    df_calc["Units"] = pd.to_numeric(df_calc["Units"], errors="coerce").fillna(0)
    df_calc["Unit Cost"] = pd.to_numeric(df_calc["Unit Cost"], errors="coerce").fillna(0.0)

    # 1. Calculate total Units
    total_units = float(df_calc["Units"].sum())

    # 2. Calculate average Unit Cost
    avg_unit_cost = float(df_calc["Unit Cost"].mean()) if len(df_calc) > 0 else 0.0

    # 3. Identify the Product with the highest Units
    if not df_calc.empty:
        max_units_idx = df_calc["Units"].idxmax()
        product_highest_units = str(df_calc.loc[max_units_idx, "Product"])
        highest_units_val = float(df_calc.loc[max_units_idx, "Units"])
    else:
        product_highest_units = "N/A"
        highest_units_val = 0.0

    # 4. Group data by Category and calculate total Units for each Category
    category_units_df = (
        df_calc.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
    )

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    df_calc["Inventory Value"] = df_calc["Units"] * df_calc["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = float(df_calc["Inventory Value"].sum())

    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "product_highest_units": product_highest_units,
        "highest_units_val": highest_units_val,
        "category_units_df": category_units_df,
        "total_inventory_value": total_inventory_value,
        "analyzed_df": df_calc,
    }


def classify_multicriteria_inventory(
    df: Any,
    sales_thresholds: Tuple[float, float, float] = (0.70, 0.90, 0.98),
    hits_thresholds: Tuple[float, float, float] = (0.70, 0.90, 0.98),
) -> Dict[str, Any]:
    """
    Performs deterministic multicriteria inventory classification into A, B, C, and D classes
    using Sales ($) and Hits.
    Validates required columns:
      - Item Number
      - Item Name
      - UoM
      - Sales ($) (or Sales)
      - Hits
    Returns:
      - valid: bool
      - missing_columns: list
      - rows: int
      - columns: int
      - column_names: list
      - total_sales: float
      - total_hits: int
      - class_summary_df: pd.DataFrame
      - crosstab_df: pd.DataFrame
      - classified_df: pd.DataFrame
    """
    import pandas as pd

    cols = list(df.columns)

    # Check Item Number
    item_num_col = None
    for cand in ["Item Number", "Item #", "Item No", "SKU", "Item ID", "Item"]:
        if cand in cols:
            item_num_col = cand
            break

    # Check Item Name
    item_name_col = None
    for cand in ["Item Name", "Description", "Product", "Product Name", "Name"]:
        if cand in cols:
            item_name_col = cand
            break

    # Check UoM
    uom_col = None
    for cand in ["UoM", "UOM", "Unit of Measure", "Unit"]:
        if cand in cols:
            uom_col = cand
            break

    # Check Sales
    sales_col = None
    for cand in ["Sales ($)", "Sales", "Total Sales", "Annual Sales", "Revenue"]:
        if cand in cols:
            sales_col = cand
            break

    # Check Hits
    hits_col = None
    for cand in ["Hits", "Pick Hits", "Order Hits", "Transaction Hits", "Picks"]:
        if cand in cols:
            hits_col = cand
            break

    missing = []
    if not item_num_col:
        missing.append("Item Number")
    if not item_name_col:
        missing.append("Item Name")
    if not uom_col:
        missing.append("UoM")
    if not sales_col:
        missing.append("Sales ($)")
    if not hits_col:
        missing.append("Hits")

    if missing:
        return {
            "valid": False,
            "missing_columns": missing,
            "found_columns": cols,
            "rows": len(df),
            "columns": len(cols),
        }

    df_calc = df.copy()

    # Clean and parse Sales
    if df_calc[sales_col].dtype == object:
        df_calc[sales_col] = (
            df_calc[sales_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    df_calc[sales_col] = pd.to_numeric(df_calc[sales_col], errors="coerce").fillna(0.0)

    # Clean and parse Hits
    if df_calc[hits_col].dtype == object:
        df_calc[hits_col] = (
            df_calc[hits_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    df_calc[hits_col] = pd.to_numeric(df_calc[hits_col], errors="coerce").fillna(0).astype(int)

    def assign_tier(val: float, cutoffs: Tuple[float, float, float]) -> str:
        if val <= cutoffs[0]:
            return "A"
        elif val <= cutoffs[1]:
            return "B"
        elif val <= cutoffs[2]:
            return "C"
        return "D"

    # Calculate Sales Class using Pareto cumulative %
    total_sales = float(df_calc[sales_col].sum())
    if total_sales > 0:
        df_sales_sorted = df_calc.sort_values(by=sales_col, ascending=False)
        cum_sales = df_sales_sorted[sales_col].cumsum()
        cum_sales_pct = cum_sales / total_sales
        df_calc["Sales Class"] = cum_sales_pct.apply(lambda x: assign_tier(x, sales_thresholds))
    else:
        df_calc["Sales Class"] = "D"

    # Calculate Hits Class using Pareto cumulative %
    total_hits = int(df_calc[hits_col].sum())
    if total_hits > 0:
        df_hits_sorted = df_calc.sort_values(by=hits_col, ascending=False)
        cum_hits = df_hits_sorted[hits_col].cumsum()
        cum_hits_pct = cum_hits / total_hits
        df_calc["Hits Class"] = cum_hits_pct.apply(lambda x: assign_tier(x, hits_thresholds))
    else:
        df_calc["Hits Class"] = "D"

    # Joint Criteria Decision Matrix Mapping (Flores & Whybark / Multicriteria Matrix)
    matrix_mapping = {
        ("A", "A"): "A",
        ("A", "B"): "A",
        ("B", "A"): "A",
        ("B", "B"): "B",
        ("A", "C"): "B",
        ("C", "A"): "B",
        ("A", "D"): "B",
        ("D", "A"): "B",
        ("B", "C"): "C",
        ("C", "B"): "C",
        ("B", "D"): "C",
        ("D", "B"): "C",
        ("C", "C"): "C",
        ("C", "D"): "D",
        ("D", "C"): "D",
        ("D", "D"): "D",
    }

    df_calc["Joint Class"] = df_calc["Sales Class"] + "-" + df_calc["Hits Class"]
    df_calc["Classification"] = df_calc.apply(
        lambda r: matrix_mapping.get((r["Sales Class"], r["Hits Class"]), "D"),
        axis=1,
    )

    # Cross-tabulation Matrix (Sales Class x Hits Class)
    crosstab_df = pd.crosstab(
        df_calc["Sales Class"],
        df_calc["Hits Class"],
        rownames=["Sales Class"],
        colnames=["Hits Class"],
    ).reindex(index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"], fill_value=0)

    # Class summary table
    strategies = {
        "A": "Top Priority: Tight daily review, weekly cycle counts, prime warehouse dock slotting",
        "B": "Moderate Priority: Standard periodic review, monthly cycle counts, moderate safety stock",
        "C": "Low Priority: Automated reorder point/min-max control, high order batches, low physical priority",
        "D": "Minimal/Tail: Dead stock / slow mover review, zero buffer, evaluate for clearance or phase-out",
    }

    summary_rows = []
    n_items = len(df_calc)
    for c in ["A", "B", "C", "D"]:
        sub = df_calc[df_calc["Classification"] == c]
        cnt = len(sub)
        s_sub = float(sub[sales_col].sum())
        h_sub = int(sub[hits_col].sum())
        summary_rows.append({
            "Class": c,
            "SKU Count": cnt,
            "% of SKUs": f"{(cnt / n_items * 100):.1f}%" if n_items > 0 else "0.0%",
            "Total Sales ($)": f"${s_sub:,.2f}",
            "% of Sales": f"{(s_sub / total_sales * 100):.1f}%" if total_sales > 0 else "0.0%",
            "Total Hits": f"{h_sub:,}",
            "% of Hits": f"{(h_sub / total_hits * 100):.1f}%" if total_hits > 0 else "0.0%",
            "Operational Strategy": strategies.get(c, ""),
        })
    class_summary_df = pd.DataFrame(summary_rows)

    return {
        "valid": True,
        "rows": n_items,
        "columns": len(cols),
        "column_names": cols,
        "sales_col": sales_col,
        "hits_col": hits_col,
        "item_num_col": item_num_col,
        "item_name_col": item_name_col,
        "uom_col": uom_col,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "class_summary_df": class_summary_df,
        "crosstab_df": crosstab_df,
        "classified_df": df_calc,
    }


def run_streamlit_app():
    """
    Renders the Streamlit Web Application when launched via `streamlit run app.py`.
    """
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        """
        **Architecture**: `Customer Question` → `Identify Model` → `Extract Inputs` → `Select Python Tool` → `Calculate` → `Verify` → `Explain`  
        *All numerical calculations are executed deterministically using Python.*
        """
    )

    agent = InventoryDecisionAgent()

    # Sidebar: Model Guide & Presets
    st.sidebar.header("Navigation & Scenarios")
    preset_choice = st.sidebar.selectbox(
        "Load Benchmark Question:",
        ["Custom Question"] + [s["title"] for s in BENCHMARK_SCENARIOS],
    )

    # Sidebar: Data Upload Section (Last section on the left sidebar)
    st.sidebar.markdown("---")
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload an inventory CSV file with Product, Category, Units, and Unit Cost columns.",
    )

    default_text = ""
    if preset_choice != "Custom Question":
        for s in BENCHMARK_SCENARIOS:
            if s["title"] == preset_choice:
                default_text = s["question"]
                break

    question_text = st.text_area(
        "Enter your natural-language inventory question:",
        value=default_text,
        height=140,
        placeholder="e.g. Our annual demand is 10,000 units, order cost is $50, and holding cost is $2. What is our EOQ?",
    )

    if st.button("Run Inventory Decision Pipeline", type="primary"):
        if not question_text.strip():
            st.warning("Please enter an inventory question.")
            return

        with st.spinner("Processing through 7-stage Supply Chain Decision Pipeline..."):
            response = agent.process_question(question_text)

        # Trace stages visually
        col1, col2, col3 = st.columns(3)
        col1.metric("Identified Model", response.model_type.value)
        col2.metric("Pipeline Status", response.verification_status.get("status", "UNKNOWN"))
        if response.calculation_result and hasattr(response.calculation_result, "to_dict"):
            d = response.calculation_result.to_dict()
            first_key = list(d.keys())[0]
            col3.metric(first_key.replace("_", " ").title(), d[first_key])

        # Verification pills
        with st.expander("Pipeline Audit Trace & Verification Checks", expanded=False):
            st.json({
                "model_type": response.model_type.value,
                "extracted_inputs": response.extracted_inputs,
                "verification": response.verification_status,
                "calculation_metrics": (
                    response.calculation_result.to_dict()
                    if response.calculation_result and hasattr(response.calculation_result, "to_dict")
                    else str(response.calculation_result)
                ),
            })

        # Main explanation output
        st.markdown(response.explanation)

    # -------------------------------------------------------------------------
    # Main Page: CSV Data Processing & Analysis
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        import pandas as pd

        st.markdown("---")

        try:
            # 1. Read the CSV file into a pandas DataFrame using single existing uploader
            df = pd.read_csv(uploaded_file)
            cols = list(df.columns)

            is_classification = any(c in cols for c in ["Sales ($)", "Sales", "Total Sales", "Annual Sales", "Revenue"]) and any(c in cols for c in ["Hits", "Picks"])
            is_inventory_value = "Units" in cols and "Unit Cost" in cols

            # -----------------------------------------------------------------
            # Capability 1: Multicriteria Inventory Classification (A, B, C, D)
            # -----------------------------------------------------------------
            if is_classification:
                st.header("🏷️ Multicriteria Inventory Classification (A, B, C, D Classes)")
                st.markdown(
                    "*Evaluates inventory items across both **Financial Velocity (Sales)** and **Operational Activity (Hits)** into **A, B, C, and D** classes.*"
                )

                analysis_cls = classify_multicriteria_inventory(df)

                if not analysis_cls["valid"]:
                    st.error(
                        f"❌ **Validation Error**: The uploaded classification file is missing required column(s): "
                        f"**{', '.join(analysis_cls['missing_columns'])}**.\n\n"
                        f"Required columns for classification: `Item Number, Item Name, UoM, Sales ($), Hits`.\n\n"
                        f"Columns detected: `{', '.join(analysis_cls['found_columns'])}`."
                    )
                else:
                    # 1. Classification Overview KPIs
                    st.subheader("1. Classification Summary Metrics")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Items", f"{analysis_cls['rows']:,}")
                    k2.metric("Total Sales", f"${analysis_cls['total_sales']:,.2f}")
                    k3.metric("Total Warehouse Hits", f"{analysis_cls['total_hits']:,}")
                    a_count = int(analysis_cls["class_summary_df"][analysis_cls["class_summary_df"]["Class"] == "A"]["SKU Count"].iloc[0])
                    k4.metric("Class A Items (Vital)", f"{a_count} ({a_count / analysis_cls['rows'] * 100:.1f}%)")

                    # 2. Class Summary Breakdown Table
                    st.subheader("2. Inventory Class Breakdown (A, B, C, D)")
                    st.dataframe(analysis_cls["class_summary_df"], use_container_width=True)

                    # 3. 2D Joint Criteria Matrix (Sales Class x Hits Class)
                    st.subheader("3. Joint Criteria Cross-Tabulation Matrix (Sales × Hits)")
                    st.caption("Distribution of SKUs mapped by Sales Class (rows) and Hits Class (columns):")
                    st.dataframe(analysis_cls["crosstab_df"], use_container_width=True)

                    # 4. Interactive Dataframe with Classification Results
                    st.subheader("4. Class-Annotated Inventory Dataframe")
                    col_f1, col_f2 = st.columns([1, 2])
                    filter_class = col_f1.selectbox(
                        "Filter by Class:",
                        ["All Classes (A, B, C, D)", "Class A", "Class B", "Class C", "Class D"],
                    )
                    search_query = col_f2.text_input("Search Item Number or Name:", "")

                    filtered_df = analysis_cls["classified_df"].copy()
                    if filter_class != "All Classes (A, B, C, D)":
                        c_target = filter_class.split()[-1]
                        filtered_df = filtered_df[filtered_df["Classification"] == c_target]

                    if search_query.strip():
                        q_lower = search_query.strip().lower()
                        s_mask = (
                            filtered_df[analysis_cls["item_num_col"]].astype(str).str.lower().str.contains(q_lower) |
                            filtered_df[analysis_cls["item_name_col"]].astype(str).str.lower().str.contains(q_lower)
                        )
                        filtered_df = filtered_df[s_mask]

                    s_col = analysis_cls["sales_col"]
                    h_col = analysis_cls["hits_col"]

                    st.dataframe(
                        filtered_df.style.format({
                            s_col: "${:,.2f}",
                            h_col: "{:,.0f}",
                        }),
                        use_container_width=True,
                    )

            # -----------------------------------------------------------------
            # Capability 2: Inventory Value & Category Analysis
            # -----------------------------------------------------------------
            if is_inventory_value:
                st.header("📊 Inventory Value & Category Analysis")
                analysis_val = analyze_inventory_dataframe(df)

                if not analysis_val["valid"]:
                    st.error(
                        f"❌ **Validation Error**: The uploaded CSV file is missing required column(s): "
                        f"**{', '.join(analysis_val['missing_columns'])}**.\n\n"
                        f"Required columns: `Product, Category, Units, Unit Cost`.\n\n"
                        f"Columns detected: `{', '.join(analysis_val['found_columns'])}`."
                    )
                else:
                    st.subheader("1. Dataset Overview & Dimensions")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Number of Rows", f"{analysis_val['rows']:,}")
                    m2.metric("Number of Columns", f"{analysis_val['columns']:,}")
                    m3.write(f"**Column Names:**\n`{', '.join(analysis_val['column_names'])}`")

                    st.caption("Uploaded Raw Dataframe Preview:")
                    st.dataframe(df, use_container_width=True)

                    st.subheader("2. Summary Metrics")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Units", f"{analysis_val['total_units']:,.0f}")
                    kpi2.metric("Average Unit Cost", f"${analysis_val['avg_unit_cost']:,.2f}")
                    kpi3.metric("Highest Units Product", analysis_val["product_highest_units"], f"{analysis_val['highest_units_val']:,.0f} units")
                    kpi4.metric("Total Inventory Value", f"${analysis_val['total_inventory_value']:,.2f}")

                    st.subheader("3. Total Units by Category")
                    st.dataframe(analysis_val["category_units_df"], use_container_width=True)

                    st.subheader("4. Inventory Data with Calculated Inventory Value")
                    st.dataframe(
                        analysis_val["analyzed_df"].style.format({
                            "Units": "{:,.0f}",
                            "Unit Cost": "${:,.2f}",
                            "Inventory Value": "${:,.2f}",
                        }),
                        use_container_width=True,
                    )

            # -----------------------------------------------------------------
            # Fallback if neither schema matches
            # -----------------------------------------------------------------
            if not is_classification and not is_inventory_value:
                st.error(
                    f"❌ **Unrecognized CSV Schema**: Found columns: `{', '.join(cols)}`.\n\n"
                    f"Supported schemas:\n"
                    f"- **Multicriteria Inventory Classification**: `Item Number, Item Name, UoM, Sales ($), Hits`\n"
                    f"- **Inventory Value & Category Analysis**: `Product, Category, Units, Unit Cost`"
                )

        except Exception as err:
            st.error(f"❌ Error processing uploaded CSV file: {err}")



HTML_DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Supply Chain Analytics Inventory Decision Agent</title>
  <style>
    :root {
      --primary: #1e3a8a;
      --primary-light: #3b82f6;
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
    h1 { font-size: 26px; font-weight: 700; display: flex; align-items: center; gap: 10px; color: #fff; }
    p.subtitle { color: var(--text-muted); font-size: 14px; margin-top: 6px; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 20px; }
    .scenario-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
    .btn-scenario { background: #334155; color: var(--text); border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; transition: all 0.2s; }
    .btn-scenario:hover { background: var(--primary-light); color: white; }
    textarea { width: 100%; height: 110px; background: #0f172a; color: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 14px; resize: vertical; margin-bottom: 14px; }
    textarea:focus { outline: none; border-color: var(--primary-light); }
    .btn-run { background: #2563eb; color: white; border: none; padding: 10px 24px; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; }
    .btn-run:hover { background: #1d4ed8; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .kpi-card { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
    .kpi-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 20px; font-weight: 700; margin-top: 6px; }
    .status-badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
    .badge-passed { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-incomplete { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .badge-error { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    .report-box { line-height: 1.6; font-size: 14px; }
    .report-box h3 { font-size: 18px; margin: 16px 0 8px 0; color: #60a5fa; border-bottom: 1px solid #334155; padding-bottom: 4px; }
    .report-box h4 { font-size: 15px; margin: 14px 0 6px 0; color: #93c5fd; }
    .report-box ul { padding-left: 20px; margin-bottom: 12px; }
    .report-box li { margin-bottom: 4px; }
    .report-box table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
    .report-box th, .report-box td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; }
    .report-box th { background: #0f172a; color: #93c5fd; }
    .traces-list { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
    .trace-item { display: flex; align-items: center; justify-content: space-between; background: #0f172a; padding: 8px 12px; border-radius: 6px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>📦 Supply Chain Analytics Inventory Decision Agent</h1>
      <p class="subtitle">Autonomous 7-stage operational reasoning pipeline powered deterministically by Python OR algorithms.</p>
    </header>

    <div class="card">
      <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-muted);">SELECT TEXTBOOK BENCHMARK SCENARIO:</div>
      <div class="scenario-buttons" id="scenarioButtons"></div>
      <textarea id="questionInput" placeholder="Enter your natural-language supply chain question here..."></textarea>
      <button class="btn-run" id="btnRun" onclick="runPipeline()">🚀 Run Inventory Decision Pipeline</button>
    </div>

    <div id="resultsArea" style="display: none;">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Identified Model</div>
          <div class="kpi-value" id="kpiModel">-</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Verification Status</div>
          <div class="kpi-value" id="kpiStatus">-</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Core Recommendation</div>
          <div class="kpi-value" id="kpiMetric" style="color: #60a5fa;">-</div>
        </div>
      </div>

      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="toggleTraces()">
          <h3 style="font-size: 15px; color: var(--text);">🔍 7-Stage Execution Audit Trace</h3>
          <span id="traceToggle" style="font-size: 12px; color: var(--text-muted);">[Toggle Details]</span>
        </div>
        <div id="tracesContainer" class="traces-list" style="display: none; margin-top: 12px;"></div>
      </div>

      <div class="card">
        <div class="report-box" id="reportContent"></div>
      </div>
    </div>
  </div>

  <script>
    const scenarios = [
      {
        title: "1. EOQ",
        q: "Our manufacturing facility has an annual demand of 12,000 units. Each order incurs a setup cost of $50, and our annual holding cost is $2 per unit per year. Each unit costs $10. What is our optimal order quantity and how often should we order?"
      },
      {
        title: "2. Quantity Discount",
        q: "Annual demand is 10,000 units and ordering cost is $100 per order. Carrying cost is 20% of item cost. The supplier offers the following price brackets: 1 to 499 units at $5.00, 500 to 999 units at $4.50, and 1000 and above at $4.00. Which quantity discount bracket should we choose to minimize total costs?"
      },
      {
        title: "3. Safety Stock & ROP",
        q: "Daily demand averages 100 units with a standard deviation of 20 units. Supplier lead time is 4 days. Management targets a 95% cycle service level. What should our safety stock and reorder point be?"
      },
      {
        title: "4. Fixed-Period Review",
        q: "Our pharmacy operates a periodic review policy with inventory reviewed every 7 days and a lead time of 3 days. Daily demand has a mean of 50 units and standard deviation of 10 units. Desired cycle service level is 95%. Currently, physical on-hand inventory is 120 units with 0 on order. How many units should we order today?"
      },
      {
        title: "5. Newsvendor",
        q: "We are procuring a seasonal winter jacket for a single-period retail season. The retail selling price is $120, unit wholesale cost is $50, and unsold jackets have a salvage value of $20. Customer demand is normally distributed with a mean of 500 units and standard deviation of 80 units. What is the optimal order quantity, critical fractile, and expected profit?"
      }
    ];

    const btnContainer = document.getElementById("scenarioButtons");
    scenarios.forEach((s) => {
      const b = document.createElement("button");
      b.className = "btn-scenario";
      b.innerText = s.title;
      b.onclick = () => { document.getElementById("questionInput").value = s.q; };
      btnContainer.appendChild(b);
    });
    document.getElementById("questionInput").value = scenarios[0].q;

    function toggleTraces() {
      const el = document.getElementById("tracesContainer");
      el.style.display = el.style.display === "none" ? "flex" : "none";
    }

    function renderMarkdown(md) {
      return md
        .replace(/### (.*)/g, "<h3>$1</h3>")
        .replace(/#### (.*)/g, "<h4>$1</h4>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\$\$(.*?)\$\$/gs, "<div style='background:#0f172a; padding:6px 10px; border-radius:4px; font-family:monospace; margin:8px 0; color:#93c5fd;'>$1</div>")
        .replace(/\$(.*?)\$/g, "<code style='background:#0f172a; padding:2px 4px; border-radius:3px; color:#93c5fd;'>$1</code>")
        .replace(/^- (.*)/gm, "<li>$1</li>")
        .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
        .replace(/\n/g, "<br>");
    }

    async function runPipeline() {
      const q = document.getElementById("questionInput").value.trim();
      if (!q) return;
      const btn = document.getElementById("btnRun");
      btn.innerText = "⏳ Executing 7-Stage Pipeline...";
      btn.disabled = true;

      try {
        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q })
        });
        const data = await res.json();

        document.getElementById("resultsArea").style.display = "block";
        document.getElementById("kpiModel").innerText = data.model_type || "Unknown";

        const st = data.verification ? data.verification.status : "DONE";
        const stEl = document.getElementById("kpiStatus");
        stEl.innerHTML = `<span class="status-badge ${st === 'PASSED' ? 'badge-passed' : (st === 'INCOMPLETE' ? 'badge-incomplete' : 'badge-error')}">${st}</span>`;

        let rec = "Analyzed";
        if (data.calculation_result) {
          const cr = data.calculation_result;
          if (cr.final_answer) {
            const keys = Object.keys(cr.final_answer);
            if (keys.length > 0) rec = `${keys[0].replace(/_/g, " ")}: ${cr.final_answer[keys[0]]}`;
          } else if (cr.eoq) {
            rec = `EOQ: ${Math.round(cr.eoq)} units`;
          } else if (cr.optimal_order_quantity) {
            rec = `Optimal Q: ${Math.round(cr.optimal_order_quantity)} units`;
          } else if (cr.reorder_point) {
            rec = `ROP: ${Math.round(cr.reorder_point)} units`;
          }
        }
        document.getElementById("kpiMetric").innerText = rec;

        const traceBox = document.getElementById("tracesContainer");
        traceBox.innerHTML = "";
        (data.traces || []).forEach(t => {
          const item = document.createElement("div");
          item.className = "trace-item";
          item.innerHTML = `<span><strong>${t.stage_name}</strong></span><span class="status-badge ${t.status === 'SUCCESS' || t.status === 'PASSED' ? 'badge-passed' : 'badge-incomplete'}">${t.status}</span>`;
          traceBox.appendChild(item);
        });

        document.getElementById("reportContent").innerHTML = renderMarkdown(data.explanation);
        document.getElementById("resultsArea").scrollIntoView({ behavior: 'smooth' });
      } catch (err) {
        alert("Error executing query: " + err);
      } finally {
        btn.innerText = "🚀 Run Inventory Decision Pipeline";
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


def run_local_web_dashboard(port: int = 8501):
    """
    Launches a standalone, zero-dependency interactive web dashboard using Python's
    built-in HTTP server. Automatically opens in your web browser.
    """
    import http.server
    import json
    import webbrowser

    agent = InventoryDecisionAgent()

    def serialize_obj(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, list):
            return [serialize_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        if hasattr(obj, "__dict__"):
            return {k: serialize_obj(v) for k, v in obj.__dict__.items()}
        return obj

    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ["/", "/index.html"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/query":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body)
                    question = payload.get("question", "")
                    resp = agent.process_question(question)
                    out = resp.to_dict()
                    out["traces"] = [
                        {"stage_name": t.stage_name, "status": t.status, "data": serialize_obj(t.data)}
                        for t in resp.traces
                    ]
                    out["extracted_inputs"] = serialize_obj(out.get("extracted_inputs", {}))
                    res_bytes = json.dumps(out).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress normal verbose HTTP logging
            pass

    server_address = ("", port)
    try:
        httpd = http.server.HTTPServer(server_address, DashboardHandler)
    except OSError:
        port = port + 1
        httpd = http.server.HTTPServer(("", port), DashboardHandler)

    url = f"http://localhost:{port}"
    print(f"\n{'='*70}")
    print(f" 📦 Supply Chain Inventory Decision Agent Web Dashboard")
    print(f" Web Server Live at: {url}")
    print(f" Press Ctrl+C to stop the server.")
    print(f"{'='*70}\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        httpd.server_close()


if __name__ == "__main__":
    # Check if running under Streamlit
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_streamlit_app()
            sys.exit(0)
    except Exception:
        pass

    # Command dispatcher
    agent = InventoryDecisionAgent()
    if len(sys.argv) > 1 and sys.argv[1] in ["--web", "-w", "web"]:
        run_local_web_dashboard(port=8501)
    elif len(sys.argv) > 1 and sys.argv[1] in ["--demo", "-d", "demo"]:
        run_benchmark_suite(agent)
    else:
        run_interactive_cli(agent)

