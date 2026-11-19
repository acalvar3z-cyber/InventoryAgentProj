''"""
inventory_tools.py
------------------
Deterministic Supply Chain Inventory Analytics Calculation Engine.
Implements the 5 core inventory management decision tools:
  1. calculate_eoq()
  2. calculate_quantity_discount()
  3. calculate_rop()
  4. calculate_periodic_review()
  5. calculate_newsvendor()

Every calculation tool deterministically computes and returns all 8 student verification sections:
  1. Inputs Extracted
  2. Variable Definitions
  3. Formula
  4. Formula with Substituted Values
  5. Intermediate Calculations
  6. Final Answer
  7. Verification Checks
  8. Assumptions

Numerical calculation steps come strictly from the deterministic Python tool.
Never generates or reconstructs numerical calculations independently in language models.
Never invents missing numerical inputs.
"""

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================================
# 1. Economic Order Quantity (EOQ)
# ============================================================================

@dataclass
class EOQResult:
    # 8 Student Verification Components
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, Any]]
    assumptions: List[str]

    # Typed Attributes & Aliases for direct property access
    annual_demand: float
    order_cost: float
    holding_cost_unit: float
    unit_cost: float
    working_days_per_year: int
    eoq: float
    recommended_order_quantity: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_purchase_cost: float
    total_annual_cost: float
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Economic Order Quantity (EOQ)",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "inputs": self.inputs_extracted,
        }


def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_unit: Optional[float] = None,
    holding_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    working_days_per_year: int = 365,
) -> EOQResult:
    """
    Computes Economic Order Quantity (EOQ) and total annual inventory costs.
    Formula:
        EOQ = sqrt(2 * D * S / H)
        where H = holding_cost_unit OR (holding_rate * unit_cost)
    Returns full 8-component student verification model.
    """
    # 1. Validate inputs (Never invent missing inputs)
    if annual_demand is None or not isinstance(annual_demand, (int, float)):
        raise ValueError("Annual demand (D) must be provided as a positive number.")
    if annual_demand <= 0:
        raise ValueError(f"Annual demand (D) must be strictly positive, got: {annual_demand}")

    if order_cost is None or not isinstance(order_cost, (int, float)):
        raise ValueError("Order cost (S) must be provided as a positive number.")
    if order_cost <= 0:
        raise ValueError(f"Order cost (S) must be strictly positive, got: {order_cost}")

    if working_days_per_year <= 0:
        raise ValueError(f"Working days per year must be positive, got: {working_days_per_year}")

    effective_unit_cost = float(unit_cost) if unit_cost is not None else 0.0
    if effective_unit_cost < 0:
        raise ValueError(f"Unit cost cannot be negative, got: {unit_cost}")

    if holding_cost_unit is not None:
        if not isinstance(holding_cost_unit, (int, float)) or holding_cost_unit <= 0:
            raise ValueError(f"holding_cost_unit must be a positive number, got: {holding_cost_unit}")
        h = float(holding_cost_unit)
    elif holding_rate is not None:
        if not isinstance(holding_rate, (int, float)) or holding_rate <= 0:
            raise ValueError(f"holding_rate must be a positive number, got: {holding_rate}")
        if unit_cost is None or effective_unit_cost <= 0:
            raise ValueError("When holding_rate is provided, a positive unit_cost must also be provided to calculate holding cost H = h * C.")
        rate = holding_rate if holding_rate <= 1.0 else (holding_rate / 100.0)
        h = rate * effective_unit_cost
    else:
        raise ValueError(
            "Missing holding cost input: must provide either holding_cost_unit (H) or holding_rate (h) along with unit_cost (C)."
        )

    D_val = float(annual_demand)
    S_val = float(order_cost)
    working_days = int(working_days_per_year)

    # 2. Deterministic Intermediate calculations
    eoq = math.sqrt((2.0 * D_val * S_val) / h)
    orders_per_year = D_val / eoq
    cycle_time_years = eoq / D_val
    cycle_time_days = cycle_time_years * working_days

    annual_ordering_cost = orders_per_year * S_val
    annual_holding_cost = (eoq / 2.0) * h
    annual_purchase_cost = D_val * effective_unit_cost
    total_annual_cost = annual_ordering_cost + annual_holding_cost + annual_purchase_cost
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": D_val,
        "order_cost": S_val,
        "holding_cost_unit": round(h, 4),
        "holding_rate": holding_rate,
        "unit_cost": effective_unit_cost if effective_unit_cost > 0 else None,
        "working_days_per_year": working_days,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand rate (units/year)",
        "S": "Fixed setup or ordering cost per replenishment order ($/order)",
        "H": "Annual inventory holding / carrying cost per unit ($/unit/year)",
        "C": "Unit purchase price / material cost ($/unit)",
        "h": "Annual percentage inventory carrying rate (decimal fraction of unit price/year)",
        "Q": "Order quantity per replenishment cycle (units)",
        "EOQ (Q*)": "Economic Order Quantity / cost-minimizing order batch size (units)",
        "AOC": "Annual Ordering Cost ($/year) = (D / Q) * S",
        "AHC": "Annual Holding Cost ($/year) = (Q / 2) * H",
        "APC": "Annual Purchase Cost ($/year) = D * C",
        "TC": "Total Annual Inventory Cost ($/year) = AOC + AHC + APC",
        "N": "Order frequency / replenishment orders placed per year = D / Q",
        "T": "Cycle time / time between consecutive replenishment orders (days)",
    }

    # 3. Formula
    formula = {
        "Holding Cost per Unit (H)": r"H = \text{holding\_cost\_unit} \quad \text{or} \quad H = h \times C",
        "Economic Order Quantity (EOQ)": r"Q^* = \sqrt{\frac{2 \times D \times S}{H}}",
        "Annual Ordering Cost (AOC)": r"\text{AOC} = \frac{D}{Q} \times S",
        "Annual Holding Cost (AHC)": r"\text{AHC} = \frac{Q}{2} \times H",
        "Annual Purchase Cost (APC)": r"\text{APC} = D \times C",
        "Total Annual Inventory Cost (TC)": r"\text{TC} = \text{AOC} + \text{AHC} + \text{APC} = \left(\frac{D}{Q} \times S\right) + \left(\frac{Q}{2} \times H\right) + (D \times C)",
        "Orders Per Year (N)": r"N = \frac{D}{Q}",
        "Cycle Time in Days (T)": r"T = \frac{Q}{D} \times \text{Working Days Per Year}",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Unit Holding Cost": rf"H = {h:.4f} \text{{ $/unit/year}}",
        "Economic Order Quantity": rf"Q^* = \sqrt{{\frac{{2 \times {D_val:,.0f} \times {S_val:.2f}}}{{{h:.4f}}}}} = \sqrt{{{2.0 * D_val * S_val / h:,.2f}}} = {eoq:,.2f} \text{{ units}}",
        "Annual Ordering Cost": rf"\text{{AOC}} = \frac{{{D_val:,.0f}}}{{{eoq:.2f}}} \times \${S_val:.2f} = {orders_per_year:.2f} \times \${S_val:.2f} = \${annual_ordering_cost:,.2f}",
        "Annual Holding Cost": rf"\text{{AHC}} = \frac{{{eoq:.2f}}}{{2}} \times \${h:.4f} = {eoq/2.0:.2f} \times \${h:.4f} = \${annual_holding_cost:,.2f}",
        "Annual Purchase Cost": rf"\text{{APC}} = {D_val:,.0f} \times \${effective_unit_cost:.2f} = \${annual_purchase_cost:,.2f}",
        "Total Annual Inventory Cost": rf"\text{{TC}} = \${annual_ordering_cost:,.2f} + \${annual_holding_cost:,.2f} + \${annual_purchase_cost:,.2f} = \${total_annual_cost:,.2f}",
        "Orders Per Year": rf"N = \frac{{{D_val:,.0f}}}{{{eoq:.2f}}} = {orders_per_year:.2f} \text{{ orders/year}}",
        "Cycle Time in Days": rf"T = \frac{{{eoq:.2f}}}{{{D_val:,.0f}}} \times {working_days} = {cycle_time_days:.1f} \text{{ days}}",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "holding_cost_per_unit_per_year": round(h, 4),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "annual_purchase_cost": round(annual_purchase_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2),
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_years": round(cycle_time_years, 4),
        "cycle_time_days": round(cycle_time_days, 1),
        "cost_difference_AOC_vs_AHC": round(cost_diff, 4),
    }

    # 6. Final Answer
    final_ans = {
        "recommended_order_quantity": round(eoq, 2),
        "optimal_order_quantity": round(eoq, 2),
        "optimal_order_quantity_integer": round(eoq),
        "total_annual_cost": round(total_annual_cost, 2),
        "order_frequency_orders_per_year": round(orders_per_year, 2),
        "cycle_time_days": round(cycle_time_days, 1),
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Positive Order Quantity",
            "condition": "Q* > 0",
            "value": round(eoq, 2),
            "passed": bool(eoq > 0),
            "explanation": "Optimal order quantity must be strictly positive.",
        },
        {
            "check": "Cost Trade-off Balance Invariant (AOC ≈ AHC)",
            "condition": "|AOC - AHC| < $0.05",
            "diff": round(cost_diff, 4),
            "passed": bool(cost_diff < 0.05),
            "explanation": "At theoretical EOQ optimality, annual ordering cost matches annual holding cost.",
        },
        {
            "check": "Positive Order Frequency",
            "condition": "N > 0",
            "value": round(orders_per_year, 2),
            "passed": bool(orders_per_year > 0),
            "explanation": "Replenishment frequency must be strictly positive.",
        },
        {
            "check": "Positive Cycle Time",
            "condition": "T > 0",
            "value": round(cycle_time_days, 1),
            "passed": bool(cycle_time_days > 0),
            "explanation": "Operating cycle duration between deliveries must be positive.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Demand rate D is continuous, constant, and known with certainty throughout the planning horizon.",
        "Fixed ordering/setup cost S is invariant to replenishment lot size Q.",
        "Unit purchase price C is constant with no quantity discounts applicable.",
        "Replenishment lead time is zero or constant with complete, instantaneous batch delivery.",
        "Holding cost H is linear and directly proportional to average cycle inventory level (EOQ / 2).",
        "No shortages, stockouts, or customer backorders are permitted.",
        "Infinite planning horizon with no warehouse storage or capital budget capacity limits.",
    ]

    return EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        verification_checks=verification_checks,
        assumptions=assumptions,
        annual_demand=D_val,
        order_cost=S_val,
        holding_cost_unit=h,
        unit_cost=effective_unit_cost,
        working_days_per_year=working_days,
        eoq=eoq,
        recommended_order_quantity=eoq,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        orders_per_year=orders_per_year,
        cycle_time_years=cycle_time_years,
        cycle_time_days=cycle_time_days,
    )


# ============================================================================
# 2. Quantity Discount Model
# ============================================================================

@dataclass
class DiscountTier:
    tier_name: str
    min_qty: float
    max_qty: float  # float('inf') for open-ended upper bracket
    unit_price: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "min_qty": self.min_qty,
            "max_qty": self.max_qty if self.max_qty != float("inf") else "inf",
            "unit_price": round(self.unit_price, 2),
        }


@dataclass
class TierEvaluation:
    tier_name: str
    min_qty: float
    max_qty: float
    unit_price: float
    holding_cost_unit: float
    unconstrained_eoq: float
    feasible_order_qty: float
    is_price_break: bool
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_purchase_cost: float
    total_annual_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "quantity_range": f"[{self.min_qty}, {self.max_qty if self.max_qty != float('inf') else 'inf'}]",
            "unit_price": round(self.unit_price, 2),
            "holding_cost_unit": round(self.holding_cost_unit, 2),
            "unconstrained_eoq": round(self.unconstrained_eoq, 2),
            "feasible_order_qty": round(self.feasible_order_qty, 2),
            "is_price_break_boundary": self.is_price_break,
            "annual_ordering_cost": round(self.annual_ordering_cost, 2),
            "annual_holding_cost": round(self.annual_holding_cost, 2),
            "annual_purchase_cost": round(self.annual_purchase_cost, 2),
            "total_annual_cost": round(self.total_annual_cost, 2),
        }


@dataclass
class QuantityDiscountResult:
    # 8 Student Verification Components
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, Any]]
    assumptions: List[str]

    # Typed Attributes & Aliases
    optimal_tier_name: str
    optimal_order_quantity: float
    optimal_unit_price: float
    lowest_total_cost: float
    baseline_eoq: float
    baseline_total_cost: float
    annual_savings: float
    percent_savings: float
    tier_evaluations: List[TierEvaluation]
    recommended_order_quantity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Quantity Discount Model (All-Units)",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "optimal_tier": self.optimal_tier_name,
            "optimal_order_quantity": self.optimal_order_quantity,
            "recommended_order_quantity": self.optimal_order_quantity,
            "lowest_total_cost": self.lowest_total_cost,
        }


def calculate_quantity_discount(
    annual_demand: float,
    order_cost: float,
    tiers: List[Union[DiscountTier, Dict[str, Any]]],
    holding_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
    unit_purchase_price: Optional[float] = None,
    holding_cost_unit: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Computes all-units quantity discount optimal replenishment policy.
    Returns full 8-component student verification model.
    """
    if fixed_holding_cost is None and holding_cost_unit is not None:
        fixed_holding_cost = holding_cost_unit
    if annual_demand is None or not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"Annual demand (D) must be positive, got: {annual_demand}")
    if order_cost is None or not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"Order cost (S) must be positive, got: {order_cost}")
    if not tiers or len(tiers) == 0:
        raise ValueError("At least one discount tier must be provided.")

    if fixed_holding_cost is not None and fixed_holding_cost > 0:
        h_mode = "fixed"
        h_rate_val = None
    elif holding_rate is not None and holding_rate > 0:
        h_mode = "rate"
        h_rate_val = holding_rate if holding_rate <= 1.0 else (holding_rate / 100.0)
    else:
        raise ValueError("Must provide either a positive holding_rate (percentage) or fixed_holding_cost ($/unit/year).")

    parsed_tiers: List[DiscountTier] = []
    for item in tiers:
        if isinstance(item, DiscountTier):
            parsed_tiers.append(item)
        elif isinstance(item, dict):
            parsed_tiers.append(
                DiscountTier(
                    tier_name=str(item.get("tier_name", f"Tier {len(parsed_tiers)+1}")),
                    min_qty=float(item["min_qty"]),
                    max_qty=float(item.get("max_qty", float("inf"))),
                    unit_price=float(item["unit_price"]),
                )
            )

    parsed_tiers.sort(key=lambda t: t.min_qty)
    D_val = float(annual_demand)
    S_val = float(order_cost)

    evaluations: List[TierEvaluation] = []
    sub_vals: Dict[str, str] = {}

    for t in parsed_tiers:
        H_i = float(fixed_holding_cost) if h_mode == "fixed" else (h_rate_val * t.unit_price)
        unconstrained_q = math.sqrt((2.0 * D_val * S_val) / H_i)

        if unconstrained_q < t.min_qty:
            feasible_q = t.min_qty
            is_break = True
            feas_note = f"Unconstrained Q ({unconstrained_q:.1f}) < min ({t.min_qty:.0f}) -> Adjusted to price break boundary: {feasible_q:.0f}"
        elif unconstrained_q > t.max_qty:
            feasible_q = t.max_qty
            is_break = False
            feas_note = f"Unconstrained Q ({unconstrained_q:.1f}) > max ({t.max_qty:.0f}) -> Infeasible for this tier"
        else:
            feasible_q = unconstrained_q
            is_break = False
            feas_note = f"Unconstrained Q ({unconstrained_q:.1f}) falls within range [{t.min_qty:.0f}, {t.max_qty:.0f}] -> Feasible"

        aoc = (D_val / feasible_q) * S_val
        ahc = (feasible_q / 2.0) * H_i
        apc = D_val * t.unit_price
        tc = aoc + ahc + apc

        evaluations.append(
            TierEvaluation(
                tier_name=t.tier_name,
                min_qty=t.min_qty,
                max_qty=t.max_qty,
                unit_price=t.unit_price,
                holding_cost_unit=H_i,
                unconstrained_eoq=unconstrained_q,
                feasible_order_qty=feasible_q,
                is_price_break=is_break,
                annual_ordering_cost=aoc,
                annual_holding_cost=ahc,
                annual_purchase_cost=apc,
                total_annual_cost=tc,
            )
        )

        sub_vals[f"{t.tier_name} Evaluation"] = (
            rf"Q^*_{{{t.tier_name}}} = \sqrt{{\frac{{2 \times {D_val:,.0f} \times {S_val:.2f}}}{{{H_i:.2f}}}}} = {unconstrained_q:.2f} \rightarrow "
            rf"Q^{{feas}} = {feasible_q:.2f} \quad [{feas_note}] \quad "
            rf"\text{{TC}} = \${aoc:,.2f} (\text{{AOC}}) + \${ahc:,.2f} (\text{{AHC}}) + \${apc:,.2f} (\text{{APC}}) = \mathbf{{\${tc:,.2f}}}"
        )

    best_eval = min(evaluations, key=lambda e: e.total_annual_cost)
    baseline_eval = evaluations[0]
    annual_savings = max(0.0, baseline_eval.total_annual_cost - best_eval.total_annual_cost)
    pct_savings = (annual_savings / baseline_eval.total_annual_cost) * 100.0 if baseline_eval.total_annual_cost > 0 else 0.0

    sub_vals["Optimal Selection Comparison"] = (
        rf"\text{{Optimal Tier}} = \mathbf{{{best_eval.tier_name}}} \ (Q = {best_eval.feasible_order_qty:,.2f} \text{{ units at }} \${best_eval.unit_price:.2f}/\text{{unit}}), "
        rf"\text{{Total Annual Cost}} = \mathbf{{\${best_eval.total_annual_cost:,.2f}}}, "
        rf"\text{{Annual Savings vs Baseline}} = \mathbf{{\${annual_savings:,.2f}}} \ ({pct_savings:.2f}\%)"
    )

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": D_val,
        "order_cost": S_val,
        "holding_rate": h_rate_val,
        "fixed_holding_cost": fixed_holding_cost,
        "tiers": [t.to_dict() for t in parsed_tiers],
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand (units/year)",
        "S": "Fixed setup or ordering cost per replenishment order ($/order)",
        "C_i": "Unit purchase price for discount tier i ($/unit)",
        "H_i": "Unit holding cost per year for tier i = h * C_i or fixed H ($/unit/year)",
        "h": "Annual inventory carrying rate (fraction of unit price/year)",
        "Q_i^*": "Unconstrained economic order quantity evaluated at unit price C_i",
        "Q_i^{feas}": "Feasible order quantity in tier i (adjusted to lower break threshold if Q_i^* < min_i)",
        "AOC_i": "Annual Ordering Cost in tier i = (D / Q_i^{feas}) * S",
        "AHC_i": "Annual Holding Cost in tier i = (Q_i^{feas} / 2) * H_i",
        "APC_i": "Annual Material Purchase Cost in tier i = D * C_i",
        "TC_i": "Total Annual Cost in tier i = AOC_i + AHC_i + APC_i",
        "Baseline EOQ": "Feasible order quantity under baseline (Tier 1) price",
        "Savings": "Net annual cost savings achieved by choosing optimal discount tier vs baseline",
    }

    # 3. Formula
    formula = {
        "Unconstrained EOQ": r"Q_i^* = \sqrt{\frac{2 \times D \times S}{H_i}}",
        "Feasible Order Quantity": r"Q_i^{feas} = \begin{cases} Q_i^* & \text{if } \text{min}_i \le Q_i^* \le \text{max}_i \\ \text{min}_i & \text{if } Q_i^* < \text{min}_i \\ \text{Infeasible} & \text{if } Q_i^* > \text{max}_i \end{cases}",
        "Total Annual Inventory Cost": r"\text{TC}(Q_i^{feas}) = \left(\frac{D}{Q_i^{feas}} \times S\right) + \left(\frac{Q_i^{feas}}{2} \times H_i\right) + (D \times C_i)",
        "Optimal Tier Decision": r"\text{Optimal Tier } i^* = \arg\min_{i} \text{TC}(Q_i^{feas})",
        "Annual Savings": r"\text{Savings} = \text{TC}(\text{Baseline Tier}) - \text{TC}(Q_{i^*}^{feas})",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = sub_vals

    # 5. Intermediate Calculations
    intermediate = {
        "tier_evaluations": [e.to_dict() for e in evaluations],
        "baseline_tier_name": baseline_eval.tier_name,
        "baseline_feasible_qty": baseline_eval.feasible_order_qty,
        "baseline_total_cost": round(baseline_eval.total_annual_cost, 2),
        "lowest_total_cost": round(best_eval.total_annual_cost, 2),
        "annual_savings": round(annual_savings, 2),
        "percent_savings": round(pct_savings, 2),
    }

    # 6. Final Answer
    final_ans = {
        "optimal_tier_name": best_eval.tier_name,
        "optimal_order_quantity": best_eval.feasible_order_qty,
        "optimal_order_quantity_integer": round(best_eval.feasible_order_qty),
        "recommended_order_quantity": best_eval.feasible_order_qty,
        "optimal_unit_price": best_eval.unit_price,
        "minimum_total_annual_cost": round(best_eval.total_annual_cost, 2),
        "annual_savings_vs_baseline": round(annual_savings, 2),
        "percent_savings_vs_baseline": round(pct_savings, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Optimal Quantity in Feasible Tier Range",
            "condition": f"{best_eval.min_qty} <= Q <= {best_eval.max_qty}",
            "value": best_eval.feasible_order_qty,
            "passed": bool(best_eval.min_qty <= best_eval.feasible_order_qty <= best_eval.max_qty),
            "explanation": "Recommended order batch size must fall inside the qualifying price bracket.",
        },
        {
            "check": "Global Cost Minimality",
            "condition": "TC(Optimal) <= TC(Tier_i) for all i",
            "passed": bool(all(best_eval.total_annual_cost <= e.total_annual_cost + 1e-4 for e in evaluations)),
            "explanation": "Chosen tier must have the lowest total annual inventory and purchase cost among all feasible tiers.",
        },
        {
            "check": "Positive Total Cost",
            "condition": "Total Cost > 0",
            "value": round(best_eval.total_annual_cost, 2),
            "passed": bool(best_eval.total_annual_cost > 0),
            "explanation": "Annual cost must be strictly positive.",
        },
        {
            "check": "Savings Non-Negative vs Baseline",
            "condition": "Savings >= 0",
            "value": round(annual_savings, 2),
            "passed": bool(annual_savings >= 0),
            "explanation": "Selecting the discount tier must not increase total cost compared to the baseline EOQ policy.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount applies to entire replenishment order once minimum lot threshold is reached.",
        "Annual customer demand D is deterministic, constant, and continuous across the horizon.",
        "Fixed ordering cost S is constant across all tiers regardless of order size.",
        "Warehouse storage space, working capital, and material handling capacity accommodate the larger discount batch sizes.",
        "Replenishment is instantaneous with 100% conforming batch deliveries.",
    ]

    return QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_tier_name=best_eval.tier_name,
        optimal_order_quantity=best_eval.feasible_order_qty,
        recommended_order_quantity=best_eval.feasible_order_qty,
        optimal_unit_price=best_eval.unit_price,
        lowest_total_cost=best_eval.total_annual_cost,
        baseline_eoq=baseline_eval.feasible_order_qty,
        baseline_total_cost=baseline_eval.total_annual_cost,
        annual_savings=annual_savings,
        percent_savings=pct_savings,
        tier_evaluations=evaluations,
    )


# ============================================================================
# 3. Safety Stock and Reorder Point (calculate_rop)
# ============================================================================

@dataclass
class ROPResult:
    # 8 Student Verification Components
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, Any]]
    assumptions: List[str]

    # Typed Attributes & Aliases
    reorder_point: float
    safety_stock: float
    case_type: str
    demand_during_lead_time_mean: float
    sigma_LT: float
    z_score: float
    cycle_service_level: Optional[float]
    demand_mean: float
    lead_time_mean: float
    demand_std: float
    lead_time_std: float

    @property
    def demand_mean_per_period(self) -> float:
        return self.demand_mean

    @property
    def demand_std_per_period(self) -> float:
        return self.demand_std

    @property
    def demand_during_lead_time_std(self) -> float:
        return self.sigma_LT

    @property
    def is_variable_lead_time(self) -> bool:
        return self.lead_time_std > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Safety Stock & Reorder Point (Continuous Review)",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "scenario": self.case_type,
        }


SafetyStockROPResult = ROPResult


def calculate_rop(
    demand_mean: Optional[float] = None,
    lead_time_mean: Optional[float] = None,
    demand_std: float = 0.0,
    lead_time_std: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    safety_stock: Optional[float] = None,
    # Backward compatibility aliases:
    daily_demand_mean: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    lead_time_days_mean: Optional[float] = None,
    lead_time_days_std: Optional[float] = None,
) -> ROPResult:
    """
    Computes Safety Stock (SS) and Reorder Point (ROP) under 4 operational demand/lead-time cases.
    Returns full 8-component student verification model.
    """
    d_m = demand_mean if demand_mean is not None else daily_demand_mean
    lt_m = lead_time_mean if lead_time_mean is not None else lead_time_days_mean
    d_s = demand_std if demand_std != 0.0 or daily_demand_std is None else daily_demand_std
    lt_s = lead_time_std if lead_time_std != 0.0 or lead_time_days_std is None else lead_time_days_std

    if d_m is None or not isinstance(d_m, (int, float)) or d_m < 0:
        raise ValueError(f"demand_mean must be a non-negative number, got: {d_m}")
    if lt_m is None or not isinstance(lt_m, (int, float)) or lt_m < 0:
        raise ValueError(f"lead_time_mean must be a non-negative number, got: {lt_m}")
    if d_s is None or not isinstance(d_s, (int, float)) or d_s < 0:
        raise ValueError(f"demand_std cannot be negative, got: {d_s}")
    if lt_s is None or not isinstance(lt_s, (int, float)) or lt_s < 0:
        raise ValueError(f"lead_time_std cannot be negative, got: {lt_s}")

    d_val = float(d_m)
    L_val = float(lt_m)
    sig_d_val = float(d_s)
    sig_L_val = float(lt_s)

    # Classify into 4 textbook operational cases
    is_d_variable = sig_d_val > 0
    is_lt_variable = sig_L_val > 0

    if not is_d_variable and not is_lt_variable:
        case_type = "constant_demand_constant_lead_time"
        case_description = "Case 1: Constant Demand and Constant Lead Time"
        sigma_LT = 0.0
        formula_sigma_str = r"\sigma_{LT} = 0"
        sub_sigma_str = r"\sigma_{LT} = 0 \text{ units (deterministic)}"
    elif is_d_variable and not is_lt_variable:
        case_type = "variable_demand_constant_lead_time"
        case_description = "Case 2: Variable Demand and Constant Lead Time"
        sigma_LT = math.sqrt(L_val * (sig_d_val ** 2))
        formula_sigma_str = r"\sigma_{LT} = \sqrt{L \times \sigma_d^2} = \sigma_d \sqrt{L}"
        sub_sigma_str = rf"\sigma_{{LT}} = {sig_d_val:.2f} \times \sqrt{{{L_val:.1f}}} = {sigma_LT:.4f} \text{{ units}}"
    elif not is_d_variable and is_lt_variable:
        case_type = "constant_demand_variable_lead_time"
        case_description = "Case 3: Constant Demand and Variable Lead Time"
        sigma_LT = math.sqrt((d_val ** 2) * (sig_L_val ** 2))
        formula_sigma_str = r"\sigma_{LT} = \sqrt{d^2 \times \sigma_L^2} = d \times \sigma_L"
        sub_sigma_str = rf"\sigma_{{LT}} = {d_val:.2f} \times {sig_L_val:.2f} = {sigma_LT:.4f} \text{{ units}}"
    else:
        case_type = "variable_demand_variable_lead_time"
        case_description = "Case 4: Variable Demand and Variable Lead Time"
        sigma_LT = math.sqrt(L_val * (sig_d_val ** 2) + (d_val ** 2) * (sig_L_val ** 2))
        formula_sigma_str = r"\sigma_{LT} = \sqrt{L \times \sigma_d^2 + d^2 \times \sigma_L^2}"
        sub_sigma_str = (
            rf"\sigma_{{LT}} = \sqrt{{{L_val:.1f} \times ({sig_d_val:.2f})^2 + ({d_val:.2f})^2 \times ({sig_L_val:.2f})^2}} = "
            rf"\sqrt{{{L_val * (sig_d_val**2):.2f} + {(d_val**2) * (sig_L_val**2):.2f}}} = {sigma_LT:.4f} \text{{ units}}"
        )

    # Validate uncertainty input requirements
    if (is_d_variable or is_lt_variable) and safety_stock is None and z_score is None and service_level is None:
        raise ValueError(
            "Service level, z-score, or safety stock is required when demand or lead time is variable (uncertainty exists). Never invent missing inputs."
        )

    # Determine Safety Factor z and Safety Stock
    if safety_stock is not None:
        ss_val = float(safety_stock)
        z = (ss_val / sigma_LT) if sigma_LT > 0 else 0.0
        csl = NormalDist().cdf(z) if sigma_LT > 0 else 1.0
        ss_sub_str = rf"\text{{SS}} = {ss_val:.2f} \text{{ units (explicitly specified)}}"
    elif z_score is not None:
        z = float(z_score)
        ss_val = z * sigma_LT
        csl = NormalDist().cdf(z)
        ss_sub_str = rf"\text{{SS}} = {z:.3f} \times {sigma_LT:.4f} = {ss_val:.2f} \text{{ units}}"
    elif service_level is not None:
        csl = service_level if service_level <= 1.0 else (service_level / 100.0)
        if csl <= 0.0 or csl >= 1.0:
            raise ValueError(f"Service level must be strictly between 0 and 1 (or 0% and 100%), got: {service_level}")
        z = NormalDist().inv_cdf(csl) if sigma_LT > 0 else 0.0
        ss_val = z * sigma_LT
        ss_sub_str = rf"\text{{SS}} = z(\text{{CSL}} = {csl*100:.1f}\%) \times \sigma_{{LT}} = {z:.3f} \times {sigma_LT:.4f} = {ss_val:.2f} \text{{ units}}"
    else:
        z = 0.0
        ss_val = 0.0
        csl = 1.0
        ss_sub_str = r"\text{SS} = 0.00 \text{ units (deterministic demand & lead time)}"

    expected_ddlt = d_val * L_val
    reorder_point = expected_ddlt + ss_val

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand_mean": d_val,
        "lead_time_mean": L_val,
        "demand_std": sig_d_val,
        "lead_time_std": sig_L_val,
        "service_level": round(csl, 4) if csl is not None else None,
        "z_score": round(z, 4),
        "safety_stock": round(ss_val, 2) if safety_stock is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average customer demand rate per period (units/period)",
        "\\sigma_d": "Standard deviation of customer demand per period (units/period)",
        "L": "Average replenishment lead time duration (periods)",
        "\\sigma_L": "Standard deviation of replenishment lead time (periods)",
        "CSL": "Cycle Service Level / target probability of no stockout during lead time",
        "z": "Standard normal safety factor corresponding to CSL, where \\Phi(z) = CSL",
        "\\sigma_{LT}": "Combined standard deviation of demand during replenishment lead time",
        "DDLT": "Expected Demand During Lead Time = d * L (units)",
        "SS": "Safety Stock inventory buffer = z * \\sigma_{LT} (units)",
        "ROP": "Reorder Point replenishment threshold = DDLT + SS (units)",
    }

    # 3. Formula
    formula = {
        "Scenario Case": case_type,
        "Expected Demand During Lead Time (DDLT)": r"\text{DDLT} = d \times L",
        "Lead Time Uncertainty (\\sigma_{LT})": formula_sigma_str,
        "Safety Stock (SS)": r"\text{SS} = z \times \sigma_{LT}",
        "Reorder Point (ROP)": r"\text{ROP} = \text{DDLT} + \text{SS} = (d \times L) + (z \times \sigma_{LT})",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Expected Demand During Lead Time": rf"\text{{DDLT}} = {d_val:.2f} \times {L_val:.2f} = {expected_ddlt:,.2f} \text{{ units}}",
        "Lead Time Uncertainty (\\sigma_{LT})": sub_sigma_str,
        "Safety Stock": ss_sub_str,
        "Reorder Point": rf"\text{{ROP}} = {expected_ddlt:,.2f} + {ss_val:.2f} = \mathbf{{{reorder_point:,.2f}\text{{ units}}}} \ (\text{{trigger replenishment at }} {round(reorder_point):,} \text{{ units}})",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "case_type": case_type,
        "expected_demand_during_lead_time": round(expected_ddlt, 2),
        "sigma_LT": round(sigma_LT, 4),
        "z_score": round(z, 4),
        "cycle_service_level": round(csl * 100, 2) if csl is not None else None,
        "safety_stock": round(ss_val, 2),
        "reorder_point": round(reorder_point, 2),
    }

    # 6. Final Answer
    final_ans = {
        "reorder_point": round(reorder_point, 2),
        "reorder_point_rounded": round(reorder_point),
        "safety_stock": round(ss_val, 2),
        "safety_stock_rounded": round(ss_val),
        "expected_lead_time_demand": round(expected_ddlt, 2),
        "stockout_probability_per_cycle_pct": round((1.0 - csl) * 100, 2) if csl is not None else 0.0,
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Safety Stock Non-Negative",
            "condition": "SS >= 0",
            "value": round(ss_val, 2),
            "passed": bool(ss_val >= 0),
            "explanation": "Safety stock buffer must never be negative.",
        },
        {
            "check": "ROP >= Expected Lead Time Demand",
            "condition": "ROP >= DDLT",
            "value": round(reorder_point, 2),
            "passed": bool(reorder_point >= expected_ddlt),
            "explanation": "Reorder point must at least cover expected replenishment lead time demand.",
        },
        {
            "check": "ROP >= Safety Stock",
            "condition": "ROP >= SS",
            "value": round(reorder_point, 2),
            "passed": bool(reorder_point >= ss_val),
            "explanation": "Reorder point must be greater than or equal to the safety stock buffer.",
        },
        {
            "check": "Sigma LT Non-Negative Invariant",
            "condition": "sigma_LT >= 0",
            "value": round(sigma_LT, 4),
            "passed": bool(sigma_LT >= 0),
            "explanation": "Standard deviation of lead time demand must be a real non-negative quantity.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Continuous inventory review policy: inventory position is monitored in real-time.",
        "Lead time demand uncertainty is normally distributed with independent period demands.",
        "Unfilled demand occurring during replenishment cycles is fully backordered.",
        "Cycle service level (CSL) defines probability that no stockout occurs during lead time.",
        "Demand rate d and lead time L are stationary throughout the planning horizon.",
    ]

    return ROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        verification_checks=verification_checks,
        assumptions=assumptions,
        reorder_point=reorder_point,
        safety_stock=ss_val,
        case_type=case_type,
        demand_during_lead_time_mean=expected_ddlt,
        sigma_LT=sigma_LT,
        z_score=z,
        cycle_service_level=csl,
        demand_mean=d_val,
        lead_time_mean=L_val,
        demand_std=sig_d_val,
        lead_time_std=sig_L_val,
    )


calculate_safety_stock_rop = calculate_rop


# ============================================================================
# 4. Fixed-Period Review (calculate_periodic_review)
# ============================================================================

@dataclass
class PeriodicReviewResult:
    # 8 Student Verification Components
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, Any]]
    assumptions: List[str]

    # Typed Attributes & Aliases
    target_inventory: float
    recommended_order_quantity: Optional[float]
    safety_stock: float
    protection_period: float
    expected_demand_protection_period: float
    sigma_protection_period: float
    z_score: float
    cycle_service_level: Optional[float]
    inventory_position: Optional[float]
    review_period: float
    lead_time: float
    demand_mean: float
    demand_std: float
    on_hand_inventory: Optional[float] = None
    on_order_inventory: float = 0.0
    backorders: float = 0.0

    @property
    def order_up_to_level(self) -> float:
        return self.target_inventory

    @property
    def protection_interval(self) -> float:
        return self.protection_period

    @property
    def expected_demand_protection_interval(self) -> float:
        return self.expected_demand_protection_period

    @property
    def std_protection_interval(self) -> float:
        return self.sigma_protection_period

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Fixed-Period Periodic Review (Base Stock Policy)",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


FixedPeriodReviewResult = PeriodicReviewResult


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    safety_stock: Optional[float] = None,
    inventory_position: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: float = 0.0,
    backorders: float = 0.0,
    # Backward compatibility aliases:
    demand_mean_per_period: Optional[float] = None,
    demand_std_per_period: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Computes periodic review parameters (Order-Up-To Level / Target Inventory):
      Protection period = review period + lead time
      Target inventory = demand during protection period + safety stock
      Order quantity = max(0, target inventory - inventory position)
    Returns full 8-component student verification model.
    """
    d_m = demand_mean if demand_mean is not None else demand_mean_per_period
    d_s = demand_std if demand_std != 0.0 or demand_std_per_period is None else demand_std_per_period

    if review_period is None or not isinstance(review_period, (int, float)) or review_period <= 0:
        raise ValueError(f"Review period (P) must be a strictly positive number, got: {review_period}")
    if lead_time is None or not isinstance(lead_time, (int, float)) or lead_time < 0:
        raise ValueError(f"Lead time (L) must be a non-negative number, got: {lead_time}")
    if d_m is None or not isinstance(d_m, (int, float)) or d_m < 0:
        raise ValueError(f"Average demand per period must be non-negative, got: {d_m}")
    if d_s is None or not isinstance(d_s, (int, float)) or d_s < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got: {d_s}")

    P_val = float(review_period)
    L_val = float(lead_time)
    d_val = float(d_m)
    sig_d_val = float(d_s)

    # Protection period = P + L
    protection_period = P_val + L_val
    expected_demand_protection_period = d_val * protection_period
    sigma_protection_period = sig_d_val * math.sqrt(protection_period)

    # Validate uncertainty input requirements
    if sigma_protection_period > 0 and safety_stock is None and z_score is None and service_level is None:
        raise ValueError(
            "Service level, z-score, or safety stock is required when demand is variable (uncertainty exists). Never invent missing inputs."
        )

    # Safety Stock
    if safety_stock is not None:
        ss_val = float(safety_stock)
        z = (ss_val / sigma_protection_period) if sigma_protection_period > 0 else 0.0
        csl = NormalDist().cdf(z) if sigma_protection_period > 0 else 1.0
        ss_sub_str = rf"\text{{SS}} = {ss_val:.2f} \text{{ units (explicitly specified)}}"
    elif z_score is not None:
        z = float(z_score)
        ss_val = z * sigma_protection_period
        csl = NormalDist().cdf(z)
        ss_sub_str = rf"\text{{SS}} = {z:.3f} \times {sigma_protection_period:.4f} = {ss_val:.2f} \text{{ units}}"
    elif service_level is not None:
        csl = service_level if service_level <= 1.0 else (service_level / 100.0)
        if csl <= 0.0 or csl >= 1.0:
            raise ValueError(f"Service level must be between 0 and 1, got: {service_level}")
        z = NormalDist().inv_cdf(csl) if sigma_protection_period > 0 else 0.0
        ss_val = z * sigma_protection_period
        ss_sub_str = rf"\text{{SS}} = z(\text{{CSL}} = {csl*100:.1f}\%) \times \sigma_{{P+L}} = {z:.3f} \times {sigma_protection_period:.4f} = {ss_val:.2f} \text{{ units}}"
    else:
        z = 0.0
        ss_val = 0.0
        csl = 1.0
        ss_sub_str = r"\text{SS} = 0.00 \text{ units (deterministic)}"

    target_inventory = expected_demand_protection_period + ss_val

    # Inventory position and recommended order quantity
    ip = None
    if inventory_position is not None:
        ip = float(inventory_position)
    elif on_hand_inventory is not None:
        ip = float(on_hand_inventory) + float(on_order_inventory) - float(backorders)

    order_qty = None
    order_sub_str = r"Q = \text{Requires current inventory position}"
    if ip is not None:
        order_qty = max(0.0, target_inventory - ip)
        order_sub_str = rf"Q = \max(0, M - \text{{IP}}) = \max(0, {target_inventory:.2f} - {ip:.2f}) = \mathbf{{{order_qty:.2f}\text{{ units}}}}"

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period": P_val,
        "lead_time": L_val,
        "demand_mean": d_val,
        "demand_std": sig_d_val,
        "service_level": round(csl, 4) if csl is not None else None,
        "z_score": round(z, 4),
        "safety_stock": round(ss_val, 2) if safety_stock is not None else None,
        "inventory_position": round(ip, 2) if ip is not None else None,
        "on_hand_inventory": round(float(on_hand_inventory), 2) if on_hand_inventory is not None else None,
        "on_order_inventory": round(float(on_order_inventory), 2),
        "backorders": round(float(backorders), 2),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "P": "Review period duration / time interval between reorders (periods)",
        "L": "Replenishment delivery lead time duration (periods)",
        "P + L": "Protection period / total duration inventory must protect against stockouts",
        "d": "Average customer demand rate per review period (units/period)",
        "\\sigma_d": "Standard deviation of demand per period (units/period)",
        "\\sigma_{P+L}": "Standard deviation of demand over protection period = \\sigma_d * sqrt(P + L)",
        "CSL": "Cycle service level / target in-stock probability over protection period",
        "z": "Normal safety factor corresponding to CSL, where \\Phi(z) = CSL",
        "SS": "Safety Stock inventory buffer = z * \\sigma_d * sqrt(P + L) (units)",
        "M": "Target Order-Up-To Level / Base Stock = d * (P + L) + SS (units)",
        "IP": "Current inventory position = On-Hand + On-Order - Backorders (units)",
        "Q": "Recommended Replenishment Order Quantity = max(0, M - IP) (units)",
    }

    # 3. Formula
    formula = {
        "Protection Period": r"\text{Protection Period} = P + L",
        "Expected Demand over Protection Period": r"\text{Expected Demand} = d \times (P + L)",
        "Demand Std Dev over Protection Period": r"\sigma_{P+L} = \sigma_d \sqrt{P + L}",
        "Safety Stock": r"\text{SS} = z \times \sigma_d \sqrt{P + L}",
        "Target Inventory Level (Order-Up-To M)": r"M = d \times (P + L) + \text{SS} = d \times (P + L) + z \times \sigma_d \sqrt{P + L}",
        "Inventory Position (IP)": r"\text{IP} = \text{On-Hand} + \text{On-Order} - \text{Backorders}",
        "Recommended Order Quantity (Q)": r"Q = \max(0, M - \text{IP})",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Protection Period": rf"\text{{Protection Period}} = {P_val:.1f} + {L_val:.1f} = {protection_period:.1f} \text{{ periods}}",
        "Expected Demand over Protection Period": rf"\text{{Expected Demand}} = {d_val:.2f} \times {protection_period:.1f} = {expected_demand_protection_period:,.2f} \text{{ units}}",
        "Demand Std Dev over Protection Period": rf"\sigma_{{P+L}} = {sig_d_val:.2f} \times \sqrt{{{protection_period:.1f}}} = {sigma_protection_period:.4f} \text{{ units}}",
        "Safety Stock": ss_sub_str,
        "Target Inventory Level (M)": rf"M = {expected_demand_protection_period:,.2f} + {ss_val:.2f} = \mathbf{{{target_inventory:,.2f}\text{{ units}}}} \ (\text{{Base stock }} {round(target_inventory):,} \text{{ units}})",
        "Inventory Position": rf"\text{{IP}} = {float(on_hand_inventory or 0):.1f} + {float(on_order_inventory):.1f} - {float(backorders):.1f} = {ip:.1f} \text{{ units}}" if ip is not None else r"\text{IP not provided}",
        "Recommended Order Quantity": order_sub_str,
    }

    # 5. Intermediate Calculations
    intermediate = {
        "protection_period": round(protection_period, 2),
        "expected_demand_protection_period": round(expected_demand_protection_period, 2),
        "sigma_protection_period": round(sigma_protection_period, 4),
        "z_score": round(z, 4),
        "cycle_service_level": round(csl * 100, 2) if csl is not None else None,
        "safety_stock": round(ss_val, 2),
        "target_inventory_M": round(target_inventory, 2),
        "inventory_position": round(ip, 2) if ip is not None else None,
        "order_quantity": round(order_qty, 2) if order_qty is not None else None,
    }

    # 6. Final Answer
    final_ans = {
        "target_inventory_level_M": round(target_inventory, 2),
        "target_inventory_order_up_to": round(target_inventory, 2),
        "target_inventory_level_M_rounded": round(target_inventory),
        "safety_stock": round(ss_val, 2),
        "safety_stock_rounded": round(ss_val),
        "protection_period": round(protection_period, 2),
        "recommended_order_quantity": round(order_qty, 2) if order_qty is not None else None,
        "recommended_order_quantity_rounded": round(order_qty) if order_qty is not None else None,
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Positive Protection Interval",
            "condition": "P + L > 0",
            "value": round(protection_period, 2),
            "passed": bool(protection_period > 0),
            "explanation": "Review cycle plus delivery lead time must be positive.",
        },
        {
            "check": "Safety Stock Non-Negative",
            "condition": "SS >= 0",
            "value": round(ss_val, 2),
            "passed": bool(ss_val >= 0),
            "explanation": "Safety stock required for periodic review must be non-negative.",
        },
        {
            "check": "Order-Up-To Level >= Expected Demand",
            "condition": "M >= Expected Demand",
            "value": round(target_inventory, 2),
            "passed": bool(target_inventory >= expected_demand_protection_period),
            "explanation": "Target inventory base stock must provide coverage for expected demand during protection period.",
        },
    ]
    if order_qty is not None:
        verification_checks.append({
            "check": "Order Quantity Non-Negative",
            "condition": "Q >= 0",
            "value": round(order_qty, 2),
            "passed": bool(order_qty >= 0),
            "explanation": "Order quantity must be non-negative; excess inventory position suppresses ordering.",
        })

    # 8. Assumptions
    assumptions = [
        "Periodic review base stock policy: inventory status is reviewed strictly at fixed time intervals P.",
        "Orders placed at review events arrive after delivery lead time L.",
        "Buffer inventory must protect against demand uncertainty over the combined duration P + L.",
        "Periodic demand is independent and identically normally distributed across consecutive review cycles.",
        "Unsatisfied customer demand during stockouts is fully backordered.",
    ]

    return PeriodicReviewResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        verification_checks=verification_checks,
        assumptions=assumptions,
        target_inventory=target_inventory,
        recommended_order_quantity=order_qty,
        safety_stock=ss_val,
        protection_period=protection_period,
        expected_demand_protection_period=expected_demand_protection_period,
        sigma_protection_period=sigma_protection_period,
        z_score=z,
        cycle_service_level=csl,
        inventory_position=ip,
        review_period=P_val,
        lead_time=L_val,
        demand_mean=d_val,
        demand_std=sig_d_val,
        on_hand_inventory=float(on_hand_inventory) if on_hand_inventory is not None else None,
        on_order_inventory=float(on_order_inventory),
        backorders=float(backorders),
    )


calculate_fixed_period_review = calculate_periodic_review


# ============================================================================
# 5. Newsvendor Model (calculate_newsvendor)
# ============================================================================

@dataclass
class NewsvendorResult:
    # 8 Student Verification Components
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, Any]]
    assumptions: List[str]

    # Typed Attributes & Aliases
    recommended_order_quantity: float
    recommended_order_quantity_int: int
    critical_ratio: float
    cost_of_underage_cu: float
    cost_of_overage_co: float
    z_score: float
    expected_lost_sales: float
    expected_sales: float
    expected_leftover_units: float
    expected_profit: float
    in_stock_probability: float
    selling_price: float
    cost_per_unit: float
    salvage_value: float
    shortage_penalty: float
    demand_mean: float
    demand_std: float

    @property
    def optimal_order_quantity(self) -> float:
        return self.recommended_order_quantity

    @property
    def optimal_order_quantity_int(self) -> int:
        return self.recommended_order_quantity_int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Newsvendor Model (Single-Period Stochastic)",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_newsvendor(
    selling_price: Optional[float] = None,
    cost_per_unit: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    demand_distribution: str = "Normal",
    # Backward compatibility alias:
    penalty_cost: Optional[float] = None,
) -> NewsvendorResult:
    """
    Computes single-period Newsvendor model optimal stocking decisions:
      Cu (underage cost) = selling_price - cost_per_unit + shortage_penalty (or explicit underage_cost)
      Co (overage cost) = cost_per_unit - salvage_value (or explicit overage_cost)
      Critical ratio (CR) = Cu / (Cu + Co)
      Recommended order quantity Q* = mu + z * sigma where Phi(z) = CR
    Returns full 8-component student verification model.
    """
    g = shortage_penalty if shortage_penalty != 0.0 or penalty_cost is None else penalty_cost

    if demand_mean is None or not isinstance(demand_mean, (int, float)) or demand_mean < 0:
        raise ValueError(f"Demand mean (mu) must be provided as a non-negative number, got: {demand_mean}")
    if demand_std is None or not isinstance(demand_std, (int, float)) or demand_std <= 0:
        raise ValueError(f"Demand standard deviation (sigma) must be strictly positive, got: {demand_std}")

    mu_val = float(demand_mean)
    sig_val = float(demand_std)
    V_val = float(salvage_value) if salvage_value is not None else 0.0
    g_val = float(g) if g is not None else 0.0

    if V_val < 0:
        raise ValueError(f"Salvage value cannot be negative, got: {salvage_value}")
    if g_val < 0:
        raise ValueError(f"Shortage penalty cannot be negative, got: {g_val}")

    if underage_cost is not None and overage_cost is not None:
        Cu = float(underage_cost)
        Co = float(overage_cost)
        P_val = float(selling_price) if selling_price is not None else 0.0
        C_val = float(cost_per_unit) if cost_per_unit is not None else 0.0
        cu_sub_str = rf"C_u = \${Cu:.2f} \text{{ (directly specified underage cost)}}"
        co_sub_str = rf"C_o = \${Co:.2f} \text{{ (directly specified overage cost)}}"
    else:
        if selling_price is None or not isinstance(selling_price, (int, float)) or selling_price <= 0:
            raise ValueError(f"Selling price (P) must be strictly positive, got: {selling_price}")
        if cost_per_unit is None or not isinstance(cost_per_unit, (int, float)) or cost_per_unit <= 0:
            raise ValueError(f"Cost per unit (C) must be strictly positive, got: {cost_per_unit}")

        P_val = float(selling_price)
        C_val = float(cost_per_unit)

        if P_val <= C_val:
            raise ValueError(f"Selling price (${P_val:.2f}) must exceed unit cost (${C_val:.2f}) for positive underage profit.")
        if C_val < V_val:
            raise ValueError(f"Unit cost (${C_val:.2f}) must exceed salvage value (${V_val:.2f}) to incur an overage loss.")

        Cu = P_val - C_val + g_val
        Co = C_val - V_val
        cu_sub_str = rf"C_u = P - C + g = \${P_val:.2f} - \${C_val:.2f} + \${g_val:.2f} = \mathbf{{\${Cu:.2f}}}"
        co_sub_str = rf"C_o = C - V = \${C_val:.2f} - \${V_val:.2f} = \mathbf{{\${Co:.2f}}}"

    if Cu <= 0:
        raise ValueError(f"Cost of underage Cu must be strictly positive, got: {Cu}")
    if Co <= 0:
        raise ValueError(f"Cost of overage Co must be strictly positive, got: {Co}")

    # Critical ratio
    CR = Cu / (Cu + Co)

    # Standard normal quantile z where Phi(z) = CR
    dist = NormalDist(mu=0.0, sigma=1.0)
    z = dist.inv_cdf(CR)

    # Optimal stocking quantity
    Q_star = mu_val + z * sig_val
    Q_star_int = round(Q_star)

    # Unit Normal Loss function G(z) = phi(z) - z * [1 - Phi(z)]
    phi_z = dist.pdf(z)
    Phi_z = dist.cdf(z)
    loss_G_z = phi_z - z * (1.0 - Phi_z)

    # Expected lost sales, expected sales, expected leftovers, and profit
    exp_lost = sig_val * loss_G_z
    exp_sales = mu_val - exp_lost
    exp_leftover = Q_star - exp_sales

    exp_rev = P_val * exp_sales
    exp_salv = V_val * exp_leftover
    exp_proc = C_val * Q_star
    exp_pen = g_val * exp_lost
    exp_profit = exp_rev + exp_salv - exp_proc - exp_pen

    # 1. Inputs Extracted
    inputs_extracted = {
        "selling_price": P_val if P_val > 0 else None,
        "cost_per_unit": C_val if C_val > 0 else None,
        "demand_mean": mu_val,
        "demand_std": sig_val,
        "salvage_value": V_val,
        "shortage_penalty": g_val,
        "underage_cost": round(Cu, 2),
        "overage_cost": round(Co, 2),
        "demand_distribution": demand_distribution,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "P": "Unit retail selling price ($/unit)",
        "C": "Unit wholesale procurement cost ($/unit)",
        "V": "Unit clearance / salvage value for unsold leftover inventory ($/unit)",
        "g": "Unit goodwill loss / customer stockout penalty ($/unit)",
        "C_u": "Cost of Underage (marginal profit of stocking one more unit) = P - C + g ($/unit)",
        "C_o": "Cost of Overage (marginal loss of stocking one unsold unit) = C - V ($/unit)",
        "CR": "Critical Ratio / Critical Fractile = C_u / (C_u + C_o)",
        "\\mu": "Mean customer demand for the single selling season (units)",
        "\\sigma": "Standard deviation of customer demand for the season (units)",
        "z^*": "Standard normal safety factor / quantile satisfying \\Phi(z^*) = CR",
        "Q^*": "Recommended optimal procurement order quantity = \\mu + z^* * \\sigma (units)",
        "G(z)": "Standard Unit Normal Loss function value = \\phi(z) - z * [1 - \\Phi(z)]",
        "\\text{Lost Sales}": "Expected unmet customer demand = \\sigma * G(z^*) (units)",
        "\\text{Sales}": "Expected units sold to customers = \\mu - Expected Lost Sales (units)",
        "\\text{Leftover}": "Expected unsold units salvaged at end of season = Q^* - Expected Sales (units)",
        "\\text{Expected Profit}": "Total expected operating profit over the single selling horizon ($)",
    }

    # 3. Formula
    formula = {
        "Cost of Underage (Cu)": r"C_u = P - C + g",
        "Cost of Overage (Co)": r"C_o = C - V",
        "Critical Ratio (Critical Fractile)": r"\text{CR} = \frac{C_u}{C_u + C_o}",
        "Optimal Order Quantity (Q*)": r"Q^* = \mu + z^* \times \sigma \quad \text{where } \Phi(z^*) = \text{CR}",
        "Unit Normal Loss Function": r"G(z^*) = \phi(z^*) - z^* \times [1 - \Phi(z^*)]",
        "Expected Lost Sales": r"\text{Lost Sales} = \sigma \times G(z^*)",
        "Expected Sales": r"\text{Sales} = \mu - \text{Lost Sales}",
        "Expected Leftover Units": r"\text{Leftover} = Q^* - \text{Sales}",
        "Expected Operating Profit": r"\text{Expected Profit} = (P \times \text{Sales}) + (V \times \text{Leftover}) - (C \times Q^*) - (g \times \text{Lost Sales})",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Cost of Underage": cu_sub_str,
        "Cost of Overage": co_sub_str,
        "Critical Ratio": rf"\text{{CR}} = \frac{{{Cu:.2f}}}{{{Cu:.2f} + {Co:.2f}}} = \frac{{{Cu:.2f}}}{{{Cu + Co:.2f}}} = \mathbf{{{CR:.4f}}} \ ({CR*100:.2f}\%)",
        "Safety Factor z*": rf"z^* = \Phi^{{-1}}({CR:.4f}) = \mathbf{{{z:.3f}}}",
        "Optimal Order Quantity": rf"Q^* = {mu_val:,.1f} + ({z:.3f} \times {sig_val:,.1f}) = {mu_val:,.1f} + {z * sig_val:,.1f} = \mathbf{{{Q_star:,.2f}\text{{ units}}}} \ (\text{{procure }} {Q_star_int:,} \text{{ units}})",
        "Expected Sales": rf"\text{{Sales}} = {mu_val:,.1f} - ({sig_val:,.1f} \times {loss_G_z:.5f}) = {mu_val:,.1f} - {exp_lost:.2f} = \mathbf{{{exp_sales:,.2f}\text{{ units}}}}",
        "Expected Leftover": rf"\text{{Leftover}} = {Q_star:.2f} - {exp_sales:.2f} = \mathbf{{{exp_leftover:,.2f}\text{{ units}}}}",
        "Expected Profit": rf"\text{{Expected Profit}} = ({P_val:.2f} \times {exp_sales:.2f}) + ({V_val:.2f} \times {exp_leftover:.2f}) - ({C_val:.2f} \times {Q_star:.2f}) - ({g_val:.2f} \times {exp_lost:.2f}) = \mathbf{{\${exp_profit:,.2f}}}",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "cost_of_underage_cu": round(Cu, 2),
        "cost_of_overage_co": round(Co, 2),
        "critical_ratio": round(CR, 4),
        "z_score": round(z, 4),
        "unit_normal_loss_G_z": round(loss_G_z, 5),
        "expected_lost_sales": round(exp_lost, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_leftover_units": round(exp_leftover, 2),
        "expected_profit": round(exp_profit, 2),
        "in_stock_probability": round(Phi_z, 4),
    }

    # 6. Final Answer
    final_ans = {
        "optimal_order_quantity": round(Q_star, 2),
        "optimal_order_quantity_rounded": Q_star_int,
        "recommended_order_quantity": round(Q_star, 2),
        "critical_ratio": round(CR, 4),
        "cost_of_underage_cu": round(Cu, 2),
        "cost_of_overage_co": round(Co, 2),
        "expected_profit": round(exp_profit, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_lost_sales": round(exp_lost, 2),
        "expected_leftover_units": round(exp_leftover, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Critical Ratio Probability Bound",
            "condition": "0 < CR < 1",
            "value": round(CR, 4),
            "passed": bool(0.0 < CR < 1.0),
            "explanation": "Critical ratio must lie strictly between 0 and 1 for finite interior solution.",
        },
        {
            "check": "Positive Underage and Overage Costs",
            "condition": "Cu > 0 and Co > 0",
            "Cu": round(Cu, 2),
            "Co": round(Co, 2),
            "passed": bool(Cu > 0 and Co > 0),
            "explanation": "Both marginal underage profit and marginal overage loss must be positive.",
        },
        {
            "check": "Positive Order Quantity",
            "condition": "Q* > 0",
            "value": round(Q_star, 2),
            "passed": bool(Q_star > 0),
            "explanation": "Procurement quantity must be strictly positive.",
        },
        {
            "check": "Conservation of Inventory Invariant",
            "condition": "|Expected Sales + Expected Leftover - Q*| < 0.1",
            "diff": round(abs(exp_sales + exp_leftover - Q_star), 4),
            "passed": bool(abs(exp_sales + exp_leftover - Q_star) < 0.1),
            "explanation": "All procured inventory must be accounted for as either sold or leftover.",
        },
        {
            "check": "Sales Bound Invariant",
            "condition": "Expected Sales <= Mean Demand",
            "passed": bool(exp_sales <= mu_val + 0.01),
            "explanation": "Expected sales cannot exceed mean market demand.",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Single-period planning horizon with no subsequent replenishments permitted.",
        "Replenishment order decision is committed prior to the single selling season.",
        "Customer demand is stochastic and follows a continuous Normal distribution.",
        "Unsold inventory is disposed of at salvage value V with no holding costs beyond the period.",
        "Unsatisfied customer demand is permanently lost (lost sales setting).",
    ]

    return NewsvendorResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        verification_checks=verification_checks,
        assumptions=assumptions,
        recommended_order_quantity=Q_star,
        recommended_order_quantity_int=Q_star_int,
        critical_ratio=CR,
        cost_of_underage_cu=Cu,
        cost_of_overage_co=Co,
        z_score=z,
        expected_lost_sales=exp_lost,
        expected_sales=exp_sales,
        expected_leftover_units=exp_leftover,
        expected_profit=exp_profit,
        in_stock_probability=Phi_z,
        selling_price=P_val,
        cost_per_unit=C_val,
        salvage_value=V_val,
        shortage_penalty=g_val,
        demand_mean=mu_val,
        demand_std=sig_val,
    )


# ============================================================================
# 6. Multicriteria Inventory Classification (BAN 517 Methodology)
# ============================================================================

@dataclass
class MulticriteriaClassificationResult:
    # 8 Student Verification Components
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, Any]]
    assumptions: List[str]

    # Typed Attributes
    sales_class: str
    sales_score: int
    hits_class: str
    hits_score: int
    sales_weight: float
    hits_weight: float
    combined_weighted_score: float
    final_inventory_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sales_class": self.sales_class,
            "sales_score": self.sales_score,
            "hits_class": self.hits_class,
            "hits_score": self.hits_score,
            "sales_weight": self.sales_weight,
            "hits_weight": self.hits_weight,
            "combined_weighted_score": self.combined_weighted_score,
            "final_inventory_class": self.final_inventory_class,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
        }


def calculate_multicriteria_classification(
    sales_class: Optional[str] = None,
    hits_class: Optional[str] = None,
    sales_score: Optional[Union[int, float]] = None,
    hits_score: Optional[Union[int, float]] = None,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> MulticriteriaClassificationResult:
    """
    Deterministic Multicriteria Inventory Classification Calculator (BAN 517 Methodology).
    Stratifies inventory using two criteria: Sales and POS (Point-of-Sale) Hits.

    Step 1 - Classify Sales and POS Hits Separately
      Pareto Cumulative Thresholds:
        A: 0-60%
        B: 61-80%
        C: 81-90%
        D: 91-100%
      Numerical Score Mapping:
        A = 4, B = 3, C = 2, D = 1

    Step 2 - Calculate the Combined Weighted Score
      Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)

    Step 3 - Assign the Final Inventory Class
      Final Inventory Class | Combined Weighted Score
      A                     | 4.00 - 3.25
      B                     | 3.24 - 2.50
      C                     | 2.49 - 1.75
      D                     | < 1.75

    Returns MulticriteriaClassificationResult with all 8 student verification sections.
    """
    score_mapping = {"A": 4, "B": 3, "C": 2, "D": 1}
    class_mapping = {4: "A", 3: "B", 2: "C", 1: "D"}

    # Resolve Sales Class & Score
    if sales_class is not None:
        sc_clean = str(sales_class).strip().upper()
        if sc_clean not in score_mapping:
            raise ValueError(f"Invalid sales_class '{sales_class}'. Must be one of: A, B, C, D.")
        s_class = sc_clean
        s_score = score_mapping[sc_clean]
    elif sales_score is not None:
        try:
            s_score = int(round(float(sales_score)))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid sales_score '{sales_score}'. Must be integer 1, 2, 3, or 4.")
        if s_score not in class_mapping:
            raise ValueError(f"Invalid sales_score {s_score}. Must be in [1, 4] corresponding to D, C, B, A.")
        s_class = class_mapping[s_score]
    else:
        raise ValueError("Missing required input: sales_class or sales_score must be provided.")

    # Resolve Hits Class & Score
    if hits_class is not None:
        hc_clean = str(hits_class).strip().upper()
        if hc_clean not in score_mapping:
            raise ValueError(f"Invalid hits_class '{hits_class}'. Must be one of: A, B, C, D.")
        h_class = hc_clean
        h_score = score_mapping[hc_clean]
    elif hits_score is not None:
        try:
            h_score = int(round(float(hits_score)))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid hits_score '{hits_score}'. Must be integer 1, 2, 3, or 4.")
        if h_score not in class_mapping:
            raise ValueError(f"Invalid hits_score {h_score}. Must be in [1, 4] corresponding to D, C, B, A.")
        h_class = class_mapping[h_score]
    else:
        raise ValueError("Missing required input: hits_class or hits_score must be provided.")

    # Validate weights
    w_sales = float(sales_weight)
    w_hits = float(hits_weight)
    if w_sales < 0 or w_hits < 0:
        raise ValueError("Weights cannot be negative.")
    if abs((w_sales + w_hits) - 1.0) > 1e-4:
        raise ValueError(f"Weights must sum to 1.0. Given: sales_weight={w_sales}, hits_weight={w_hits} (sum={w_sales + w_hits}).")

    # Step 2: Combined Weighted Score
    # Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)
    part_sales = round(w_sales * s_score, 4)
    part_hits = round(w_hits * h_score, 4)
    combined_score = round(w_sales * s_score + w_hits * h_score, 2)

    # Step 3: Assign Final Inventory Class
    if combined_score >= 3.25:
        final_class = "A"
    elif combined_score >= 2.50:
        final_class = "B"
    elif combined_score >= 1.75:
        final_class = "C"
    else:
        final_class = "D"

    # 1. Inputs Extracted
    inputs_extracted = {
        "sales_class": s_class,
        "sales_score": s_score,
        "hits_class": h_class,
        "hits_score": h_score,
        "sales_weight": w_sales,
        "hits_weight": w_hits,
        "sales_weight_pct": f"{w_sales * 100:.1f}%",
        "hits_weight_pct": f"{w_hits * 100:.1f}%",
    }

    # 2. Variable Definitions
    variable_definitions = {
        "Sales Class": "Pareto tier for sales revenue / financial velocity (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Sales Score": "Numerical score assigned to Sales Class (A=4, B=3, C=2, D=1)",
        "Hits Class": "Pareto tier for POS pick/transaction frequency (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Hits Score": "Numerical score assigned to Hits Class (A=4, B=3, C=2, D=1)",
        "Sales Weight (w_sales)": f"Relative weight assigned to financial velocity ({w_sales * 100:.0f}%)",
        "Hits Weight (w_hits)": f"Relative weight assigned to warehouse pick activity ({w_hits * 100:.0f}%)",
        "Combined Weighted Score": "Multi-attribute composite score: Combined Weighted Score = w_sales(Sales Score) + w_hits(Hits Score)",
        "Final Inventory Class": "Overall SKU priority class: A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75)",
    }

    # 3. Formula
    formula = {
        "Step 1 - Score Conversion": r"\text{Score}(A) = 4, \quad \text{Score}(B) = 3, \quad \text{Score}(C) = 2, \quad \text{Score}(D) = 1",
        "Step 2 - Combined Weighted Score": r"\text{Combined Weighted Score} = 0.65(\text{Sales Score}) + 0.35(\text{Hits Score})",
        "Step 3 - Final Class Thresholds": r"A: [3.25, 4.00], \quad B: [2.50, 3.24], \quad C: [1.75, 2.49], \quad D: < 1.75",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Step 1 - Score Conversion": rf"\text{{Sales Score}} = {s_score} \ (\text{{Class }} {s_class}), \quad \text{{Hits Score}} = {h_score} \ (\text{{Class }} {h_class})",
        "Step 2 - Combined Weighted Score": rf"\text{{Combined Weighted Score}} = 0.65({s_score}) + 0.35({h_score}) = {part_sales:.2f} + {part_hits:.2f} = \mathbf{{{combined_score:.2f}}}",
        "Step 3 - Final Class Thresholds": rf"\text{{Score }} {combined_score:.2f} \implies \mathbf{{\text{{Final Class }} {final_class}}}",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "sales_score": s_score,
        "hits_score": h_score,
        "weighted_sales_component": part_sales,
        "weighted_hits_component": part_hits,
        "combined_weighted_score": combined_score,
        "final_inventory_class": final_class,
    }

    # 6. Final Answer
    final_ans = {
        "final_inventory_class": final_class,
        "combined_weighted_score": combined_score,
        "sales_class": s_class,
        "sales_score": s_score,
        "hits_class": h_class,
        "hits_score": h_score,
    }

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Weights Sum to 1.0 (100%)",
            "passed": abs((w_sales + w_hits) - 1.0) < 1e-4,
            "details": f"w_sales ({w_sales:.2f}) + w_hits ({w_hits:.2f}) = {w_sales + w_hits:.2f}",
        },
        {
            "check": "Sales Score within Valid Range [1, 4]",
            "passed": 1 <= s_score <= 4,
            "details": f"Sales Score = {s_score} (Class {s_class})",
        },
        {
            "check": "Hits Score within Valid Range [1, 4]",
            "passed": 1 <= h_score <= 4,
            "details": f"Hits Score = {h_score} (Class {h_class})",
        },
        {
            "check": "Combined Weighted Score within Theoretical Bounds [1.00, 4.00]",
            "passed": 1.00 <= combined_score <= 4.00,
            "details": f"Calculated Combined Weighted Score = {combined_score:.2f}",
        },
        {
            "check": "Final Inventory Class Threshold Consistency",
            "passed": (
                (final_class == "A" and combined_score >= 3.25)
                or (final_class == "B" and 2.50 <= combined_score < 3.25)
                or (final_class == "C" and 1.75 <= combined_score < 2.50)
                or (final_class == "D" and combined_score < 1.75)
            ),
            "details": f"Score {combined_score:.2f} assigned to Class {final_class}",
        },
    ]

    # 8. Assumptions
    assumptions = [
        "Step 1: Sales and POS Hits are ranked independently using Pareto cumulative thresholds: A (0-60%), B (61-80%), C (81-90%), D (91-100%).",
        "Classification to numerical score mapping follows: A = 4, B = 3, C = 2, D = 1.",
        "Step 2: Combined Weighted Score applies a 65% weight to Sales and a 35% weight to POS Hits: Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score).",
        "Step 3: Final Inventory Class follows rubric boundaries: A (4.00-3.25), B (3.24-2.50), C (2.49-1.75), D (< 1.75).",
    ]

    return MulticriteriaClassificationResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_ans,
        verification_checks=verification_checks,
        assumptions=assumptions,
        sales_class=s_class,
        sales_score=s_score,
        hits_class=h_class,
        hits_score=h_score,
        sales_weight=w_sales,
        hits_weight=w_hits,
        combined_weighted_score=combined_score,
        final_inventory_class=final_class,
    )
