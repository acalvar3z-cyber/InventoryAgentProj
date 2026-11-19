"""
inventory_agent.py
------------------
Supply Chain Analytics Inventory Decision Agent.

Implements the 7-stage decision pipeline architecture:
  Customer Question
        ↓
  Identify Model
        ↓
   Extract Inputs
        ↓
  Select Python Tool
        ↓
     Calculate
        ↓
      Verify
        ↓
     Explain

Numerical calculations are executed purely via Python in `inventory_tools.py`.
"""

from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import inventory_tools as it


class InventoryModelType(Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review"
    NEWSVENDOR = "Newsvendor"
    MULTICRITERIA_CLASSIFICATION = "Multicriteria Inventory Classification"
    UNKNOWN = "Unknown / General Query"


@dataclass
class AgentTraceStep:
    stage_name: str
    status: str
    data: Dict[str, Any]


@dataclass
class AgentResponse:
    question: str
    model_type: InventoryModelType
    extracted_inputs: Dict[str, Any]
    calculation_result: Any
    verification_status: Dict[str, Any]
    explanation: str
    traces: List[AgentTraceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "model_type": self.model_type.value,
            "extracted_inputs": self.extracted_inputs,
            "calculation_result": (
                self.calculation_result.to_dict()
                if hasattr(self.calculation_result, "to_dict")
                else str(self.calculation_result)
            ),
            "verification": self.verification_status,
            "explanation": self.explanation,
        }


class InventoryDecisionAgent:
    """
    Autonomous decision agent for supply chain inventory analytics.
    Follows strict pipeline:
    Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # ========================================================================
    # STAGE 1: Identify Model
    # ========================================================================
    def identify_model(self, question: str) -> InventoryModelType:
        """
        Classifies user query into one of the 5 supported inventory models.
        Uses semantic keyword scoring and pattern matching.
        """
        q = question.lower()

        # Score weights for each model
        scores = {
            InventoryModelType.EOQ: 0,
            InventoryModelType.QUANTITY_DISCOUNT: 0,
            InventoryModelType.SAFETY_STOCK_ROP: 0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0,
            InventoryModelType.NEWSVENDOR: 0,
            InventoryModelType.MULTICRITERIA_CLASSIFICATION: 0,
        }

        # 0. Multicriteria Inventory Classification indicators (BAN 517 Methodology)
        if any(term in q for term in [
            "multicriteria", "stratification", "inventory classification", "combined weighted score",
            "pos hits", "hits score", "sales score", "final inventory class", "weighted score",
            "point-of-sale hits", "sales class", "hits class", "pareto classification",
        ]):
            scores[InventoryModelType.MULTICRITERIA_CLASSIFICATION] += 12
        if "65%" in q or "35%" in q or "final category" in q or "class a and hits" in q:
            scores[InventoryModelType.MULTICRITERIA_CLASSIFICATION] += 6

        # 1. Newsvendor indicators
        if any(term in q for term in ["newsvendor", "newsboy", "single period", "salvage", "overage", "underage", "critical fractile", "critical ratio", "perishable"]):
            scores[InventoryModelType.NEWSVENDOR] += 10
        if "perishable" in q or "salvage value" in q:
            scores[InventoryModelType.NEWSVENDOR] += 5

        # 2. Quantity Discount indicators
        if any(term in q for term in ["discount", "price break", "volume discount", "tier", "all-units", "brackets", "bracket"]):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 10
        if "tier" in q or "break" in q:
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 5

        # 3. Fixed-Period Review indicators
        if any(term in q for term in ["fixed period", "fixed-period", "periodic review", "order-up-to", "base stock", "order up to", "review period", "protection interval", "target inventory"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 10
        if "reviewed every" in q or "review interval" in q or "review every" in q:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 5

        # 4. Safety Stock & ROP indicators
        if any(term in q for term in ["safety stock", "reorder point", "rop", "service level", "cycle service level", "stockout probability", "continuous review", "lead time demand"]):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 10
        if "lead time" in q and ("standard deviation" in q or "std dev" in q or "uncertainty" in q or "service level" in q):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 4

        # 5. EOQ indicators
        if any(term in q for term in [
            "eoq", "economic order quantity", "ordering cost", "order cost", "setup cost",
            "setup expense", "cost per order", "cost of placing an order", "ordering/setup cost",
            "order/setup cost", "holding cost", "carrying cost", "annual demand",
        ]):
            scores[InventoryModelType.EOQ] += 8
        if "how many units should we order each time" in q or "optimal order quantity" in q:
            scores[InventoryModelType.EOQ] += 4

        # Disambiguation logic:
        # If question has "discount", prioritize Quantity Discount over simple EOQ
        if scores[InventoryModelType.QUANTITY_DISCOUNT] >= 10:
            return InventoryModelType.QUANTITY_DISCOUNT

        # If question has "review period" or "order-up-to", prioritize Fixed-Period Review
        if scores[InventoryModelType.FIXED_PERIOD_REVIEW] >= 10:
            return InventoryModelType.FIXED_PERIOD_REVIEW

        # If question mentions salvage or single period, prioritize Newsvendor
        if scores[InventoryModelType.NEWSVENDOR] >= 10:
            return InventoryModelType.NEWSVENDOR

        # If Safety Stock / ROP is high
        if scores[InventoryModelType.SAFETY_STOCK_ROP] >= 10:
            return InventoryModelType.SAFETY_STOCK_ROP

        # Best non-zero score
        best_model = max(scores, key=scores.get)
        if scores[best_model] > 0:
            return best_model

        return InventoryModelType.UNKNOWN

    # ========================================================================
    # STAGE 2: Extract Inputs
    # ========================================================================
    def _find_number_near(
        self,
        patterns: List[str],
        text: str,
        default: Optional[float] = None,
        reject_prefix: Optional[List[str]] = None,
        reject_suffix: Optional[List[str]] = None,
        must_not_have_percent: bool = False,
    ) -> Optional[float]:
        """
        Extracts a numerical value from text matching patterns, applying immediate
        prefix/suffix validations and trailing percent checks to prevent false captures.
        """
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                # 1. Percent validation
                if must_not_have_percent:
                    trailing = text[m.end():m.end() + 10]
                    if re.match(r"^\s*(?:%|percent)", trailing, re.IGNORECASE):
                        continue

                # 2. Reject prefix (check immediate 25 chars preceding match)
                if reject_prefix:
                    prefix = text[max(0, m.start() - 25):m.start()].lower()
                    if any(re.search(rp, prefix) for rp in reject_prefix):
                        continue

                # 3. Reject suffix (check immediate 25 chars succeeding number)
                if reject_suffix:
                    suffix = text[m.end():min(len(text), m.end() + 25)].lower()
                    if any(re.search(rs, suffix) for rs in reject_suffix):
                        continue

                # 4. Parse group 1 as float
                try:
                    raw_val = m.group(1)
                    if raw_val is None:
                        continue
                    val_str = str(raw_val).replace(",", "")
                    return float(val_str)
                except (ValueError, IndexError):
                    continue
        return default

    def _extract_eoq_inputs(self, q: str) -> Dict[str, Any]:
        """
        Extracts EOQ parameters. Recognizes all ordering cost synonyms:
        - ordering cost
        - order cost
        - setup cost
        - setup expense
        - cost per order
        - cost of placing an order
        - ordering/setup cost
        """
        # Ordering cost patterns (prefix and postfix numbers)
        order_cost_patterns = [
            r"(?:cost\s*(?:of|to)\s*plac(?:ing|e)\s*(?:an?|each)?\s*order)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:ordering\s*\/\s*setup\s*cost|order\s*\/\s*setup\s*cost|ordering\s*and\s*setup\s*cost)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:setup\s*expense[s]?)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:cost\s*(?:per|\/)\s*order)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:ordering\s*cost[s]?)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:order\s*cost[s]?)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:setup\s*cost[s]?)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:is\s*(?:the\s*)?)?(?:cost\s*(?:of|to)\s*plac(?:ing|e)\s*(?:an?|each)?\s*order)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:ordering\s*\/\s*setup\s*cost|order\s*\/\s*setup\s*cost)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:setup\s*expense[s]?)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:cost\s*(?:per|\/)\s*order|(?:per|\/)\s*order)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:ordering\s*cost[s]?|order\s*cost[s]?|setup\s*cost[s]?)",
            r"\bS\b\s*(?:is|=|:)\s*\$?(\d+(?:\.\d+)?)",
        ]
        s = self._find_number_near(order_cost_patterns, q)

        # Annual demand patterns
        demand_patterns = [
            r"(?:annual\s*demand|demand\s*per\s*year|yearly\s*demand|demand\s*annually)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:units\s*(?:per|\/)\s*year|units\s*annually|units\s*a\s*year)",
            r"(?:demand|\bD\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ]
        d = self._find_number_near(
            demand_patterns,
            q,
            reject_prefix=[r"standard\s*deviation\s*(?:of)?\s*$", r"std\s*dev\s*(?:of)?\s*$"],
            reject_suffix=[r"^\s*(?:standard\s*deviation|std\s*dev|sigma)"],
        )

        # Fixed unit holding cost H (dollar amount, not percentage)
        holding_patterns = [
            r"(?:fixed\s*holding\s*cost|holding\s*cost\s*per\s*unit|carrying\s*cost\s*per\s*unit|storage\s*cost\s*per\s*unit)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:holding\s*cost|carrying\s*cost|storage\s*cost|\bH\b)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit\s*(?:per|\/)\s*year|per\s*unit\s*annually|\/\s*unit\s*\/\s*year|\/\s*unit)",
        ]
        h = self._find_number_near(
            holding_patterns,
            q,
            must_not_have_percent=True,
            reject_suffix=[r"^\s*(?:%|percent)"],
        )

        # Percentage holding cost / carrying rate
        holding_rate_patterns = [
            r"(?:percentage\s*holding\s*cost|holding\s*rate|carrying\s*rate|annual\s*holding\s*rate|inventory\s*carrying\s*rate|carrying\s*charge)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%?",
            r"(?:holding\s*cost\s*rate|holding\s*cost\s*percentage)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%?",
            r"(\d+(?:\.\d+)?)\s*%\s*(?:annual\s*)?(?:holding|carrying|of\s*item\s*cost|of\s*unit\s*price|of\s*purchase\s*price)",
            r"(?:holding\s*cost|carrying\s*cost)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%",
        ]
        h_rate = self._find_number_near(holding_rate_patterns, q)
        if h_rate is not None and h_rate > 1.0:
            h_rate = h_rate / 100.0

        # Unit purchase cost / price
        unit_cost_patterns = [
            r"(?:unit\s*purchase\s*price|purchase\s*price|unit\s*cost|unit\s*price|purchase\s*cost|item\s*cost|cost\s*per\s*unit|price\s*per\s*unit)\s*(?:is|are|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:each\s*unit\s*costs?|each\s*costs?)\s*(?:is|are|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"costs?\s*\$?(\d+(?:\.\d+)?)\s*each",
            r"\$(\d+(?:\.\d+)?)\s*(?:per\s*unit|each|\/\s*unit)",
            r"\bC\b\s*(?:is|=|:)\s*\$?(\d+(?:\.\d+)?)",
        ]
        c = self._find_number_near(unit_cost_patterns, q, default=None)

        # Operating days
        working_days = self._find_number_near([
            r"(\d+)\s*(?:working\s*days|days\s*per\s*year|operating\s*days|days\s*a\s*year)",
        ], q, default=365)

        return {
            "annual_demand": d,
            "order_cost": s,
            "holding_cost_unit": h,
            "holding_rate": h_rate,
            "unit_cost": c,
            "working_days_per_year": int(working_days),
        }

    def _extract_quantity_discount_inputs(self, q: str) -> Dict[str, Any]:
        """
        Extracts Quantity Discount inputs, distinguishing:
        - quantity break
        - unit price
        - ordering cost (all synonyms)
        - fixed holding cost
        - percentage holding cost
        - unit purchase price
        """
        eoq_ins = self._extract_eoq_inputs(q)
        d = eoq_ins["annual_demand"]
        s = eoq_ins["order_cost"]

        # Percentage holding cost
        h_rate = self._find_number_near([
            r"(?:percentage\s*holding\s*cost|holding\s*rate|carrying\s*rate|holding\s*cost\s*(?:rate|percentage)|carrying\s*charge)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%?",
            r"(\d+(?:\.\d+)?)\s*%\s*(?:annual\s*)?(?:holding|carrying|of\s*(?:unit\s*)?price|of\s*item\s*cost)",
            r"(?:holding\s*cost|carrying\s*cost)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%",
        ], q)
        if h_rate is not None and h_rate > 1.0:
            h_rate = h_rate / 100.0

        # Fixed holding cost (dollar amount per unit/year)
        h_fixed = self._find_number_near([
            r"(?:fixed\s*holding\s*cost|holding\s*cost\s*per\s*unit|carrying\s*cost\s*per\s*unit|storage\s*cost\s*per\s*unit)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)(?!\s*%)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:per\s*unit\s*(?:per|\/)\s*year|\/\s*unit\s*\/\s*year|\/\s*unit\s*holding\s*cost)",
        ], q, must_not_have_percent=True, reject_suffix=[r"^\s*(?:%|percent)"])

        # Baseline unit purchase price
        base_unit_price = self._find_number_near([
            r"(?:base(?:line)?\s*unit\s*purchase\s*price|base(?:line)?\s*unit\s*price|regular\s*(?:unit\s*)?price|standard\s*(?:unit\s*)?price|normal\s*(?:unit\s*)?price|unit\s*purchase\s*price)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:initial\s*price|list\s*price|catalog\s*price)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
        ], q)

        # Discount tiers:
        tiers: List[it.DiscountTier] = []

        # 1. Bounded ranges: "1 to 499 units at $10.00", "1-499: $10"
        bounded = re.findall(
            r"(?:(?:tier|bracket)\s*\d+[:\s]*)?(\d+)\s*(?:to|-)\s*(\d+)\s*(?:units)?(?:\s*(?:at|@|is|=|price|costs?|cost)\s*)\$?(\d+(?:\.\d+)?)",
            q,
            re.IGNORECASE,
        )
        for idx, (min_q, max_q, price) in enumerate(bounded):
            tiers.append(
                it.DiscountTier(
                    tier_name=f"Tier {idx + 1}",
                    min_qty=float(min_q),
                    max_qty=float(max_q),
                    unit_price=float(price),
                )
            )

        # 2. Unbounded ranges: "500 and above at $9.00", "500+ units at $9"
        unbounded = re.findall(
            r"(\d+)\s*(?:\+|and\s*above|or\s*more|above|and\s*up|and\s*more)\s*(?:units)?(?:\s*(?:at|@|is|=|price|costs?|cost)\s*)\$?(\d+(?:\.\d+)?)",
            q,
            re.IGNORECASE,
        )
        for min_q, price in unbounded:
            tier_idx = len(tiers) + 1
            tiers.append(
                it.DiscountTier(
                    tier_name=f"Tier {tier_idx}",
                    min_qty=float(min_q),
                    max_qty=float("inf"),
                    unit_price=float(price),
                )
            )

        # 3. Explicit single quantity break (e.g. quantity break at 500 units for $9.00 with baseline $10)
        if not tiers:
            single_break = re.search(
                r"(?:quantity\s*break|discount\s*threshold|discount\s*at)\s*(?:at|of|is|=|:)?\s*(\d+)\s*(?:units)?(?:\s*(?:at|@|price|drops\s*to|is|=|costs?)\s*)\$?(\d+(?:\.\d+)?)",
                q,
                re.IGNORECASE,
            )
            if single_break and base_unit_price is not None:
                break_qty = float(single_break.group(1))
                discount_price = float(single_break.group(2))
                tiers = [
                    it.DiscountTier("Tier 1 (Base)", 1.0, break_qty - 1.0, float(base_unit_price)),
                    it.DiscountTier("Tier 2 (Discount)", break_qty, float("inf"), discount_price),
                ]

        tiers.sort(key=lambda t: t.min_qty)

        return {
            "annual_demand": d,
            "order_cost": s,
            "holding_rate": h_rate,
            "fixed_holding_cost": h_fixed,
            "tiers": tiers,
            "unit_purchase_price": base_unit_price or (tiers[0].unit_price if tiers else None),
        }

    def _extract_rop_inputs(self, q: str) -> Dict[str, Any]:
        """
        Extracts Safety Stock & Reorder Point inputs, distinguishing:
        - average demand
        - demand standard deviation
        - average lead time
        - lead-time standard deviation
        - service level
        """
        # 1. Demand standard deviation (sigma_d)
        d_std = self._find_number_near([
            r"(?:demand\s*(?:standard\s*deviation|std\s*dev|sigma)|standard\s*deviation\s*of\s*(?:daily\s*)?demand|sigma_d)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:daily\s*demand\s*(?:standard\s*deviation|std\s*dev))\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:standard\s*deviation|std\s*dev)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:units)?\s*(?:demand\s*(?:standard\s*deviation|std\s*dev)|standard\s*deviation\s*of\s*demand)",
        ], q, default=0.0)

        # 2. Average demand (d_mean)
        d_mean = self._find_number_near([
            r"(?:average\s*(?:daily\s*)?demand|mean\s*(?:daily\s*)?demand|daily\s*demand\s*averages?|expected\s*(?:daily\s*)?demand)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:demand\s*averages?|average\s*demand|mean\s*demand|daily\s*demand)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*units\s*(?:per\s*day|\/\s*day|daily)",
            r"\bdemand\b\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ], q, reject_prefix=[r"standard\s*deviation\s*(?:of)?\s*$", r"std\s*dev\s*(?:of)?\s*$"], reject_suffix=[r"^\s*(?:standard\s*deviation|std\s*dev|sigma)"])

        # 3. Lead time standard deviation (sigma_L)
        lt_std = self._find_number_near([
            r"(?:lead\s*time\s*(?:standard\s*deviation|std\s*dev|sigma)|standard\s*deviation\s*of\s*lead\s*time|lead-time\s*(?:standard\s*deviation|std\s*dev)|sigma_l|sigma_lt)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:days?|weeks?)?\s*(?:lead\s*time\s*(?:std\s*dev|standard\s*deviation)|std\s*dev\s*of\s*lead\s*time)",
        ], q, default=0.0)

        # 4. Average lead time (L_mean)
        lt_mean = self._find_number_near([
            r"(?:average\s*lead\s*time|mean\s*lead\s*time|expected\s*lead\s*time)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:lead\s*time|\bL\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"delivery\s*(?:takes|time\s*is)\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?)",
            r"(\d+(?:\.\d+)?)\s*(?:days?|weeks?)\s*(?:lead\s*time|delivery\s*time)",
        ], q, reject_prefix=[r"standard\s*deviation\s*(?:of)?\s*$", r"std\s*dev\s*(?:of)?\s*$"], reject_suffix=[r"^\s*(?:standard\s*deviation|std\s*dev|sigma)"])

        # 5. Service level
        sl = self._find_number_near([
            r"(?:service\s*level|cycle\s*service\s*level|\bCSL\b|target\s*service\s*level)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%",
            r"(?:service\s*level|cycle\s*service\s*level|\bCSL\b|target\s*service\s*level)\s*(?:is|of|=|:|at)?\s*(0\.\d+)",
            r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s*)?service\s*level",
            r"in-stock\s*probability\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%?",
        ], q, default=None)
        if sl is not None and sl > 1.0:
            sl = sl / 100.0

        # 6. Explicit z-score
        z = self._find_number_near([
            r"(?:z[-_ ]?score|z[-_ ]?value|\bz\b)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        ], q, default=None)

        # 7. Explicit safety stock
        ss = self._find_number_near([
            r"(?:safety\s*stock|buffer\s*stock|\bSS\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"maintain\s*(?:a\s*)?safety\s*stock\s*of\s*(\d+(?:\.\d+)?)",
        ], q, default=None)

        return {
            "demand_mean": d_mean,
            "demand_std": d_std,
            "lead_time_mean": lt_mean,
            "lead_time_std": lt_std,
            "service_level": sl,
            "z_score": z,
            "safety_stock": ss,
        }

    def _extract_periodic_review_inputs(self, q: str) -> Dict[str, Any]:
        """
        Extracts Fixed-Period Review inputs, distinguishing:
        - demand (mean and standard deviation)
        - review period
        - lead time
        - safety stock
        - inventory position
        - on-hand inventory
        - on-order inventory
        - backorders
        """
        # 1. Review period P
        p = self._find_number_near([
            r"(?:review\s*period|review\s*interval|review\s*cycle|order\s*interval|ordering\s*cycle)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:reviewed\s*every|ordered\s*every|review\s*every)\s*(\d+(?:\.\d+)?)",
            r"every\s*(\d+(?:\.\d+)?)\s*(?:days?|weeks?|months?)",
            r"\bP\b\s*(?:is|=|:)\s*(\d+(?:\.\d+)?)",
        ], q)

        # 2. Lead time L
        l = self._find_number_near([
            r"(?:replenishment\s*lead\s*time|delivery\s*lead\s*time|order\s*lead\s*time|lead\s*time|\bL\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"delivery\s*(?:takes|time\s*is)\s*(\d+(?:\.\d+)?)",
        ], q)

        # 3. Demand std dev
        d_std = self._find_number_near([
            r"(?:demand\s*(?:standard\s*deviation|std\s*dev|sigma)|standard\s*deviation\s*of\s*demand|sigma_d)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:standard\s*deviation|std\s*dev|sigma)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ], q, default=0.0)

        # 4. Demand mean
        d_mean = self._find_number_near([
            r"(?:average\s*demand|mean\s*demand|expected\s*demand|demand\s*averages?)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*units\s*per\s*(?:day|week|month|period)",
            r"\bdemand\b\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ], q, reject_prefix=[r"standard\s*deviation\s*(?:of)?\s*$", r"std\s*dev\s*(?:of)?\s*$"], reject_suffix=[r"^\s*(?:standard\s*deviation|std\s*dev|sigma)"])

        # 5. Service level / z-score / safety stock
        sl = self._find_number_near([
            r"(?:service\s*level|cycle\s*service\s*level|\bCSL\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)\s*%",
            r"(?:service\s*level|cycle\s*service\s*level|\bCSL\b)\s*(?:is|of|=|:|at)?\s*(0\.\d+)",
            r"(\d+(?:\.\d+)?)\s*%\s*(?:cycle\s*)?service\s*level",
        ], q, default=None)
        if sl is not None and sl > 1.0:
            sl = sl / 100.0

        z = self._find_number_near([
            r"(?:z[-_ ]?score|z[-_ ]?value|\bz\b)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        ], q, default=None)

        ss = self._find_number_near([
            r"(?:safety\s*stock|buffer\s*stock|\bSS\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ], q, default=None)

        # 6. Inventory position, on-hand, on-order, backorders
        ip = self._find_number_near([
            r"(?:current\s*)?(?:inventory\s*position|\bIP\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ], q, default=None)

        on_hand = self._find_number_near([
            r"(?:on-hand\s*inventory|on\s*hand\s*inventory|on-hand\s*stock|stock\s*on\s*hand|on-hand|\bOH\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"current\s*(?:on-hand|stock|inventory)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*units\s*(?:on\s*hand|in\s*stock)",
            r"(?:currently\s*(?:have|is)?)\s*(\d+(?:\.\d+)?)\s*units\s*(?:on\s*hand|in\s*stock)",
        ], q, reject_prefix=[r"on[- ]?order", r"backorder"], default=None)

        on_order = self._find_number_near([
            r"(?:on-order\s*inventory|on\s*order\s*inventory|pipeline\s*inventory|open\s*orders?|on-order|\bOO\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*units\s*(?:on\s*order|in\s*transit|pipeline)",
            r"purchase\s*order\s*(?:open\s*)?for\s*(\d+(?:\.\d+)?)\s*units",
        ], q, default=0.0)

        backorders = self._find_number_near([
            r"(?:backorders?|backlog|backordered\s*units|unfilled\s*orders?|\bBO\b)\s*(?:is|are|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*units\s*(?:backordered|on\s*backorder|in\s*backlog)",
        ], q, default=0.0)

        return {
            "review_period": p,
            "lead_time": l,
            "demand_mean": d_mean,
            "demand_std": d_std,
            "service_level": sl,
            "z_score": z,
            "safety_stock": ss,
            "inventory_position": ip,
            "on_hand_inventory": on_hand,
            "on_order_inventory": on_order,
            "backorders": backorders,
        }

    def _extract_newsvendor_inputs(self, q: str) -> Dict[str, Any]:
        """
        Extracts Newsvendor inputs, distinguishing:
        - selling price
        - purchase cost
        - salvage value
        - underage / shortage cost
        - overage / excess cost
        - demand distribution
        - mean demand
        - demand standard deviation
        """
        # 1. Direct underage / shortage cost
        cu = self._find_number_near([
            r"(?:underage\s*cost|cost\s*of\s*underage|shortage\s*cost|cost\s*of\s*shortage|stockout\s*cost|lost\s*profit\s*per\s*unit|Cu|\bC_u\b)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage)",
        ], q, default=None)

        # 2. Direct overage / excess cost
        co = self._find_number_near([
            r"(?:overage\s*cost|cost\s*of\s*overage|excess\s*cost|cost\s*of\s*excess|cost\s*of\s*leftover|Co|\bC_o\b)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"\$?(\d+(?:\.\d+)?)\s*(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage)",
        ], q, default=None)

        # 3. Selling price
        p = self._find_number_near([
            r"(?:selling\s*price|retail\s*price|retail\s*selling\s*price|sale\s*price|regular\s*price|retail\s*value|\bP\b)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:sells?\s*for|sold\s*(?:for|at))\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)\s*(?:selling\s*price|retail\s*price|retail)",
        ], q, default=None)

        # 4. Purchase cost
        c = self._find_number_near([
            r"(?:unit\s*wholesale\s*cost|wholesale\s*cost|wholesale\s*price|unit\s*purchase\s*cost|purchase\s*cost|purchase\s*price|cost\s*per\s*unit|unit\s*cost|\bC\b)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:costs?\s*(?:to\s*purchase|to\s*buy|each)?|bought\s*for)\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)\s*(?:wholesale|purchase\s*cost|cost\s*per\s*unit)",
        ], q, default=None)

        # 5. Salvage value
        v = self._find_number_near([
            r"(?:salvage\s*value|salvage\s*price|scrap\s*value|clearance\s*price|liquidation\s*value|salvage|\bV\b)\s*(?:is|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
            r"(?:salvaged?\s*(?:for|at)|sold\s*on\s*clearance\s*(?:for|at))\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)\s*(?:salvage\s*value|scrap\s*value|salvage)",
        ], q, default=0.0)

        # 6. Shortage penalty
        pen = self._find_number_near([
            r"(?:shortage\s*penalty|goodwill\s*penalty|loss\s*of\s*goodwill|goodwill\s*cost|penalty\s*cost|customer\s*goodwill\s*loss|\bg\b)[^.\d$]{0,25}?(?:is|are|of|=|:|at)?\s*\$?(\d+(?:\.\d+)?)",
        ], q, default=0.0)

        # 7. Demand distribution type
        dist_type = "Normal"
        if re.search(r"\buniform\b", q, re.IGNORECASE):
            dist_type = "Uniform"

        # 8. Demand std dev
        d_std = self._find_number_near([
            r"(?:demand\s*(?:standard\s*deviation|std\s*dev|sigma)|standard\s*deviation\s*of\s*demand|sigma_d|\bsigma\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"(?:standard\s*deviation|std\s*dev)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"with\s*(?:a\s*)?standard\s*deviation\s*(?:of|is)?\s*(\d+(?:\.\d+)?)",
        ], q, default=None)

        # 9. Demand mean
        d_mean = self._find_number_near([
            r"(?:mean\s*demand|average\s*demand|expected\s*demand|demand\s*mean|demand\s*averages?|\bmu\b)\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
            r"expected\s*demand\s*(?:is\s*)?(?:mean\s*)?(\d+(?:\.\d+)?)",
            r"normally\s*distributed\s*with\s*(?:a\s*)?mean\s*(?:of|is)?\s*(\d+(?:\.\d+)?)",
            r"\bdemand\b\s*(?:is|of|=|:|at)?\s*(\d+(?:\.\d+)?)",
        ], q, reject_prefix=[r"standard\s*deviation\s*(?:of)?\s*$", r"std\s*dev\s*(?:of)?\s*$"], reject_suffix=[r"^\s*(?:standard\s*deviation|std\s*dev|sigma)"], default=None)

        return {
            "selling_price": p,
            "cost_per_unit": c,
            "demand_mean": d_mean,
            "demand_std": d_std,
            "salvage_value": v,
            "shortage_penalty": pen,
            "underage_cost": cu,
            "overage_cost": co,
            "demand_distribution": dist_type,
        }

    def _extract_multicriteria_inputs(self, q: str) -> Dict[str, Any]:
        """
        Extracts parameters for Multicriteria Inventory Classification (BAN 517 Methodology):
        - Sales Class (A, B, C, D)
        - Hits Class (A, B, C, D)
        - Sales Score (1, 2, 3, 4)
        - Hits Score (1, 2, 3, 4)
        - Sales Weight (defaults to 0.65)
        - Hits Weight (defaults to 0.35)
        """
        # Sales Class
        sales_class = None
        m = re.search(r"(?:sales\s*(?:class|tier|category)\s*(?:is|=|:)?\s*([a-d])\b|class\s*([a-d])\s*sales|sales\s*(?:in|of)?\s*class\s*([a-d])\b)", q, re.IGNORECASE)
        if m:
            sales_class = next((g.upper() for g in m.groups() if g), None)

        # Hits Class
        hits_class = None
        m = re.search(r"(?:(?:pos\s*)?hits\s*(?:class|tier|category)\s*(?:is|=|:)?\s*([a-d])\b|class\s*([a-d])\s*(?:pos\s*)?hits|(?:pos\s*)?hits\s*(?:in|of)?\s*class\s*([a-d])\b)", q, re.IGNORECASE)
        if m:
            hits_class = next((g.upper() for g in m.groups() if g), None)

        # Sales Score
        sales_score = None
        m = re.search(r"sales\s*score\s*(?:is|=|:)?\s*([1-4])\b", q, re.IGNORECASE)
        if m:
            sales_score = int(m.group(1))

        # Hits Score
        hits_score = None
        m = re.search(r"(?:pos\s*)?hits\s*score\s*(?:is|=|:)?\s*([1-4])\b", q, re.IGNORECASE)
        if m:
            hits_score = int(m.group(1))

        # Weights
        sales_weight = 0.65
        m = re.search(r"(?:weight\s*(?:of|for)?\s*)?sales\s*(?:weight)?\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*%", q, re.IGNORECASE)
        if m:
            sales_weight = float(m.group(1)) / 100.0

        hits_weight = 0.35
        m = re.search(r"(?:weight\s*(?:of|for)?\s*)?(?:pos\s*)?hits\s*(?:weight)?\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\s*%", q, re.IGNORECASE)
        if m:
            hits_weight = float(m.group(1)) / 100.0

        return {
            "sales_class": sales_class,
            "hits_class": hits_class,
            "sales_score": sales_score,
            "hits_score": hits_score,
            "sales_weight": sales_weight,
            "hits_weight": hits_weight,
        }

    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Dict[str, Any]:
        """
        Parses numerical and operational parameters from natural language.
        Examines the entire question and recognizes domain synonyms.
        """
        q = question.replace(",", "")  # Normalize 12,000 -> 12000

        if model_type == InventoryModelType.EOQ:
            return self._extract_eoq_inputs(q)
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return self._extract_quantity_discount_inputs(q)
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return self._extract_rop_inputs(q)
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return self._extract_periodic_review_inputs(q)
        elif model_type == InventoryModelType.NEWSVENDOR:
            return self._extract_newsvendor_inputs(q)
        elif model_type == InventoryModelType.MULTICRITERIA_CLASSIFICATION:
            return self._extract_multicriteria_inputs(q)

        return {}

    def extract_parameter_meanings(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        question: str,
    ) -> Dict[str, str]:
        """
        Returns human-readable operational interpretations for every extracted parameter.
        """
        meanings: Dict[str, str] = {}

        if model_type == InventoryModelType.EOQ:
            if inputs.get("annual_demand") is not None:
                meanings["annual_demand"] = f"Annual customer demand: {inputs['annual_demand']:,.0f} units/year"
            if inputs.get("order_cost") is not None:
                meanings["order_cost"] = f"Ordering / setup cost per replenishment order: ${inputs['order_cost']:,.2f}"
            if inputs.get("holding_cost_unit") is not None:
                meanings["holding_cost_unit"] = f"Fixed unit holding cost: ${inputs['holding_cost_unit']:,.2f} per unit/year"
            if inputs.get("holding_rate") is not None:
                meanings["holding_rate"] = f"Percentage holding rate: {inputs['holding_rate']*100:.1f}% of unit purchase price"
            if inputs.get("unit_cost") is not None:
                meanings["unit_cost"] = f"Unit purchase price / cost: ${inputs['unit_cost']:,.2f} per unit"
            if inputs.get("working_days_per_year") is not None:
                meanings["working_days_per_year"] = f"Operating days per year: {inputs['working_days_per_year']} days"

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if inputs.get("annual_demand") is not None:
                meanings["annual_demand"] = f"Annual customer demand: {inputs['annual_demand']:,.0f} units/year"
            if inputs.get("order_cost") is not None:
                meanings["order_cost"] = f"Ordering / setup cost per order: ${inputs['order_cost']:,.2f}"
            if inputs.get("holding_rate") is not None:
                meanings["holding_rate"] = f"Percentage holding cost rate: {inputs['holding_rate']*100:.1f}% of unit price"
            if inputs.get("fixed_holding_cost") is not None:
                meanings["fixed_holding_cost"] = f"Fixed holding cost: ${inputs['fixed_holding_cost']:,.2f} per unit/year"
            if inputs.get("unit_purchase_price") is not None:
                meanings["unit_purchase_price"] = f"Baseline unit purchase price: ${inputs['unit_purchase_price']:,.2f}"
            if inputs.get("tiers"):
                tier_descs = [f"{t.tier_name} ({t.min_qty:.0f}-{'+' if t.max_qty == float('inf') else f'{t.max_qty:.0f}'}: ${t.unit_price:.2f})" for t in inputs["tiers"]]
                meanings["tiers"] = f"Extracted {len(inputs['tiers'])} discount tiers: {', '.join(tier_descs)}"

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if inputs.get("demand_mean") is not None:
                meanings["demand_mean"] = f"Average demand rate: {inputs['demand_mean']:,.2f} units/period"
            if inputs.get("demand_std") is not None:
                meanings["demand_std"] = f"Demand standard deviation (sigma_d): {inputs['demand_std']:,.2f} units/period"
            if inputs.get("lead_time_mean") is not None:
                meanings["lead_time_mean"] = f"Average replenishment lead time (L): {inputs['lead_time_mean']:,.2f} periods"
            if inputs.get("lead_time_std") is not None:
                meanings["lead_time_std"] = f"Lead-time standard deviation (sigma_L): {inputs['lead_time_std']:,.2f} periods"
            if inputs.get("service_level") is not None:
                meanings["service_level"] = f"Desired cycle service level (CSL): {inputs['service_level']*100:.1f}%"
            if inputs.get("z_score") is not None:
                meanings["z_score"] = f"Safety factor (z-score): {inputs['z_score']:.3f}"
            if inputs.get("safety_stock") is not None:
                meanings["safety_stock"] = f"Explicit safety stock buffer: {inputs['safety_stock']:,.2f} units"

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            if inputs.get("review_period") is not None:
                meanings["review_period"] = f"Review period length (P): {inputs['review_period']:,.2f} periods"
            if inputs.get("lead_time") is not None:
                meanings["lead_time"] = f"Delivery lead time (L): {inputs['lead_time']:,.2f} periods"
            if inputs.get("demand_mean") is not None:
                meanings["demand_mean"] = f"Average demand rate: {inputs['demand_mean']:,.2f} units/period"
            if inputs.get("demand_std") is not None:
                meanings["demand_std"] = f"Demand standard deviation: {inputs['demand_std']:,.2f} units/period"
            if inputs.get("service_level") is not None:
                meanings["service_level"] = f"Target cycle service level: {inputs['service_level']*100:.1f}%"
            if inputs.get("safety_stock") is not None:
                meanings["safety_stock"] = f"Explicit safety stock: {inputs['safety_stock']:,.2f} units"
            if inputs.get("inventory_position") is not None:
                meanings["inventory_position"] = f"Total inventory position (IP): {inputs['inventory_position']:,.2f} units"
            if inputs.get("on_hand_inventory") is not None:
                meanings["on_hand_inventory"] = f"Physical on-hand inventory: {inputs['on_hand_inventory']:,.2f} units"
            if inputs.get("on_order_inventory") is not None and inputs["on_order_inventory"] > 0:
                meanings["on_order_inventory"] = f"Pipeline inventory on order: {inputs['on_order_inventory']:,.2f} units"
            if inputs.get("backorders") is not None and inputs["backorders"] > 0:
                meanings["backorders"] = f"Unfulfilled customer backorders: {inputs['backorders']:,.2f} units"

        elif model_type == InventoryModelType.NEWSVENDOR:
            if inputs.get("selling_price") is not None:
                meanings["selling_price"] = f"Retail selling price per unit (P): ${inputs['selling_price']:,.2f}"
            if inputs.get("cost_per_unit") is not None:
                meanings["cost_per_unit"] = f"Wholesale purchase cost per unit (C): ${inputs['cost_per_unit']:,.2f}"
            if inputs.get("salvage_value") is not None:
                meanings["salvage_value"] = f"Salvage / scrap value per leftover unit (V): ${inputs['salvage_value']:,.2f}"
            if inputs.get("underage_cost") is not None:
                meanings["underage_cost"] = f"Direct underage / shortage cost (Cu): ${inputs['underage_cost']:,.2f}"
            if inputs.get("overage_cost") is not None:
                meanings["overage_cost"] = f"Direct overage / excess cost (Co): ${inputs['overage_cost']:,.2f}"
            if inputs.get("shortage_penalty") is not None and inputs["shortage_penalty"] > 0:
                meanings["shortage_penalty"] = f"Goodwill / penalty cost for unsatisfied demand (g): ${inputs['shortage_penalty']:,.2f}"
            if inputs.get("demand_mean") is not None:
                meanings["demand_mean"] = f"Expected / mean demand (mu): {inputs['demand_mean']:,.2f} units"
            if inputs.get("demand_std") is not None:
                meanings["demand_std"] = f"Demand standard deviation (sigma): {inputs['demand_std']:,.2f} units"
            if inputs.get("demand_distribution") is not None:
                meanings["demand_distribution"] = f"Assumed demand distribution: {inputs['demand_distribution']}"

        elif model_type == InventoryModelType.MULTICRITERIA_CLASSIFICATION:
            if inputs.get("sales_class") is not None:
                meanings["sales_class"] = f"Sales Pareto Tier: Class {inputs['sales_class']}"
            if inputs.get("sales_score") is not None:
                meanings["sales_score"] = f"Sales numerical score: {inputs['sales_score']}"
            if inputs.get("hits_class") is not None:
                meanings["hits_class"] = f"POS Hits Pareto Tier: Class {inputs['hits_class']}"
            if inputs.get("hits_score") is not None:
                meanings["hits_score"] = f"POS Hits numerical score: {inputs['hits_score']}"
            if inputs.get("sales_weight") is not None:
                meanings["sales_weight"] = f"Sales criterion weight: {inputs['sales_weight']*100:.1f}%"
            if inputs.get("hits_weight") is not None:
                meanings["hits_weight"] = f"POS Hits criterion weight: {inputs['hits_weight']*100:.1f}%"

        return meanings

    def check_missing_parameters(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        question: str,
    ) -> Tuple[bool, List[str], Dict[str, str]]:
        """
        Examines extracted inputs against model requirements after scanning the entire question.
        Returns:
            is_complete (bool)
            missing_params (List[str])
            missing_reasons (Dict[str, str])
        """
        missing_params: List[str] = []
        missing_reasons: Dict[str, str] = {}

        if model_type == InventoryModelType.EOQ:
            if inputs.get("annual_demand") is None:
                missing_params.append("Annual Demand (D)")
                missing_reasons["Annual Demand (D)"] = "Total customer demand rate per year (e.g. 'annual demand of 12,000 units')."
            if inputs.get("order_cost") is None:
                missing_params.append("Ordering / Setup Cost (S)")
                missing_reasons["Ordering / Setup Cost (S)"] = (
                    "Fixed administrative and setup expense per replenishment order "
                    "(synonyms: ordering cost, order cost, setup cost, setup expense, cost per order, cost of placing an order, ordering/setup cost)."
                )
            if inputs.get("holding_cost_unit") is None and inputs.get("holding_rate") is None:
                missing_params.append("Holding Cost (H or h)")
                missing_reasons["Holding Cost (H or h)"] = (
                    "Cost of holding inventory per unit per year ($/unit/year) or annual percentage carrying rate (%)."
                )
            elif inputs.get("holding_rate") is not None and (inputs.get("unit_cost") is None or inputs.get("unit_cost") <= 0):
                missing_params.append("Unit Purchase Price (C)")
                missing_reasons["Unit Purchase Price (C)"] = (
                    "Required to calculate unit holding cost H = h * C when a percentage holding rate is provided."
                )

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if inputs.get("annual_demand") is None:
                missing_params.append("Annual Demand (D)")
                missing_reasons["Annual Demand (D)"] = "Total customer demand rate per year."
            if inputs.get("order_cost") is None:
                missing_params.append("Ordering / Setup Cost (S)")
                missing_reasons["Ordering / Setup Cost (S)"] = "Fixed cost per replenishment order."
            if not inputs.get("tiers") or len(inputs["tiers"]) < 2:
                missing_params.append("Quantity Breaks / Discount Tiers")
                missing_reasons["Quantity Breaks / Discount Tiers"] = (
                    "Discount pricing schedule with quantity break thresholds and corresponding unit prices (e.g. 1-499: $10, 500+: $9)."
                )
            if inputs.get("fixed_holding_cost") is None and inputs.get("holding_rate") is None:
                missing_params.append("Holding Cost")
                missing_reasons["Holding Cost"] = "Either fixed holding cost per unit/year or percentage holding rate of unit price."

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if inputs.get("demand_mean") is None:
                missing_params.append("Average Demand (d)")
                missing_reasons["Average Demand (d)"] = "Mean daily or periodic customer demand rate."
            if inputs.get("lead_time_mean") is None:
                missing_params.append("Average Lead Time (L)")
                missing_reasons["Average Lead Time (L)"] = "Replenishment delivery lead time duration."

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            if inputs.get("review_period") is None:
                missing_params.append("Review Period (P)")
                missing_reasons["Review Period (P)"] = "Interval between inventory reviews and reordering cycles (e.g. every 14 days)."
            if inputs.get("lead_time") is None:
                missing_params.append("Lead Time (L)")
                missing_reasons["Lead Time (L)"] = "Delivery lead time from order placement until receipt."
            if inputs.get("demand_mean") is None:
                missing_params.append("Average Demand (d)")
                missing_reasons["Average Demand (d)"] = "Expected demand rate per review period."

        elif model_type == InventoryModelType.NEWSVENDOR:
            if inputs.get("demand_mean") is None:
                missing_params.append("Mean Demand (mu)")
                missing_reasons["Mean Demand (mu)"] = "Expected customer demand for the single selling period."
            if inputs.get("demand_std") is None:
                missing_params.append("Demand Standard Deviation (sigma)")
                missing_reasons["Demand Standard Deviation (sigma)"] = "Standard deviation of demand reflecting market uncertainty."
            has_explicit_costs = inputs.get("underage_cost") is not None and inputs.get("overage_cost") is not None
            has_price_cost = inputs.get("selling_price") is not None and inputs.get("cost_per_unit") is not None
            if not has_explicit_costs and not has_price_cost:
                missing_params.append("Cost / Price Economics")
                missing_reasons["Cost / Price Economics"] = (
                    "Either retail selling price (P) and unit purchase cost (C), "
                    "or direct underage cost (Cu) and overage cost (Co)."
                )

        elif model_type == InventoryModelType.MULTICRITERIA_CLASSIFICATION:
            if inputs.get("sales_class") is None and inputs.get("sales_score") is None:
                missing_params.append("Sales Class or Sales Score")
                missing_reasons["Sales Class or Sales Score"] = "Sales Pareto tier (A, B, C, D) or numerical score (4, 3, 2, 1)."
            if inputs.get("hits_class") is None and inputs.get("hits_score") is None:
                missing_params.append("POS Hits Class or Hits Score")
                missing_reasons["POS Hits Class or Hits Score"] = "Point-of-Sale Hits Pareto tier (A, B, C, D) or numerical score (4, 3, 2, 1)."

        is_complete = len(missing_params) == 0
        return is_complete, missing_params, missing_reasons

    def explain_missing_inputs(
        self,
        question: str,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        meanings: Dict[str, str],
        missing_params: List[str],
        missing_reasons: Dict[str, str],
    ) -> str:
        """
        Synthesizes a structured response when information is missing from the customer query.
        Lists extracted values/meanings and provides actionable guidance on missing items.
        """
        lines = []
        lines.append(f"### Incomplete Information: Additional Parameters Required for {model_type.value}\n")

        lines.append("#### 1. Extracted Parameters & Operational Meaning")
        if meanings:
            for param, meaning in meanings.items():
                lines.append(f"- **{param}**: {meaning}")
        else:
            lines.append("- *No valid numerical parameters could be identified from the question.*")

        lines.append("\n#### 2. Missing Required Parameters")
        lines.append(f"To compute optimal inventory decisions using **{model_type.value}**, the following parameters are required but missing:\n")
        for param in missing_params:
            reason = missing_reasons.get(param, "Required for mathematical evaluation.")
            lines.append(f"- **{param}**: {reason}")

        lines.append("\n#### 3. Guidance")
        lines.append(
            "Please provide the missing numerical values in your question. "
            "The agent recognizes standard supply chain terminology and common synonyms."
        )

        return "\n".join(lines)


    # ========================================================================
    # STAGE 3: Select Python Tool
    # ========================================================================
    def select_tool(self, model_type: InventoryModelType) -> Callable:
        """
        Maps the identified model to its corresponding calculation tool in inventory_tools.py.
        """
        mapping = {
            InventoryModelType.EOQ: it.calculate_eoq,
            InventoryModelType.QUANTITY_DISCOUNT: it.calculate_quantity_discount,
            InventoryModelType.SAFETY_STOCK_ROP: it.calculate_rop,
            InventoryModelType.FIXED_PERIOD_REVIEW: it.calculate_periodic_review,
            InventoryModelType.NEWSVENDOR: it.calculate_newsvendor,
            InventoryModelType.MULTICRITERIA_CLASSIFICATION: it.calculate_multicriteria_classification,
        }
        if model_type not in mapping:
            raise ValueError(f"No calculation tool available for model type: {model_type}")
        return mapping[model_type]

    # ========================================================================
    # STAGE 4: Calculate
    # ========================================================================
    def calculate(self, tool: Callable, inputs: Dict[str, Any]) -> Any:
        """
        Executes the Python numerical calculation engine deterministically.
        Filters out None keys and extra semantic metadata keys not accepted by the tool.
        """
        sig = inspect.signature(tool)
        valid_keys = sig.parameters.keys()
        filtered_inputs = {k: v for k, v in inputs.items() if v is not None and k in valid_keys}
        return tool(**filtered_inputs)

    # ========================================================================
    # STAGE 5: Verify
    # ========================================================================
    def verify(self, result: Any, model_type: InventoryModelType, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates calculation results against domain invariants and physical constraints.
        Verification checks are computed deterministically by the Python calculation tools.
        """
        tool_checks = getattr(result, "verification_checks", None)
        if tool_checks is not None:
            is_valid = all(c.get("passed", False) for c in tool_checks)
            return {
                "status": "PASSED" if is_valid else "WARNING",
                "all_passed": is_valid,
                "checks": tool_checks,
            }

        checks: List[Dict[str, Any]] = []
        is_valid = True

        if model_type == InventoryModelType.EOQ:
            res: it.EOQResult = result
            # Check 1: EOQ > 0
            eoq_pos = res.eoq > 0
            checks.append({"rule": "Order Quantity > 0", "passed": eoq_pos, "val": res.eoq})
            # Check 2: At optimality AOC approx equals AHC
            diff = abs(res.annual_ordering_cost - res.annual_holding_cost)
            cost_balance = diff < 1.0  # Within 1 dollar
            checks.append({"rule": "AOC equals AHC at EOQ", "passed": cost_balance, "diff": round(diff, 4)})
            # Overall validity
            is_valid = eoq_pos and cost_balance

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            res: it.QuantityDiscountResult = result
            q_pos = res.optimal_order_quantity > 0
            cost_pos = res.lowest_total_cost > 0
            savings_check = res.lowest_total_cost <= res.baseline_total_cost
            checks.append({"rule": "Optimal Q > 0", "passed": q_pos, "val": res.optimal_order_quantity})
            checks.append({"rule": "Optimal Cost <= Baseline Cost", "passed": savings_check, "savings": res.annual_savings})
            is_valid = q_pos and cost_pos and savings_check

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            res: it.SafetyStockROPResult = result
            ss_non_neg = res.safety_stock >= 0
            rop_ge_ss = res.reorder_point >= res.safety_stock
            checks.append({"rule": "Safety Stock >= 0", "passed": ss_non_neg, "val": res.safety_stock})
            checks.append({"rule": "ROP >= Safety Stock", "passed": rop_ge_ss, "rop": res.reorder_point, "ss": res.safety_stock})
            is_valid = ss_non_neg and rop_ge_ss

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            res: it.FixedPeriodReviewResult = result
            ss_pos = res.safety_stock >= 0
            order_up_to_pos = res.order_up_to_level >= res.safety_stock
            checks.append({"rule": "Safety Stock >= 0", "passed": ss_pos, "val": res.safety_stock})
            checks.append({"rule": "Order-Up-To Level >= Safety Stock", "passed": order_up_to_pos, "M": res.order_up_to_level})
            is_valid = ss_pos and order_up_to_pos

        elif model_type == InventoryModelType.NEWSVENDOR:
            res: it.NewsvendorResult = result
            cr_range = 0.0 < res.critical_ratio < 1.0
            profit_eval = res.expected_profit is not None
            checks.append({"rule": "Critical Ratio in (0, 1)", "passed": cr_range, "cr": res.critical_ratio})
            checks.append({"rule": "Cu > 0 and Co > 0", "passed": (res.cost_of_underage_cu > 0 and res.cost_of_overage_co > 0)})
            is_valid = cr_range and profit_eval

        return {
            "status": "PASSED" if is_valid else "WARNING",
            "all_passed": is_valid,
            "checks": checks,
        }

    # ========================================================================
    # STAGE 6: Explain
    # ========================================================================
    def explain(
        self,
        question: str,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any,
        verification: Dict[str, Any],
    ) -> str:
        """
        Synthesizes an executive and educational decision report.
        Exposes all 8 deterministic student verification components:
          1. Inputs Extracted
          2. Variable Definitions
          3. Formula
          4. Formula with Substituted Values
          5. Intermediate Calculations
          6. Final Answer
          7. Verification Checks
          8. Assumptions
        All numerical values and substitution steps originate strictly from the deterministic Python tool.
        """
        lines = []
        lines.append(f"### Inventory Decision Analysis: {model_type.value}\n")

        # 1. Inputs Extracted
        if hasattr(result, "inputs_extracted") and result.inputs_extracted:
            lines.append("#### 1. Inputs Extracted")
            for k, v in result.inputs_extracted.items():
                if v is not None:
                    lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")
            lines.append("")

        # 2. Variable Definitions
        if hasattr(result, "variable_definitions") and result.variable_definitions:
            lines.append("#### 2. Variable Definitions")
            for var_name, var_def in result.variable_definitions.items():
                lines.append(f"- **{var_name}**: {var_def}")
            lines.append("")

        # 3. Formula
        if hasattr(result, "formula") and result.formula:
            lines.append("#### 3. Formula")
            for name, f_expr in result.formula.items():
                lines.append(f"- **{name}**:\n  $${f_expr}$$")
            lines.append("")

        # 4. Formula with Substituted Values
        if hasattr(result, "formula_with_substituted_values") and result.formula_with_substituted_values:
            lines.append("#### 4. Formula with Substituted Values")
            for name, sub_expr in result.formula_with_substituted_values.items():
                lines.append(f"- **{name}**:\n  $${sub_expr}$$")
            lines.append("")

        # 5. Intermediate Calculations
        if hasattr(result, "intermediate_calculations") and result.intermediate_calculations:
            lines.append("#### 5. Intermediate Calculations")
            for k, v in result.intermediate_calculations.items():
                if v is not None and not isinstance(v, list):
                    lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")
            lines.append("")

        # Special: Tier Evaluation Table for Quantity Discount
        if model_type == InventoryModelType.QUANTITY_DISCOUNT and hasattr(result, "tier_evaluations"):
            lines.append("##### All-Units Discount Tier Evaluation Table")
            lines.append("| Tier | Range (Units) | Unit Price | Unconstrained EOQ | Feasible Q | Ordering Cost | Holding Cost | Purchase Cost | Total Annual Cost | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in result.tier_evaluations:
                max_str = f"{t.max_qty:,.0f}" if t.max_qty != float("inf") else "Above"
                status = "**OPTIMAL**" if t.tier_name == result.optimal_tier_name else ("Price Break" if t.is_price_break else "Feasible")
                lines.append(
                    f"| {t.tier_name} | {t.min_qty:,.0f} - {max_str} | ${t.unit_price:.2f} | {t.unconstrained_eoq:,.1f} | {t.feasible_order_qty:,.1f} | ${t.annual_ordering_cost:,.0f} | ${t.annual_holding_cost:,.0f} | ${t.annual_purchase_cost:,.0f} | ${t.total_annual_cost:,.2f} | {status} |"
                )
            lines.append("")

        # 6. Final Answer
        if hasattr(result, "final_answer") and result.final_answer:
            lines.append("#### 6. Final Answer")
            for k, v in result.final_answer.items():
                lines.append(f"- **{k.replace('_', ' ').title()}**: **{v}**")
            lines.append("")

        # 7. Verification Checks
        v_checks = getattr(result, "verification_checks", None) or verification.get("checks", [])
        if v_checks:
            lines.append("#### 7. Verification Checks")
            lines.append(f"Overall Status: **{verification.get('status', 'PASSED')}**\n")
            for c in v_checks:
                status_icon = "PASSED" if c.get("passed", False) else "WARNING"
                check_name = c.get("check") or c.get("rule", "Invariant")
                cond = c.get("condition", "")
                expl = c.get("explanation", "")
                detail = f" (Condition: `{cond}`)" if cond else ""
                lines.append(f"- [{status_icon}] **{check_name}**{detail}: {expl}")
            lines.append("")

        # 8. Assumptions
        if hasattr(result, "assumptions") and result.assumptions:
            lines.append("#### 8. Operational Assumptions")
            for asm in result.assumptions:
                lines.append(f"- {asm}")
            lines.append("")

        # Managerial Insights
        lines.append("#### Managerial Insights")
        if model_type == InventoryModelType.EOQ:
            lines.append("- Ordering below EOQ increases annual ordering overhead; ordering above EOQ ties up excessive working capital in carrying inventory.")
            lines.append("- The total cost curve is relatively flat around EOQ, meaning small adjustments for packaging lots or pallet sizes will not dramatically increase cost.")
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            lines.append(f"- Taking advantage of the volume discount at **{result.optimal_tier_name}** lowers purchasing costs significantly enough to overcome the increased holding costs.")
            lines.append("- Ensure warehouse storage capacity and working capital limits accommodate larger batch sizes before signing the purchase agreement.")
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            csl = result.cycle_service_level if result.cycle_service_level is not None else 0.95
            lines.append(f"- With a {csl * 100:.1f}% Cycle Service Level, stockouts will occur in only {(1 - csl)*100:.1f}% of replenishment cycles.")
            lines.append("- To reduce safety stock without sacrificing service levels, focus on vendor lead-time compression and reducing lead-time variability.")
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            lines.append("- Periodic review systems bundle orders across multiple SKUs on fixed schedules, which reduces ordering transportation costs compared to continuous review.")
            lines.append("- However, safety stock is higher than continuous review because it must protect against demand uncertainty over $(P + L)$ rather than just $L$.")
        elif model_type == InventoryModelType.NEWSVENDOR:
            if result.critical_ratio > 0.5:
                lines.append(f"- Because profit margin ($C_u = \\${result.cost_of_underage_cu:.2f}$) is substantially higher than holding/overage loss ($C_o = \\${result.cost_of_overage_co:.2f}$), the optimal policy orders **above the mean** ({result.optimal_order_quantity_int:,} vs {result.demand_mean:,.0f}) to avoid costly stockouts.")
            else:
                lines.append(f"- Because overage penalty ($C_o = \\${result.cost_of_overage_co:.2f}$) exceeds unit profit ($C_u = \\${result.cost_of_underage_cu:.2f}$), the optimal policy is conservative and orders **below the mean**.")

        return "\n".join(lines)

    # ========================================================================
    # End-to-End Orchestrator Pipeline
    # ========================================================================
    def process_question(self, question: str) -> AgentResponse:
        """
        Executes the full 7-stage Supply Chain Decision Pipeline:
        Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
        """
        traces: List[AgentTraceStep] = []

        # Step 1: Customer Question
        traces.append(AgentTraceStep("Customer Question", "SUCCESS", {"question": question}))

        # Step 2: Identify Model
        model_type = self.identify_model(question)
        traces.append(AgentTraceStep("Identify Model", "SUCCESS", {"identified_model": model_type.value}))

        if model_type == InventoryModelType.UNKNOWN:
            explanation = (
                "### Unable to Identify Inventory Model\n"
                "The question could not be mapped to any of the 5 supported inventory models:\n"
                "1. **Economic Order Quantity (EOQ)**\n"
                "2. **Quantity Discount Model**\n"
                "3. **Safety Stock and Reorder Point (SS & ROP)**\n"
                "4. **Fixed-Period Review (Periodic Order-Up-To Level)**\n"
                "5. **Newsvendor Model (Single-Period Stochastic)**\n\n"
                "Please include relevant inventory parameters such as demand, order costs, holding costs, lead time, or prices."
            )
            return AgentResponse(
                question=question,
                model_type=model_type,
                extracted_inputs={},
                calculation_result=None,
                verification_status={"status": "FAILED", "reason": "Unknown model"},
                explanation=explanation,
                traces=traces,
            )

        # Step 3: Extract Inputs
        try:
            inputs = self.extract_inputs(question, model_type)
            meanings = self.extract_parameter_meanings(model_type, inputs, question)
            traces.append(AgentTraceStep("Extract Inputs", "SUCCESS", {"inputs": inputs, "meanings": meanings}))
        except Exception as e:
            traces.append(AgentTraceStep("Extract Inputs", "ERROR", {"error": str(e)}))
            raise

        # Check for missing required parameters after examining the entire question
        is_complete, missing_params, missing_reasons = self.check_missing_parameters(model_type, inputs, question)
        if not is_complete:
            traces.append(AgentTraceStep("Validate Inputs", "INCOMPLETE", {"missing": missing_params}))
            explanation = self.explain_missing_inputs(question, model_type, inputs, meanings, missing_params, missing_reasons)
            return AgentResponse(
                question=question,
                model_type=model_type,
                extracted_inputs=inputs,
                calculation_result=None,
                verification_status={"status": "INCOMPLETE", "missing_parameters": missing_params},
                explanation=explanation,
                traces=traces,
            )

        # Step 4: Select Python Tool
        try:
            tool = self.select_tool(model_type)
            traces.append(AgentTraceStep("Select Python Tool", "SUCCESS", {"tool_name": tool.__name__}))
        except Exception as e:
            traces.append(AgentTraceStep("Select Python Tool", "ERROR", {"error": str(e)}))
            raise

        # Step 5: Calculate
        try:
            calc_result = self.calculate(tool, inputs)
            traces.append(AgentTraceStep("Calculate", "SUCCESS", {"result": calc_result.to_dict()}))
        except Exception as e:
            traces.append(AgentTraceStep("Calculate", "ERROR", {"error": str(e)}))
            err_msg = (
                f"### Calculation Error: Missing or Invalid Inputs for {model_type.value}\n"
                f"Could not calculate results due to error: `{str(e)}`\n\n"
                f"**Extracted Inputs**: `{inputs}`\n\n"
                "Please verify that all necessary parameters were supplied in the question."
            )
            return AgentResponse(
                question=question,
                model_type=model_type,
                extracted_inputs=inputs,
                calculation_result=None,
                verification_status={"status": "ERROR", "error": str(e)},
                explanation=err_msg,
                traces=traces,
            )

        # Step 6: Verify
        verification = self.verify(calc_result, model_type, inputs)
        traces.append(AgentTraceStep("Verify", verification["status"], verification))

        # Step 7: Explain
        explanation = self.explain(question, model_type, inputs, calc_result, verification)
        traces.append(AgentTraceStep("Explain", "SUCCESS", {"explanation_length": len(explanation)}))

        return AgentResponse(
            question=question,
            model_type=model_type,
            extracted_inputs=inputs,
            calculation_result=calc_result,
            verification_status=verification,
            explanation=explanation,
            traces=traces,
        )
