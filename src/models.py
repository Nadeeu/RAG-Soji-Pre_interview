"""
Data models for Airworthiness Directive (AD) extraction pipeline.

Defines Pydantic models for:
- AD extraction output (JSON format)
- Aircraft evaluation
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from enum import Enum


class Authority(str, Enum):
    """Aviation regulatory authorities."""
    FAA = "FAA"
    EASA = "EASA"


# ============================================================
# Output Models - Format JSON yang diminta
# ============================================================

class ApplicabilityRulesOutput(BaseModel):
    """
    Applicability rules dalam format output JSON.
    
    Example:
    {
        "aircraft_models": ["MD-11", "MD-11F"],
        "msn_constraints": null,
        "excluded_if_modifications": ["SB-XXX"],
        "required_modifications": []
    }
    """
    aircraft_models: List[str] = Field(
        default_factory=list,
        description="List of applicable aircraft model designations"
    )
    msn_constraints: Optional[Any] = Field(
        None,
        description="MSN constraints (ranges, specific numbers, or null for all)"
    )
    excluded_if_modifications: List[str] = Field(
        default_factory=list,
        description="Modifications that exclude aircraft if already applied"
    )
    required_modifications: List[str] = Field(
        default_factory=list,
        description="Modifications required for AD to apply"
    )


class ADExtractedOutput(BaseModel):
    """
    Final output format untuk extracted AD rules.
    
    Example:
    {
        "ad_id": "FAA-2025-23-53",
        "applicability_rules": {
            "aircraft_models": ["MD-11", "MD-11F"],
            "msn_constraints": null,
            "excluded_if_modifications": ["SB-XXX"],
            "required_modifications": []
        }
    }
    """
    ad_id: str = Field(..., description="Unique AD identifier")
    applicability_rules: ApplicabilityRulesOutput = Field(
        default_factory=ApplicabilityRulesOutput,
        description="Extracted applicability rules"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ad_id": "FAA-2025-23-53",
                "applicability_rules": {
                    "aircraft_models": ["MD-11", "MD-11F"],
                    "msn_constraints": None,
                    "excluded_if_modifications": ["SB-XXX"],
                    "required_modifications": []
                }
            }
        }


# ============================================================
# Evaluation Models
# ============================================================

class AircraftConfiguration(BaseModel):
    """Aircraft configuration for evaluation."""
    model: str = Field(..., description="Aircraft model designation")
    msn: int = Field(..., description="Manufacturer Serial Number")
    modifications_applied: List[str] = Field(
        default_factory=list,
        description="List of applied modifications/service bulletins"
    )
    
    def __str__(self) -> str:
        mods = ", ".join(self.modifications_applied) if self.modifications_applied else "None"
        return f"{self.model} (MSN: {self.msn}, Mods: {mods})"


class EvaluationResult(BaseModel):
    """Result of evaluating an aircraft against an AD."""
    aircraft: AircraftConfiguration
    ad_id: str
    is_affected: bool
    reason: str = Field(..., description="Explanation of the evaluation result")
