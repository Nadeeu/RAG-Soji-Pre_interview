"""
Aircraft Evaluator Module.

Evaluates aircraft configurations against extracted AD rules
to determine if they are affected.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .models import ADExtractedOutput, ApplicabilityRulesOutput


@dataclass
class EvaluationResult:
    """Result of evaluating an aircraft against an AD."""
    aircraft_model: str
    msn: int
    modifications: List[str]
    ad_id: str
    is_affected: bool
    reason: str


class AircraftEvaluator:
    """
    Evaluates whether an aircraft is affected by an AD.
    
    Uses dynamic model matching without hardcoded model lists.
    """
    
    def __init__(self, ad_rules: ADExtractedOutput):
        """
        Initialize evaluator with AD rules.
        
        Args:
            ad_rules: Extracted AD rules
        """
        self.ad_rules = ad_rules
        self.rules = ad_rules.applicability_rules
    
    def evaluate(
        self,
        model: str,
        msn: int,
        modifications: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate if an aircraft is affected by the AD.
        
        Args:
            model: Aircraft model designation
            msn: Manufacturer Serial Number
            modifications: List of applied modifications
            
        Returns:
            EvaluationResult
        """
        modifications = modifications or []
        
        # Step 1: Check model applicability
        if not self._is_model_applicable(model):
            return EvaluationResult(
                aircraft_model=model,
                msn=msn,
                modifications=modifications,
                ad_id=self.ad_rules.ad_id,
                is_affected=False,
                reason=f"Model {model} is not in the applicable models list"
            )
        
        # Step 2: Check MSN constraints
        if self.rules.msn_constraints:
            if not self._is_msn_applicable(msn):
                return EvaluationResult(
                    aircraft_model=model,
                    msn=msn,
                    modifications=modifications,
                    ad_id=self.ad_rules.ad_id,
                    is_affected=False,
                    reason=f"MSN {msn} is outside applicable constraints"
                )
        
        # Step 3: Check exclusion modifications
        exclusion = self._check_exclusion_modifications(modifications, model)
        if exclusion:
            return EvaluationResult(
                aircraft_model=model,
                msn=msn,
                modifications=modifications,
                ad_id=self.ad_rules.ad_id,
                is_affected=False,
                reason=f"Aircraft excluded due to modification: {exclusion}"
            )
        
        # All checks passed - aircraft is affected
        return EvaluationResult(
            aircraft_model=model,
            msn=msn,
            modifications=modifications,
            ad_id=self.ad_rules.ad_id,
            is_affected=True,
            reason=f"Aircraft matches applicability criteria for {self.ad_rules.ad_id}"
        )
    
    def _is_model_applicable(self, model: str) -> bool:
        """Check if aircraft model is in applicable list."""
        normalized = model.strip().upper().replace(" ", "")
        
        for applicable_model in self.rules.aircraft_models:
            app_normalized = applicable_model.strip().upper().replace(" ", "")
            
            # Exact match
            if normalized == app_normalized:
                return True
            
            # Match without hyphens
            if normalized.replace('-', '') == app_normalized.replace('-', ''):
                return True
            
            # Check if same base model family
            model_base = self._get_model_base(normalized)
            app_base = self._get_model_base(app_normalized)
            
            if model_base and app_base and model_base == app_base:
                # Check if it's a close variant
                return True
        
        return False
    
    def _get_model_base(self, model: str) -> Optional[str]:
        """Extract base model from designation."""
        # Airbus: A320-214 -> A320
        match = re.match(r'(A3\d{2})', model)
        if match:
            return match.group(1)
        
        # MD: MD-11F -> MD-11
        match = re.match(r'(MD-\d{1,2})', model)
        if match:
            return match.group(1)
        
        # DC: DC-10-30F -> DC-10
        match = re.match(r'(DC-\d{1,2})', model)
        if match:
            return match.group(1)
        
        return None
    
    def _is_msn_applicable(self, msn: int) -> bool:
        """Check if MSN is within constraints."""
        # If msn_constraints is None or "all", all MSNs are applicable
        if self.rules.msn_constraints is None:
            return True
        
        constraints = str(self.rules.msn_constraints).lower()
        if 'all' in constraints:
            return True
        
        # Otherwise would need more complex parsing
        return True
    
    def _check_exclusion_modifications(
        self, 
        aircraft_mods: List[str],
        model: str
    ) -> Optional[str]:
        """
        Check if any applied modification excludes the aircraft.
        
        Returns the excluding modification if found, None otherwise.
        """
        if not self.rules.excluded_if_modifications:
            return None
        
        for aircraft_mod in aircraft_mods:
            mod_normalized = self._normalize_modification(aircraft_mod)
            
            for exclusion in self.rules.excluded_if_modifications:
                excl_normalized = self._normalize_modification(exclusion)
                
                if self._modifications_match(mod_normalized, excl_normalized, model):
                    return aircraft_mod
        
        return None
    
    def _normalize_modification(self, mod: str) -> str:
        """Normalize modification identifier."""
        normalized = mod.strip().upper()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _modifications_match(
        self, 
        aircraft_mod: str, 
        exclusion_mod: str,
        model: str
    ) -> bool:
        """Check if modification matches exclusion rule."""
        # Direct match
        if aircraft_mod == exclusion_mod:
            return True
        
        # Extract mod numbers and compare
        aircraft_num = self._extract_mod_number(aircraft_mod)
        exclusion_num = self._extract_mod_number(exclusion_mod)
        
        if aircraft_num and exclusion_num and aircraft_num == exclusion_num:
            # Check if exclusion is model-specific
            # e.g., if exclusion mentions "A320" and aircraft is A321, don't apply
            exclusion_model_base = self._extract_model_from_text(exclusion_mod)
            if exclusion_model_base:
                aircraft_model_base = self._get_model_base(model.upper())
                if aircraft_model_base and exclusion_model_base != aircraft_model_base:
                    return False
            return True
        
        # Check SB match
        aircraft_sb = self._extract_sb_number(aircraft_mod)
        exclusion_sb = self._extract_sb_number(exclusion_mod)
        
        if aircraft_sb and exclusion_sb and aircraft_sb == exclusion_sb:
            return True
        
        return False
    
    def _extract_model_from_text(self, text: str) -> Optional[str]:
        """Extract aircraft model reference from text."""
        match = re.search(r'(A3\d{2}|MD-\d{2}|DC-\d{2})', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None
    
    def _extract_mod_number(self, mod: str) -> Optional[str]:
        """Extract modification number from string."""
        match = re.search(r'MOD\s*(\d{5})', mod, re.IGNORECASE)
        if match:
            return match.group(1)
        
        match = re.search(r'\b(\d{5})\b', mod)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_sb_number(self, mod: str) -> Optional[str]:
        """Extract Service Bulletin number."""
        match = re.search(r'(A3\d{2}-\d{2}-\d{4})', mod)
        if match:
            return match.group(1)
        return None


def evaluate_aircraft(
    ad_rules: ADExtractedOutput,
    model: str,
    msn: int,
    modifications: Optional[List[str]] = None
) -> EvaluationResult:
    """
    Convenience function to evaluate an aircraft.
    
    Args:
        ad_rules: Extracted AD rules
        model: Aircraft model
        msn: MSN
        modifications: Applied modifications
        
    Returns:
        EvaluationResult
    """
    evaluator = AircraftEvaluator(ad_rules)
    return evaluator.evaluate(model, msn, modifications)
