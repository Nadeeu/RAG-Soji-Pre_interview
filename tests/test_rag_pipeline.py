"""
Tests for RAG Pipeline.
"""

import pytest
from src.models import (
    ADExtractedOutput, 
    ApplicabilityRulesOutput,
    AircraftConfiguration,
    EvaluationResult
)


class TestModels:
    """Test Pydantic models."""
    
    def test_applicability_rules_output(self):
        """Test ApplicabilityRulesOutput model."""
        rules = ApplicabilityRulesOutput(
            aircraft_models=["MD-11", "MD-11F"],
            msn_constraints=None,
            excluded_if_modifications=["SB-XXX"],
            required_modifications=[]
        )
        
        assert len(rules.aircraft_models) == 2
        assert "MD-11" in rules.aircraft_models
        assert rules.msn_constraints is None
        assert "SB-XXX" in rules.excluded_if_modifications
    
    def test_ad_extracted_output(self):
        """Test ADExtractedOutput model."""
        output = ADExtractedOutput(
            ad_id="FAA-2025-23-53",
            applicability_rules=ApplicabilityRulesOutput(
                aircraft_models=["MD-11", "MD-11F"],
                excluded_if_modifications=[]
            )
        )
        
        assert output.ad_id == "FAA-2025-23-53"
        assert len(output.applicability_rules.aircraft_models) == 2
    
    def test_ad_extracted_output_json(self):
        """Test JSON serialization."""
        output = ADExtractedOutput(
            ad_id="FAA-2025-23-53",
            applicability_rules=ApplicabilityRulesOutput(
                aircraft_models=["MD-11"],
                msn_constraints=None,
                excluded_if_modifications=[],
                required_modifications=[]
            )
        )
        
        json_dict = output.model_dump()
        
        assert json_dict["ad_id"] == "FAA-2025-23-53"
        assert json_dict["applicability_rules"]["aircraft_models"] == ["MD-11"]
        assert json_dict["applicability_rules"]["msn_constraints"] is None
    
    def test_aircraft_configuration(self):
        """Test AircraftConfiguration model."""
        aircraft = AircraftConfiguration(
            model="A320-214",
            msn=5234,
            modifications_applied=["mod 24591"]
        )
        
        assert aircraft.model == "A320-214"
        assert aircraft.msn == 5234
        assert "mod 24591" in aircraft.modifications_applied
    
    def test_evaluation_result(self):
        """Test EvaluationResult model."""
        aircraft = AircraftConfiguration(model="MD-11F", msn=48400)
        result = EvaluationResult(
            aircraft=aircraft,
            ad_id="FAA-2025-23-53",
            is_affected=True,
            reason="Aircraft model MD-11F is in applicable models list"
        )
        
        assert result.is_affected is True
        assert result.ad_id == "FAA-2025-23-53"


class TestDocumentLoader:
    """Test document loading and chunking."""
    
    def test_text_chunker(self):
        """Test text chunking."""
        from src.document_loader import TextChunker
        
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Create test text
        text = "A" * 250  # 250 characters
        chunks = chunker._split_text(text)  # Using private method for direct testing
        
        assert len(chunks) >= 2
        assert all(len(c) <= 100 for c in chunks)
    
    def test_ad_detection_faa(self):
        """Test FAA AD detection from text."""
        from src.document_loader import PDFLoader
        
        loader = PDFLoader("data")
        
        # Test FAA AD pattern
        text = "This is FAA AD 2025-23-53 about engine inspection."
        ad_id, authority = loader._detect_ad_info(text, "test.pdf")
        
        assert authority == "FAA"
        assert "2025-23-53" in ad_id
    
    def test_ad_detection_easa(self):
        """Test EASA AD detection from text."""
        from src.document_loader import PDFLoader
        
        loader = PDFLoader("data")
        
        # Test EASA AD pattern
        text = "EASA AD 2025-0254R1 regarding wing inspection."
        ad_id, authority = loader._detect_ad_info(text, "test.pdf")
        
        assert authority == "EASA"
        assert "2025-0254" in ad_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
