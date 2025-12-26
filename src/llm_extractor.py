"""
LLM-based Rule Extractor using NVIDIA API.

This module uses LLM to extract structured applicability rules
from AD document chunks retrieved from the vector store.
"""

import json
from typing import List, Dict, Optional
from openai import OpenAI

from .config import NVIDIA_API_KEY, NVIDIA_BASE_URL, LLM_MODEL
from .models import ADExtractedOutput, ApplicabilityRulesOutput
from .vector_store import VectorStore


class LLMExtractor:
    """
    Extracts structured AD rules using LLM.
    """
    
    EXTRACTION_PROMPT = """You are an aviation safety expert. Extract the applicability rules from the following Airworthiness Directive (AD) document.

Return ONLY a valid JSON object with this exact structure:
{{
    "ad_id": "string (e.g., FAA-2025-23-53 or EASA-2025-0254R1)",
    "applicability_rules": {{
        "aircraft_models": ["list of applicable aircraft model designations"],
        "msn_constraints": null,
        "excluded_if_modifications": ["list of modifications that EXCLUDE aircraft if already applied"],
        "required_modifications": []
    }}
}}

Important rules:
1. For aircraft_models, list ALL specific model designations mentioned (e.g., "MD-11", "MD-11F", "A320-214", "A320-232")
2. For excluded_if_modifications, extract modification numbers (e.g., "mod 24591") and service bulletins (e.g., "SB A320-57-1089") that EXCLUDE aircraft when already applied
3. If AD says "except those on which mod XXXXX has been embodied", add that mod to excluded_if_modifications
4. msn_constraints should be null if AD applies to "all manufacturer serial numbers"
5. Return ONLY the JSON object, no explanation or markdown

AD Document Content:
{context}

Extract the applicability rules:"""

    def __init__(
        self,
        api_key: str = NVIDIA_API_KEY,
        base_url: str = NVIDIA_BASE_URL,
        model: str = LLM_MODEL
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    def extract_from_text(self, text: str, ad_id_hint: Optional[str] = None) -> ADExtractedOutput:
        """
        Extract AD rules from text using LLM.
        
        Args:
            text: Document text containing AD information
            ad_id_hint: Optional hint for AD ID
            
        Returns:
            Structured ADExtractedOutput
        """
        prompt = self.EXTRACTION_PROMPT.format(context=text[:8000])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an aviation safety expert. Extract information precisely and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            
            return ADExtractedOutput(
                ad_id=data.get("ad_id", ad_id_hint or "UNKNOWN"),
                applicability_rules=ApplicabilityRulesOutput(
                    aircraft_models=data.get("applicability_rules", {}).get("aircraft_models", []),
                    msn_constraints=data.get("applicability_rules", {}).get("msn_constraints"),
                    excluded_if_modifications=data.get("applicability_rules", {}).get("excluded_if_modifications", []),
                    required_modifications=data.get("applicability_rules", {}).get("required_modifications", [])
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return ADExtractedOutput(
                ad_id=ad_id_hint or "UNKNOWN",
                applicability_rules=ApplicabilityRulesOutput()
            )
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text that might contain other content."""
        # Try to find JSON object in the text
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in response")
        
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        
        raise ValueError("Incomplete JSON object")
    
    def extract_from_vector_store(
        self, 
        vector_store: VectorStore, 
        ad_id: str
    ) -> ADExtractedOutput:
        """
        Extract AD rules using documents from vector store.
        
        Args:
            vector_store: Vector store containing AD documents
            ad_id: AD ID to extract rules for
            
        Returns:
            Structured ADExtractedOutput
        """
        # Get all documents for this AD
        docs = vector_store.get_all_documents(ad_id=ad_id)
        
        if not docs:
            print(f"No documents found for AD: {ad_id}")
            return ADExtractedOutput(
                ad_id=ad_id,
                applicability_rules=ApplicabilityRulesOutput()
            )
        
        # Combine relevant documents
        combined_text = "\n\n".join([doc["content"] for doc in docs])
        
        # Extract using LLM
        return self.extract_from_text(combined_text, ad_id_hint=ad_id)
    
    def extract_with_rag(
        self,
        vector_store: VectorStore,
        ad_id: str,
        query: str = "applicability aircraft models modifications exclusions"
    ) -> ADExtractedOutput:
        """
        Extract AD rules using RAG approach.
        
        Args:
            vector_store: Vector store
            ad_id: AD ID
            query: Query for retrieving relevant chunks
            
        Returns:
            Structured ADExtractedOutput
        """
        # Query for relevant chunks
        results = vector_store.query_by_ad(
            query_text=query,
            ad_id=ad_id,
            n_results=10
        )
        
        if not results:
            return self.extract_from_vector_store(vector_store, ad_id)
        
        # Combine retrieved chunks
        context = "\n\n".join([r["content"] for r in results])
        
        return self.extract_from_text(context, ad_id_hint=ad_id)
    
    def ask_question(
        self,
        vector_store: VectorStore,
        question: str,
        ad_id: Optional[str] = None,
        n_results: int = 5
    ) -> str:
        """
        Answer a question about AD documents using RAG.
        
        Args:
            vector_store: Vector store containing AD documents
            question: User's question
            ad_id: Optional AD ID to filter results
            n_results: Number of chunks to retrieve
            
        Returns:
            Answer string from LLM
        """
        # Retrieve relevant chunks
        if ad_id:
            results = vector_store.query_by_ad(
                query_text=question,
                ad_id=ad_id,
                n_results=n_results
            )
        else:
            results = vector_store.query(
                query_text=question,
                n_results=n_results
            )
        
        if not results:
            return "Sorry, no relevant information was found to answer your question."
        
        # Build context from retrieved chunks
        context_parts = []
        sources = set()
        for r in results:
            context_parts.append(r["content"])
            if "ad_id" in r.get("metadata", {}):
                sources.add(r["metadata"]["ad_id"])
        
        context = "\n\n---\n\n".join(context_parts)
        sources_str = ", ".join(sources) if sources else "unknown"
        
        # Create Q&A prompt
        qa_prompt = f"""You are an aviation safety expert assistant. Answer the following question based ONLY on the provided context from Airworthiness Directive (AD) documents.

Context from AD documents:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the information in the context above
2. If the answer cannot be found in the context, say "Information not found in the available documents"
3. Be specific and cite relevant details from the ADs
4. Always answer in English
5. If asked about applicability, list specific aircraft models
6. If asked about exclusions or modifications, list specific mod numbers or service bulletins

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an aviation safety expert. Provide accurate and helpful answers about Airworthiness Directives."},
                {"role": "user", "content": qa_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Append sources
        answer += f"\n\nSource: {sources_str}"
        
        return answer


def get_llm_extractor() -> LLMExtractor:
    """Factory function to create LLM extractor."""
    return LLMExtractor()
