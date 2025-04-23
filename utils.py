# utils.py
import re
import pandas as pd
import spacy
from typing import Dict, List, Tuple, Any

class PIIMasker: 
    def __init__(self):
        # Load spaCy models for named entity recognition
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_de = spacy.load("de_core_news_sm")
        except:
            print("Warning: spaCy models not loaded. Falling back to regex-only detection.")
            self.nlp_en = None
            self.nlp_de = None
        
        # Compile regex patterns for better performance
        self.patterns = {
            'full_name': r'\b[A-Z][a-zA-Z\'\-]+ [A-Z][a-zA-Z\'\-]+\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_number': r'(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b|\b\d{5}[-\s]?\d{5,6}\b',
            'dob': r'\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4})\b',
            'aadhar_num': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'credit_debit_no': r'\b(?:\d{4}[-\s]?){4}\b',  # Card number pattern
            'cvv_no': r'\bCVV:?\s*\d{3,4}\b|\b[Cc]ode:?\s*\d{3,4}\b|\b\d{3,4}\b(?=\s*(?:security\s*code|verification))',
            'expiry_no': r'\b(?:0[1-9]|1[0-2])[\/\-](?:[0-9]{2}|[0-9]{4})\b|\bexp(?:ires|iry)?:?\s*(?:0[1-9]|1[0-2])[\/\-](?:[0-9]{2}|[0-9]{4})\b'
        }
        
        # Compile all patterns for performance
        self.compiled_patterns = {
            entity: re.compile(pattern) 
            for entity, pattern in self.patterns.items()
        }
        
        # Store original entities for unmasking
        self.entity_store = {}
        self.entity_counter = 0
    
    def detect_language(self, text: str) -> str:
        """Detect if text is English or German."""
        # Simple heuristic: Check for common German words
        german_words = ['der', 'die', 'das', 'und', 'ist', 'für', 'nicht', 'sie', 'mit', 'auf']
        text_lower = text.lower()
        german_count = sum(1 for word in german_words if f" {word} " in f" {text_lower} ")
        
        # If more than 2 German words are found, classify as German
        return 'de' if german_count > 2 else 'en'
    
    def find_entities(self, text: str) -> List[Dict[str, Any]]:
        """Find all PII entities in the text using regex and spaCy."""
        entities = []
        
        # 1. Use regex patterns
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                entities.append({
                    "position": [start, end],
                    "classification": entity_type,
                    "entity": text[start:end]
                })
        
        # 2. Use spaCy if available
        if self.nlp_en is not None:
            # Detect language
            lang = self.detect_language(text)
            nlp = self.nlp_en if lang == 'en' else self.nlp_de
            
            # Process with spaCy
            doc = nlp(text)
            
            # Map spaCy entities to our entity types
            entity_mapping = {
                'PERSON': 'full_name',
                'ORG': None,  # Skip organizations
                'GPE': None,  # Skip geopolitical entities
                'DATE': None,  # We use regex for dates of birth
                'CARDINAL': None,  # Skip numbers
            }
            
            for ent in doc.ents:
                mapped_type = entity_mapping.get(ent.label_)
                if mapped_type:
                    start, end = ent.start_char, ent.end_char
                    
                    # Check for overlaps with regex findings
                    overlap = False
                    for existing in entities:
                        e_start, e_end = existing["position"]
                        if (start <= e_end and end >= e_start):
                            overlap = True
                            break
                    
                    if not overlap:
                        entities.append({
                            "position": [start, end],
                            "classification": mapped_type,
                            "entity": text[start:end]
                        })
        
        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x["position"][0])
        
        # Remove duplicate detections
        filtered_entities = []
        for entity in entities:
            duplicate = False
            for existing in filtered_entities:
                if entity["position"] == existing["position"]:
                    duplicate = True
                    break
            if not duplicate:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def mask_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Mask PII entities in the text.
        
        Args:
            text: The input text to mask
            
        Returns:
            Tuple of (masked_text, list_of_entities)
        """
        entities = self.find_entities(text)
        masked_text = text
        
        # Clear previous entity store for this text
        self.entity_store = {}
        
        # Apply masking from end to beginning to avoid index shifting
        for entity in sorted(entities, key=lambda x: x["position"][0], reverse=True):
            start, end = entity["position"]
            entity_type = entity["classification"]
            original_value = entity["entity"]
            
            # Store for unmasking later
            entity_id = self.entity_counter
            self.entity_counter += 1
            self.entity_store[entity_id] = {
                "type": entity_type,
                "value": original_value
            }
            
            # Replace with mask
            masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
        
        return masked_text, entities
    
    def unmask_text(self, masked_text: str) -> str:
        """Restore original values from masked text.
        
        This is a placeholder as the assignment doesn't specify unmasking logic.
        In a real implementation, we would need a way to track masked entities.
        """
        return masked_text  # In this implementation, we don't unmask

def detect_language(text: str) -> str:
    """Detect if text is primarily English or German."""
    # Simple heuristic based on common German words
    german_words = ['der', 'die', 'das', 'und', 'ist', 'für', 'nicht', 'sie', 'mit', 'auf']
    text_lower = text.lower()
    german_count = sum(1 for word in german_words if f" {word} " in f" {text_lower} ")
    
    # If more than 2 German words are found, classify as German
    return 'de' if german_count > 2 else 'en'

def clean_email(text: str) -> str:
    """Clean email text for classification."""
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase (preserve German umlauts)
    text = text.lower().strip()
    
    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset for training."""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Ensure columns exist
    if 'email' not in df_processed.columns or 'type' not in df_processed.columns:
        raise ValueError("DataFrame must contain 'email' and 'type' columns")
    
    # Clean emails
    df_processed['cleaned_email'] = df_processed['email'].apply(clean_email)
    
    # Detect language for each email
    df_processed['language'] = df_processed['email'].apply(detect_language)
    
    # Handle any missing values
    df_processed = df_processed.dropna(subset=['email', 'type'])
    
    return df_processed