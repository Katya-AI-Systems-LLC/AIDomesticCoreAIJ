"""
Named Entity Recognition
========================

Extract named entities from text.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    CRYPTO_ADDRESS = "CRYPTO"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"


@dataclass
class Entity:
    """Named entity."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = None


class NERExtractor:
    """
    Named Entity Recognition extractor.
    
    Features:
    - Multiple entity types
    - Pattern-based extraction
    - ML-based extraction
    - Custom entity types
    - Confidence scoring
    
    Example:
        >>> ner = NERExtractor()
        >>> entities = ner.extract("John works at Apple in Cupertino")
    """
    
    # Regex patterns
    PATTERNS = {
        EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        EntityType.PHONE: r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
        EntityType.URL: r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w.-]*',
        EntityType.DATE: r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
        EntityType.TIME: r'\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?[AaPp][Mm])?\b',
        EntityType.MONEY: r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|BTC|ETH)\b',
        EntityType.PERCENT: r'\b\d+(?:\.\d+)?%\b',
        EntityType.CRYPTO_ADDRESS: r'\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59})\b'
    }
    
    def __init__(self, use_ml: bool = False,
                 model_name: str = None):
        """
        Initialize NER extractor.
        
        Args:
            use_ml: Use ML-based extraction
            model_name: Model name for ML extraction
        """
        self.use_ml = use_ml
        self.model_name = model_name
        
        self._ml_model = None
        
        if use_ml:
            self._load_ml_model()
        
        logger.info(f"NER Extractor initialized (ML={use_ml})")
    
    def _load_ml_model(self):
        """Load ML model."""
        try:
            import spacy
            self._ml_model = spacy.load(self.model_name or "en_core_web_sm")
            logger.info("SpaCy model loaded")
        except ImportError:
            logger.warning("SpaCy not installed, using pattern-based extraction")
            self.use_ml = False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.use_ml = False
    
    def extract(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        # Pattern-based extraction
        entities.extend(self._extract_patterns(text))
        
        # ML-based extraction
        if self.use_ml and self._ml_model:
            entities.extend(self._extract_ml(text))
        else:
            # Fallback heuristic extraction
            entities.extend(self._extract_heuristic(text))
        
        # Deduplicate
        entities = self._deduplicate(entities)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        return entities
    
    def _extract_patterns(self, text: str) -> List[Entity]:
        """Extract using regex patterns."""
        entities = []
        
        for entity_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0
                ))
        
        return entities
    
    def _extract_ml(self, text: str) -> List[Entity]:
        """Extract using ML model."""
        entities = []
        
        doc = self._ml_model(text)
        
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT
        }
        
        for ent in doc.ents:
            entity_type = type_mapping.get(ent.label_, None)
            if entity_type:
                entities.append(Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_heuristic(self, text: str) -> List[Entity]:
        """Heuristic extraction for names and organizations."""
        entities = []
        
        # Simple capitalized word sequences (potential names/orgs)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        
        # Common organization indicators
        org_indicators = ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Foundation', 
                         'University', 'Institute', 'Bank', 'Group']
        
        for match in re.finditer(cap_pattern, text):
            word = match.group(1)
            
            # Skip common words
            if word.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were']:
                continue
            
            # Determine type
            if any(ind in word for ind in org_indicators):
                entity_type = EntityType.ORGANIZATION
            elif len(word.split()) >= 2:
                entity_type = EntityType.PERSON
            else:
                continue
            
            entities.append(Entity(
                text=word,
                entity_type=entity_type,
                start=match.start(),
                end=match.end(),
                confidence=0.6
            ))
        
        return entities
    
    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity.text, entity.entity_type, entity.start)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique
    
    def extract_type(self, text: str,
                     entity_type: EntityType) -> List[Entity]:
        """Extract specific entity type."""
        entities = self.extract(text)
        return [e for e in entities if e.entity_type == entity_type]
    
    def anonymize(self, text: str,
                  types: List[EntityType] = None) -> str:
        """
        Anonymize entities in text.
        
        Args:
            text: Input text
            types: Entity types to anonymize (None = all)
            
        Returns:
            Anonymized text
        """
        entities = self.extract(text)
        
        if types:
            entities = [e for e in entities if e.entity_type in types]
        
        # Sort by position (reverse to maintain offsets)
        entities.sort(key=lambda e: e.start, reverse=True)
        
        result = text
        for entity in entities:
            placeholder = f"[{entity.entity_type.value}]"
            result = result[:entity.start] + placeholder + result[entity.end:]
        
        return result
    
    def __repr__(self) -> str:
        return f"NERExtractor(ml={self.use_ml})"
