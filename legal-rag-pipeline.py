# rag_pipeline/legal_rag_pipeline.py
# Enhanced RAG pipeline for Legal AI Intake System
# Based on PodGPT's architecture with legal-specific optimizations

import os
import json
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

# Core ML libraries
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    WhisperProcessor, WhisperForConditionalGeneration,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Legal-specific imports
from legal_entity_extractor import LegalEntityExtractor
from multilingual_processor import MultilingualProcessor
from audio_quality_analyzer import AudioQualityAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    """Legal document representation"""
    content: str
    metadata: Dict[str, Any]
    jurisdiction: str
    legal_domain: str
    language: str
    embedding: Optional[np.ndarray] = None
    
@dataclass
class RAGResponse:
    """RAG pipeline response"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    legal_context: List[str]
    follow_up_questions: List[str]
    processing_metadata: Dict[str, Any]

class LegalRAGPipeline:
    """
    Legal-optimized RAG pipeline based on PodGPT architecture
    Handles multilingual legal intake with voice processing
    """
    
    def __init__(self, models_dir: str = "./models", config_path: str = "./config/rag_config.json"):
        self.models_dir = Path(models_dir)
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.embeddings_model = None
        self.legal_bert_model = None
        self.whisper_model = None
        self.vector_store = None
        self.legal_entity_extractor = None
        self.multilingual_processor = None
        self.audio_analyzer = None
        
        # Legal knowledge base
        self.legal_documents: List[LegalDocument] = []
        self.legal_ontology = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "avg_response_time": 0,
            "accuracy_scores": [],
            "cost_tracking": {"whisper": 0, "openai": 0, "nvidia": 0}
        }
        
        logger.info("Initializing Legal RAG Pipeline...")
        self._initialize_models()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for legal RAG pipeline"""
        return {
            "embeddings": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "legal_model_name": "nlpaueb/legal-bert-base-uncased",
                "dimension": 384,
                "batch_size": 32
            },
            "whisper": {
                "model_size": "large-v3",
                "language": "multilingual",
                "quality_threshold": 0.7
            },
            "legal_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_relevance_score": 0.3,
                "max_retrieved_docs": 5
            },
            "languages": {
                "supported": ["en", "es", "fr", "de", "zh", "ar", "hi", "ja", "ko", "pt", "it", "ru", "tr"],
                "default": "en"
            },
            "nvidia_flamingo": {
                "enabled": True,
                "quality_fallback_threshold": 0.5,
                "cost_optimization": True
            }
        }
    
    def _initialize_models(self):
        """Initialize all models and components"""
        try:
            # 1. Initialize embedding models
            self._load_embedding_models()
            
            # 2. Initialize speech processing
            self._load_speech_models()
            
            # 3. Initialize legal-specific components
            self._load_legal_components()
            
            # 4. Load legal document database
            self._load_legal_database()
            
            # 5. Initialize vector store
            self._initialize_vector_store()
            
            logger.info("✅ All models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def _load_embedding_models(self):
        """Load embedding models for semantic search"""
        
        # General purpose embeddings
        embeddings_model_path = self.models_dir / "sentence-transformers-legal"
        if embeddings_model_path.exists():
            self.embeddings_model = SentenceTransformer(str(embeddings_model_path))
            logger.info("✅ Loaded general embeddings model")
        else:
            # Fallback to online model
            self.embeddings_model = SentenceTransformer(self.config["embeddings"]["model_name"])
            logger.info("⬇️  Using online embeddings model")
        
        # Legal-specific BERT
        legal_bert_path = self.models_dir / "legal-bert-base"
        if legal_bert_path.exists():
            self.legal_bert_tokenizer = AutoTokenizer.from_pretrained(str(legal_bert_path))
            self.legal_bert_model = AutoModel.from_pretrained(str(legal_bert_path))
            logger.info("✅ Loaded legal BERT model")
        else:
            logger.warning("⚠️  Legal BERT model not found, using general embeddings")
    
    def _load_speech_models(self):
        """Load speech processing models"""
        
        # Whisper for high-quality audio
        whisper_path = self.models_dir / "whisper-large-v3"
        if whisper_path.exists():
            self.whisper_processor = WhisperProcessor.from_pretrained(str(whisper_path))
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                str(whisper_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.whisper_model = self.whisper_model.cuda()
            logger.info("✅ Loaded Whisper model")
        else:
            logger.warning("⚠️  Whisper model not found, will use API fallback")
        
        # Audio quality analyzer
        self.audio_analyzer = AudioQualityAnalyzer()
    
    def _load_legal_components(self):
        """Load legal-specific processing components"""
        
        # Legal entity extractor
        self.legal_entity_extractor = LegalEntityExtractor(
            model_path=self.models_dir / "legal-ner-multilingual"
        )
        
        # Multilingual processor
        self.multilingual_processor = MultilingualProcessor(
            supported_languages=self.config["languages"]["supported"]
        )
        
        # Load legal ontology
        ontology_path = self.models_dir / "legal_ontology.json"
        if ontology_path.exists():
            with open(ontology_path, 'r') as f:
                self.legal_ontology = json.load(f)
                logger.info("✅ Loaded legal ontology")
    
    def _load_legal_database(self):
        """Load legal document database"""
        
        legal_docs_dir = self.models_dir / "legal_documents"
        if not legal_docs_dir.exists():
            logger.warning("⚠️  Legal documents directory not found")
            return
        
        document_count = 0
        
        # Load different legal document types
        for doc_type_dir in legal_docs_dir.iterdir():
            if doc_type_dir.is_dir():
                logger.info(f"Loading legal documents from: {doc_type_dir.name}")
                
                # Process JSON files in each directory
                for json_file in doc_type_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            documents = json.load(f)
                            
                        for doc_data in documents:
                            legal_doc = LegalDocument(
                                content=doc_data.get("content", ""),
                                metadata=doc_data.get("metadata", {}),
                                jurisdiction=doc_data.get("jurisdiction", "federal"),
                                legal_domain=doc_data.get("legal_domain", "general"),
                                language=doc_data.get("language", "en")
                            )
                            
                            # Generate embedding
                            legal_doc.embedding = self._generate_embedding(legal_doc.content)
                            self.legal_documents.append(legal_doc)
                            document_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"✅ Loaded {document_count} legal documents")
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store with legal documents"""
        
        if not self.legal_documents:
            logger.warning("⚠️  No legal documents loaded, skipping vector store initialization")
            return
        
        # Prepare embeddings and metadata
        embeddings = np.array([doc.embedding for doc in self.legal_documents if doc.embedding is not None])
        
        if len(embeddings) == 0:
            logger.error("❌ No valid embeddings found")
            return
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings)
        
        logger.info(f"✅ Initialized FAISS vector store with {len(embeddings)} embeddings")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the best available model"""
        
        if self.legal_bert_model and len(text) < 512:  # Legal BERT for short texts
            return self._generate_legal_bert_embedding(text)
        elif self.embeddings_model:  # General embeddings for longer texts
            return self.embeddings_model.encode(text, normalize_embeddings=True)
        else:
            logger.error("No embedding model available")
            return np.zeros(384)  # Fallback zero vector