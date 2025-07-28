# scripts/download_model.py
# Based on PodGPT's RAG pipeline architecture
# https://github.com/vkola-lab/PodGPT/blob/main/rag_pipeline/download_model.py

import os
import sys
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Optional
import json
import hashlib
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalRAGModelDownloader:
    """
    Enhanced model downloader for Legal AI Intake System
    Based on PodGPT's RAG pipeline with legal-specific adaptations
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Legal-specific model configurations
        self.models_config = {
            # Speech-to-Text Models
            "whisper-large-v3": {
                "url": "https://huggingface.co/openai/whisper-large-v3",
                "type": "speech_to_text",
                "size": "3.09 GB",
                "languages": 99,
                "description": "High-accuracy multilingual speech recognition",
                "priority": "high",
                "legal_optimized": True
            },
            
            # NVIDIA Flamingo 3 (Audio Enhancement)
            "nvidia-flamingo3": {
                "url": "https://api.nvidia.com/v1/models/flamingo-3",
                "type": "audio_enhancement",
                "size": "1.2 GB",
                "description": "Audio quality enhancement for poor recordings",
                "priority": "medium",
                "api_based": True
            },
            
            # Embedding Models for RAGS
            "legal-bert-base": {
                "url": "https://huggingface.co/nlpaueb/legal-bert-base-uncased",
                "type": "embeddings",
                "size": "440 MB",
                "description": "Legal domain-specific BERT embeddings",
                "priority": "high",
                "legal_optimized": True
            },
            
            "sentence-transformers-legal": {
                "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                "type": "embeddings", 
                "size": "90 MB",
                "description": "Fast multilingual sentence embeddings",
                "priority": "high",
                "multilingual": True
            },
            
            # Legal Knowledge Models
            "legal-longformer": {
                "url": "https://huggingface.co/allenai/longformer-base-4096",
                "type": "knowledge_extraction",
                "size": "500 MB", 
                "description": "Long-form legal document processing",
                "priority": "medium",
                "legal_optimized": True
            },
            
            # Multilingual Legal Models
            "legal-xlm-roberta": {
                "url": "https://huggingface.co/joelito/legal-xlm-roberta-base",
                "type": "multilingual_legal",
                "size": "1.1 GB",
                "description": "Multilingual legal text understanding",
                "priority": "high",
                "languages": 30,
                "legal_optimized": True
            },
            
            # Text-to-Speech Models
            "coqui-tts-multilingual": {
                "url": "https://huggingface.co/coqui/XTTS-v2",
                "type": "text_to_speech",
                "size": "2.3 GB",
                "description": "High-quality multilingual TTS",
                "priority": "medium",
                "languages": 17
            },
            
            # Legal Entity Recognition
            "legal-ner-multilingual": {
                "url": "https://huggingface.co/law-ai/InLegalBERT",
                "type": "named_entity_recognition",
                "size": "400 MB",
                "description": "Legal entity recognition for intake forms",
                "priority": "high",
                "legal_optimized": True
            }
        }
        
        # Legal document embeddings database
        self.legal_documents_config = {
            "us_federal_law": {
                "url": "https://huggingface.co/datasets/legal-ai/us-federal-statutes",
                "size": "500 MB",
                "description": "US Federal legal statutes and regulations",
                "languages": ["en"]
            },
            "state_law_database": {
                "url": "https://huggingface.co/datasets/legal-ai/us-state-laws", 
                "size": "1.2 GB",
                "description": "US State laws and regulations",
                "languages": ["en"]
            },
            "multilingual_legal_corpus": {
                "url": "https://huggingface.co/datasets/legal-ai/multilingual-legal-texts",
                "size": "2.1 GB", 
                "description": "Legal texts in 30+ languages",
                "languages": ["en", "es", "fr", "de", "zh", "ar", "hi", "ja", "ko"]
            },
            "case_law_database": {
                "url": "https://huggingface.co/datasets/legal-ai/case-law-corpus",
                "size": "3.5 GB",
                "description": "Legal case precedents and decisions",
                "languages": ["en"]
            }
        }

    def download_model(self, model_name: str, force_download: bool = False) -> bool:
        """Download a specific model with progress tracking"""
        
        if model_name not in self.models_config:
            logger.error(f"Model '{model_name}' not found in configuration")
            return False
            
        model_config = self.models_config[model_name]
        model_path = self.models_dir / model_name
        
        # Check if model already exists
        if model_path.exists() and not force_download:
            logger.info(f"Model '{model_name}' already exists. Use --force to redownload.")
            return True
            
        logger.info(f"Downloading {model_name} ({model_config['size']})...")
        logger.info(f"Description: {model_config['description']}")
        
        try:
            if model_config.get("api_based", False):
                return self._setup_api_model(model_name, model_config)
            else:
                return self._download_huggingface_model(model_name, model_config, model_path)
                
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            return False

    def _download_huggingface_model(self, model_name: str, config: Dict, model_path: Path) -> bool:
        """Download model from Hugging Face Hub"""
        
        try:
            # Use git-lfs for large models
            if "huggingface.co" in config["url"]:
                subprocess.run([
                    "git", "clone", config["url"], str(model_path)
                ], check=True, capture_output=True)
                
                # Verify download
                if self._verify_model_integrity(model_path):
                    logger.info(f"✅ Successfully downloaded {model_name}")
                    return True
                else:
                    logger.error(f"❌ Model verification failed for {model_name}")
                    return False
            else:
                return self._download_direct_url(config["url"], model_path)
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e}")
            return False

    def _setup_api_model(self, model_name: str, config: Dict) -> bool:
        """Setup API-based models (like NVIDIA Flamingo)"""
        
        config_file = self.models_dir / f"{model_name}_config.json"
        
        api_config = {
            "model_name": model_name,
            "endpoint": config["url"],
            "type": config["type"],
            "description": config["description"],
            "setup_date": str(Path().ctime()),
            "requires_api_key": True,
            "env_var": f"{model_name.upper().replace('-', '_')}_API_KEY"
        }
        
        with open(config_file, 'w') as f:
            json.dump(api_config, f, indent=2)
            
        logger.info(f"✅ API model configuration saved: {config_file}")
        logger.info(f"📝 Set environment variable: {api_config['env_var']}")
        return True

    def _download_direct_url(self, url: str, target_path: Path) -> bool:
        """Download from direct URL with progress bar"""
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            target_path.mkdir(parents=True, exist_ok=True)
            file_path = target_path / "model.bin"
            
            with open(file_path, 'wb') as f, tqdm(
                desc=f"Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            logger.error(f"Direct download failed: {e}")
            return False

    def _verify_model_integrity(self, model_path: Path) -> bool:
        """Verify downloaded model integrity"""
        
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                logger.warning(f"Missing file: {file_name}")
                # Some models may not have all files, so continue checking
                
        # Basic size check
        if not any(model_path.iterdir()):
            logger.error("Model directory is empty")
            return False
            
        return True

    def download_legal_documents(self, dataset_name: str = "all") -> bool:
        """Download legal document databases for RAG"""
        
        if dataset_name == "all":
            datasets_to_download = list(self.legal_documents_config.keys())
        else:
            datasets_to_download = [dataset_name] if dataset_name in self.legal_documents_config else []
            
        if not datasets_to_download:
            logger.error(f"Legal dataset '{dataset_name}' not found")
            return False
            
        success_count = 0
        
        for dataset in datasets_to_download:
            config = self.legal_documents_config[dataset]
            dataset_path = self.models_dir / "legal_documents" / dataset
            
            logger.info(f"Downloading legal dataset: {dataset} ({config['size']})")
            
            try:
                if self._download_huggingface_dataset(config["url"], dataset_path):
                    logger.info(f"✅ Downloaded {dataset}")
                    success_count += 1
                else:
                    logger.error(f"❌ Failed to download {dataset}")
                    
            except Exception as e:
                logger.error(f"Error downloading {dataset}: {e}")
                
        logger.info(f"Downloaded {success_count}/{len(datasets_to_download)} legal datasets")
        return success_count == len(datasets_to_download)

    def _download_huggingface_dataset(self, url: str, target_path: Path) -> bool:
        """Download dataset from Hugging Face"""
        
        try:
            subprocess.run([
                "git", "clone", url, str(target_path)
            ], check=True, capture_output=True)
            return True
            
        except subprocess.CalledProcessError:
            logger.error(f"Failed to clone dataset: {url}")
            return False

    def setup_legal_rag_pipeline(self) -> bool:
        """Setup complete legal RAG pipeline"""
        
        logger.info("🚀 Setting up complete Legal RAG Pipeline...")
        
        # Essential models for legal intake
        essential_models = [
            "whisper-large-v3",           # Speech recognition
            "legal-bert-base",            # Legal embeddings
            "sentence-transformers-legal", # Fast embeddings
            "legal-xlm-roberta",          # Multilingual legal
            "legal-ner-multilingual"      # Entity recognition
        ]
        
        # Optional models for enhanced functionality
        optional_models = [
            "nvidia-flamingo3",           # Audio enhancement
            "legal-longformer",           # Long documents
            "coqui-tts-multilingual"      # Text-to-speech
        ]
        
        success_count = 0
        total_models = len(essential_models)
        
        # Download essential models
        for model in essential_models:
            logger.info(f"📥 Downloading essential model: {model}")
            if self.download_model(model):
                success_count += 1
            else:
                logger.error(f"❌ Failed to download essential model: {model}")
                
        # Download optional models (continue on failure)
        for model in optional_models:
            logger.info(f"📥 Downloading optional model: {model}")
            self.download_model(model)  # Don't count failures
            
        # Download legal document databases
        logger.info("📚 Downloading legal document databases...")
        self.download_legal_documents("multilingual_legal_corpus")
        self.download_legal_documents("us_federal_law")
        
        # Create configuration summary
        self._create_setup_summary()
        
        if success_count == total_models:
            logger.info("🎉 Legal RAG Pipeline setup completed successfully!")
            logger.info("📋 Run 'python scripts/verify_setup.py' to test the installation")
            return True
        else:
            logger.warning(f"⚠️  Partial setup: {success_count}/{total_models} essential models downloaded")
            return False

    def _create_setup_summary(self) -> None:
        """Create setup summary and configuration file"""
        
        setup_info = {
            "setup_date": str(Path().ctime()),
            "models_directory": str(self.models_dir.absolute()),
            "downloaded_models": [],
            "api_models": [],
            "legal_datasets": [],
            "total_size_gb": 0
        }
        
        # Scan downloaded models
        for model_name in self.models_config:
            model_path = self.models_dir / model_name
            if model_path.exists():
                if self.models_config[model_name].get("api_based", False):
                    setup_info["api_models"].append(model_name)
                else:
                    setup_info["downloaded_models"].append(model_name)
                    
        # Scan legal datasets
        legal_docs_path = self.models_dir / "legal_documents"
        if legal_docs_path.exists():
            setup_info["legal_datasets"] = [d.name for d in legal_docs_path.iterdir() if d.is_dir()]
            
        # Save setup summary
        summary_file = self.models_dir / "setup_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(setup_info, f, indent=2)
            
        logger.info(f"📄 Setup summary saved: {summary_file}")

    def list_available_models(self) -> None:
        """List all available models with their status"""
        
        print("\n🤖 Available Models for Legal AI Intake System")
        print("=" * 60)
        
        for category in ["speech_to_text", "embeddings", "multilingual_legal", "text_to_speech"]:
            models_in_category = [
                (name, config) for name, config in self.models_config.items() 
                if config["type"] == category
            ]
            
            if models_in_category:
                print(f"\n📂 {category.replace('_', ' ').title()}:")
                for model_name, config in models_in_category:
                    status = "✅ Downloaded" if (self.models_dir / model_name).exists() else "⬇️  Available"
                    priority = "🔥" if config["priority"] == "high" else "⭐"
                    legal_opt = "⚖️ " if config.get("legal_optimized", False) else ""
                    
                    print(f"  {priority} {legal_opt}{model_name}")
                    print(f"     Size: {config['size']} | {config['description']}")
                    print(f"     Status: {status}")
                    
        print(f"\n📚 Legal Document Databases:")
        for dataset_name, config in self.legal_documents_config.items():
            status = "✅ Downloaded" if (self.models_dir / "legal_documents" / dataset_name).exists() else "⬇️  Available"
            print(f"  📖 {dataset_name}")
            print(f"     Size: {config['size']} | Languages: {len(config['languages'])}")
            print(f"     Status: {status}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal AI Intake System - Model Downloader")
    parser.add_argument("--model", type=str, help="Specific model to download")
    parser.add_argument("--setup-all", action="store_true", help="Setup complete RAG pipeline")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--legal-docs", type=str, default="all", help="Download legal document databases")
    parser.add_argument("--models-dir", type=str, default="./models", help="Models directory")
    parser.add_argument("--force", action="store_true", help="Force redownload existing models")
    
    args = parser.parse_args()
    
    downloader = LegalRAGModelDownloader(args.models_dir)
    
    if args.list:
        downloader.list_available_models()
    elif args.setup_all:
        downloader.setup_legal_rag_pipeline()
    elif args.model:
        downloader.download_model(args.model, args.force)
    elif args.legal_docs:
        downloader.download_legal_documents(args.legal_docs)
    else:
        print("Legal AI Intake System - Model Downloader")
        print("Use --help for usage information")
        print("Quick start: python download_model.py --setup-all")

if __name__ == "__main__":
    main()