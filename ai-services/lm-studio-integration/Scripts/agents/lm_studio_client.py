
import os
import json
import logging
import requests
from typing import Dict, List, Union, Optional
import hashlib
import time

class LMStudioClient:
    """Client for interacting with LM Studio local API for AI completions and embeddings"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('LM_STUDIO_URL', 'http://localhost:1234')
        self.logger = logging.getLogger('LMStudioClient')
        self.session = requests.Session()
        self.models = self._get_available_models()
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return [model["id"] for model in response.json()["data"]]
        except Exception as e:
            self.logger.warning(f"Failed to get models: {str(e)}")
            return []
    
    def generate_completion(
        self, 
        prompt: str, 
        model: str = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Dict:
        """Generate a completion using LM Studio API"""
        if not model and self.models:
            model = self.models[0]
        elif not model:
            model = "gemma-3-12b-it"  # Default fallback
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Completion generation failed: {str(e)}")
            raise
    
    def stream_completion(
        self, 
        prompt: str, 
        model: str = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """Stream a completion using LM Studio API"""
        if not model and self.models:
            model = self.models[0]
        elif not model:
            model = "gemma-3-12b-it"  # Default fallback
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data:'):
                        data = line[5:].strip()
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to decode JSON: {data}")
                            continue
        except Exception as e:
            self.logger.error(f"Stream completion failed: {str(e)}")
            raise
    
    def generate_embedding(
        self, 
        text: str,
        model: str = "text-embedding-ada-002"  # Default embedding model
    ) -> List[float]:
        """Generate embeddings for text using LM Studio API"""
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def batch_generate_embeddings(
        self, 
        texts: List[str],
        model: str = "text-embedding-ada-002"  # Default embedding model
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts using LM Studio API"""
        payload = {
            "model": model,
            "input": texts
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {str(e)}")
            raise
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        model: str = None
    ) -> Dict:
        """Analyze image using multimodal capabilities of LM Studio"""
        if not model and self.models:
            model = self.models[0]
        elif not model:
            model = "llava-1.5-7b"  # Default multimodal model
        
        # Encode image to base64
        import base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Image analysis failed: {str(e)}")
            raise
    
    def classify_document(
        self,
        text: str,
        categories: List[str],
        model: str = None
    ) -> Dict[str, float]:
        """Classify document into predefined categories"""
        categories_str = ", ".join(categories)
        prompt = f"""Classify the following text into one of these categories: {categories_str}.
        Respond with only the category name and confidence score (0-1), formatted as "category:confidence".
        
        Text: {text}
        """
        
        result = self.generate_completion(prompt, model)
        try:
            content = result["choices"][0]["message"]["content"]
            category, confidence = content.strip().split(":")
            return {
                "category": category.strip(),
                "confidence": float(confidence.strip()),
                "raw_response": content
            }
        except Exception as e:
            self.logger.error(f"Classification parsing failed: {str(e)}")
            return {
                "category": None,
                "confidence": 0.0,
                "raw_response": result["choices"][0]["message"]["content"]
            }
    
    def generate_dewey_classification(
        self,
        title: str,
        content: str,
        model: str = None
    ) -> str:
        """Generate Dewey Decimal classification for content"""
        prompt = f"""Generate a Dewey Decimal classification number for the following content.
        Respond with only the decimal number (e.g., 025.04).
        
        Title: {title}
        Content: {content[:1000]}...
        """
        
        result = self.generate_completion(prompt, model)
        try:
            content = result["choices"][0]["message"]["content"]
            # Extract just the dewey decimal number using regex
            import re
            match = re.search(r'\d{3}(?:\.\d+)?', content)
            if match:
                return match.group(0)
            return content.strip()
        except Exception as e:
            self.logger.error(f"Dewey classification failed: {str(e)}")
            return "000.000"  # Default classification

