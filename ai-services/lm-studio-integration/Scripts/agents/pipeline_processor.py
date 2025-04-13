
import os
import sys
import json
import logging
import hashlib
import time
from datetime import datetime
import mimetypes
import uuid
import re
from typing import Dict, List, Optional, Tuple, Union, Any

# Import the LM Studio client
from lm_studio_client import LMStudioClient

# Database connections
import redis
from pymongo import MongoClient
from neo4j import GraphDatabase
from meilisearch import Client as MeiliSearchClient
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import psycopg2
from psycopg2.extras import Json as PgJson
from minio import Minio

# File processing
import magic
import fitz  # PyMuPDF
import textract
import cv2
import numpy as np
from PIL import Image
import imagehash


class PipelineProcessor:
    """AI processing pipeline for raw data ingestion and analysis."""
    
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.setup_connections()
        self.lm_client = LMStudioClient(os.getenv('LM_STUDIO_URL'))
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = "/app/logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/pipeline.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('PipelineProcessor')
            # Video - return None, will use specialized video processing
            elif mime_type.startswith('video/'):
                return None
                
            # Default case - try textract
            else:
                try:
                    return textract.process(file_path).decode('utf-8', errors='ignore')
                except:
                    self.logger.warning(f"Could not extract text from {file_path}")
                    return None
        except Exception as e:
            self.logger.error(f"Error extracting content: {str(e)}")
            return None
    
    def generate_embeddings(self, content: str, mime_type: str) -> Dict:
        """Generate embeddings for content"""
        try:
            # Select embedding model based on content type
            if mime_type.startswith('image/'):
                model = self.config['embedding_models'].get('image', 'clip-vit-base-patch32')
            else:
                model = self.config['embedding_models'].get('text', 'text-embedding-ada-002')
            
            # Check if embedding exists in cache
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            cache_key = f"embedding:{model}:{content_hash}"
            
            cached_embedding = self.redis.get(cache_key)
            if cached_embedding:
                self.logger.info(f"Using cached embedding for {content_hash}")
                return json.loads(cached_embedding)
            
            # Generate new embedding
            if len(content) > 8000:
                # For long content, we'll chunk it
                chunks = [content[i:i+8000] for i in range(0, len(content), 8000)]
                # Get embeddings for each chunk
                embeddings = []
                for chunk in chunks:
                    embedding = self.lm_client.generate_embedding(chunk, model)
                    embeddings.append(embedding)
                # Average the embeddings
                embedding_result = {
                    "model": model,
                    "embedding": self._average_embeddings(embeddings),
                    "chunks": len(chunks)
                }
            else:
                # For short content, get a single embedding
                embedding = self.lm_client.generate_embedding(content, model)
                embedding_result = {
                    "model": model,
                    "embedding": embedding,
                    "chunks": 1
                }
            
            # Cache the result
            self.redis.setex(cache_key, 3600, json.dumps(embedding_result))
            
            return embedding_result
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return {"model": "none", "embedding": [], "error": str(e)}
    
    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Average multiple embeddings into a single vector"""
        if not embeddings:
            return []
        
        # Get embedding dimension
        dim = len(embeddings[0])
        
        # Sum embeddings
        avg_embedding = [0.0] * dim
        for emb in embeddings:
            for i in range(dim):
                avg_embedding[i] += emb[i]
        
        # Divide by count
        for i in range(dim):
            avg_embedding[i] /= len(embeddings)
        
        # Normalize
        norm = sum(x**2 for x in avg_embedding) ** 0.5
        return [x / norm for x in avg_embedding]
    
    def analyze_content(self, content: str, mime_type: str, file_path: str) -> Dict:
        """Analyze content using LM Studio AI"""
        try:
            # Initialize results
            result = {
                "tags": [],
                "categories": [],
                "summary": "",
                "entities": [],
                "sentiment": {},
                "language": "",
                "title": ""
            }
            
            # Text-based analysis
            if content and (mime_type.startswith('text/') or 
                          mime_type in ['application/pdf', 'application/msword',
                                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document']):
                
                # Generate title
                title_prompt = f"Create a concise, descriptive title for this document in 10 words or less: {content[:1000]}..."
                title_result = self.lm_client.generate_completion(title_prompt)
                result["title"] = title_result["choices"][0]["message"]["content"].strip()
                
                # Generate summary
                summary_prompt = f"Summarize the following text in 2-3 sentences: {content[:2000]}..."
                summary_result = self.lm_client.generate_completion(summary_prompt)
                result["summary"] = summary_result["choices"][0]["message"]["content"].strip()
                
                # Generate tags
                tags_prompt = f"Generate 5-10 relevant tags for this content, separated by commas: {content[:2000]}..."
                tags_result = self.lm_client.generate_completion(tags_prompt)
                tags = tags_result["choices"][0]["message"]["content"].strip().split(',')
                result["tags"] = [tag.strip() for tag in tags if tag.strip()]
                
                # Categorize content
                categories = self.config.get("categories", [])
                if categories:
                    category_result = self.lm_client.classify_document(content[:2000], categories)
                    result["categories"] = [category_result["category"]]
                    result["category_confidence"] = category_result["confidence"]
                
                # Extract entities
                entities_prompt = f"Extract important entities (people, organizations, locations) from this text as a JSON array: {content[:2000]}..."
                entities_result = self.lm_client.generate_completion(entities_prompt)
                try:
                    entities_text = entities_result["choices"][0]["message"]["content"].strip()
                    # Extract JSON array from response
                    import re
                    json_match = re.search(r'\[.*\]', entities_text, re.DOTALL)
                    if json_match:
                        entities = json.loads(json_match.group(0))
                        result["entities"] = entities
                except:
                    self.logger.warning("Could not parse entities JSON")
                
                # Detect language
                language_prompt = f"Identify the language of this text, respond with just the ISO language code: {content[:500]}..."
                language_result = self.lm_client.generate_completion(language_prompt)
                result["language"] = language_result["choices"][0]["message"]["content"].strip()
                
            # Image-based analysis
            elif mime_type.startswith('image/'):
                # Analyze image
                image_prompt = "Describe this image in detail, including objects, people, activities, location, and mood."
                image_result = self.lm_client.analyze_image(file_path, image_prompt)
                result["description"] = image_result["choices"][0]["message"]["content"].strip()
                
                # Generate tags from image
                tags_prompt = "Generate 5-10 relevant tags for this image, separated by commas."
                tags_result = self.lm_client.analyze_image(file_path, tags_prompt)
                tags = tags_result["choices"][0]["message"]["content"].strip().split(',')
                result["tags"] = [tag.strip() for tag in tags if tag.strip()]
                
                # Detect objects
                objects_prompt = "List the main objects visible in this image as a comma-separated list."
                objects_result = self.lm_client.analyze_image(file_path, objects_prompt)
                objects = objects_result["choices"][0]["message"]["content"].strip().split(',')
                result["objects"] = [obj.strip() for obj in objects if obj.strip()]
                
                # Generate image hash for similarity detection
                try:
                    img = Image.open(file_path)
                    result["phash"] = str(imagehash.phash(img))
                    result["width"] = img.width
                    result["height"] = img.height
                except:
                    self.logger.warning(f"Could not generate image hash for {file_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            return {"error": str(e)}
    
    def generate_dewey_classification(self, title: str, content: str, tags: List[str]) -> str:
        """Generate Dewey Decimal classification for content"""
        try:
            # Prepare content for classification
            tags_str = ", ".join(tags)
            prompt_content = content[:1500] if content else ""
            
            # Create prompt for LM Studio
            prompt = f"""Generate a Dewey Decimal classification number for the following content.
            Respond with only the decimal number (e.g., 025.04).
            
            Title: {title}
            Tags: {tags_str}
            Content: {prompt_content}
            
            The Dewey Decimal ranges are:
            000-099: Computer science, information & general works
            100-199: Philosophy & psychology
            200-299: Religion
            300-399: Social sciences
            400-499: Language
            500-599: Science
            600-699: Technology
            700-799: Arts & recreation
            800-899: Literature
            900-999: History & geography
            
            Choose the most specific classification possible using 3 digits and up to 2 decimal places.
            """
            
            # Check cache
            cache_key = f"dewey:{hashlib.md5(prompt.encode('utf-8')).hexdigest()}"
            cached_dewey = self.redis.get(cache_key)
            if cached_dewey:
                return cached_dewey.decode('utf-8')
            
            # Generate classification
            result = self.lm_client.generate_completion(prompt)
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse decimal number with regex
            import re
            match = re.search(r'\d{3}(?:\.\d+)?', content)
            if match:
                dewey = match.group(0)
            else:
                dewey = "000.00"  # Default classification
            
            # Cache the result
            self.redis.setex(cache_key, 86400, dewey)  # Cache for 24 hours
            
            return dewey
            
        except Exception as e:
            self.logger.error(f"Error generating Dewey classification: {str(e)}")
            return "000.00"  # Default classification
    
    def record_processing_start(self, file_id: str, file_path: str, priority: str):
        """Record the start of processing in InfluxDB"""
        try:
            point = Point("file_processing") \
                .tag("file_id", file_id) \
                .tag("priority", priority) \
                .tag("file_name", os.path.basename(file_path)) \
                .tag("status", "started") \
                .field("path", file_path) \
                .field("size", os.path.getsize(file_path)) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket="aimetrics", org="aiorg", record=point)
            
        except Exception as e:
            self.logger.error(f"Error recording processing start: {str(e)}")
    
    def record_processing_complete(self, file_id: str, metadata: Dict):
        """Record the completion of processing in InfluxDB"""
        try:
            point = Point("file_processing") \
                .tag("file_id", file_id) \
                .tag("priority", metadata.get('priority', 'unknown')) \
                .tag("file_name", metadata.get('file_name', 'unknown')) \
                .tag("status", "completed") \
                .tag("dewey_decimal", metadata.get('dewey_decimal', '000.00')) \
                .field("processing_time_ms", (datetime.utcnow() - metadata.get('created_at', datetime.utcnow())).total_seconds() * 1000) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket="aimetrics", org="aiorg", record=point)
            
        except Exception as e:
            self.logger.error(f"Error recording processing completion: {str(e)}")
    
    def record_processing_error(self, file_id: str, file_path: str, error: str):
        """Record processing error in InfluxDB"""
        try:
            point = Point("file_processing") \
                .tag("file_id", file_id) \
                .tag("file_name", os.path.basename(file_path)) \
                .tag("status", "error") \
                .field("error", error) \
                .time(datetime.utcnow())
            
            self.write_api.write(bucket="aimetrics", org="aiorg", record=point)
            
        except Exception as e:
            self.logger.error(f"Error recording processing error: {str(e)}")
    
    def store_processed_data(self, file_id: str, file_path: str, metadata: Dict, content: str):
        """Store processed data in databases"""
        try:
            # 1. Store file in MinIO
            bucket_name = "raw-data"
            object_name = f"{metadata['dewey_decimal']}/{file_id}/{os.path.basename(file_path)}"
            
            with open(file_path, 'rb') as file_data:
                file_stat = os.stat(file_path)
                self.minio.put_object(
                    bucket_name, 
                    object_name, 
                    file_data, 
                    file_stat.st_size,
                    metadata={
                        "dewey-decimal": metadata['dewey_decimal'],
                        "file-id": file_id,
                        "content-type": metadata['mime_type']
                    }
                )
            
            # 2. Store metadata in MongoDB
            mongo_metadata = metadata.copy()
            # Remove embeddings from MongoDB storage (they'll go to vector store)
            if 'embeddings' in mongo_metadata:
                del mongo_metadata['embeddings']
            
            # Convert datetime objects to strings for MongoDB
            for key, value in mongo_metadata.items():
                if isinstance(value, datetime):
                    mongo_metadata[key] = value.isoformat()
            
            # Add storage location
            mongo_metadata['storage_location'] = {
                'bucket': bucket_name,
                'object': object_name
            }
            
            # Store in MongoDB
            self.db.processed_files.insert_one(mongo_metadata)
            
            # 3. Store in PostgreSQL for indexing
            with self.postgres.cursor() as cursor:
                # Insert into metadata.files table
                cursor.execute("""
                INSERT INTO metadata.files (dewey_decimal, file_name, file_path, file_type, file_size)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """, (
                    metadata['dewey_decimal'],
                    metadata['file_name'],
                    file_path,
                    metadata.get('extension', ''),
                    metadata.get('file_size', 0)
                ))
                
                # Get the inserted file ID
                file_db_id = cursor.fetchone()[0]
                
                # Insert tags
                for tag in metadata.get('tags', []):
                    # Try to find existing tag or insert new one
                    cursor.execute("""
                    WITH ins AS (
                        INSERT INTO metadata.tags (name, category)
                        VALUES (%s, %s)
                        ON CONFLICT (name, category) DO NOTHING
                        RETURNING id
                    )
                    SELECT id FROM ins
                    UNION
                    SELECT id FROM metadata.tags WHERE name = %s AND category = %s
                    """, (tag, 'auto', tag, 'auto'))
                    
                    tag_id = cursor.fetchone()[0]
                    
                    # Insert file-tag relationship
                    cursor.execute("""
                    INSERT INTO metadata.file_tags (file_id, tag_id, confidence)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (file_id, tag_id) DO NOTHING
                    """, (file_db_id, tag_id, 1.0))
                
                # Insert into dewey catalog if not exists
                dewey_num = metadata['dewey_decimal']
                dewey_category = ''
                for dewey_range, category in self.config.get('dewey_ranges', {}).items():
                    start, end = dewey_range.split('-')
                    if int(start) <= int(dewey_num.split('.')[0]) <= int(end):
                        dewey_category = category
                        break
                
                cursor.execute("""
                INSERT INTO indexes.dewey_catalog (dewey_decimal, category, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (dewey_decimal) DO NOTHING
                """, (dewey_num, dewey_category, metadata.get('title', '')))
                
                # Insert vector location if embeddings exist
                if 'embeddings' in metadata:
                    cursor.execute("""
                    INSERT INTO indexes.vector_locations (file_id, vector_store, vector_id, embedding_model, dimensions)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (
                        file_db_id,
                        'meilisearch',
                        file_id,
                        metadata['embeddings'].get('model', 'unknown'),
                        len(metadata['embeddings'].get('embedding', []))
                    ))
                
                # Commit the transaction
                self.postgres.commit()
            
            # 4. Index in Meilisearch
            search_document = {
                'id': file_id,
                'dewey_decimal': metadata['dewey_decimal'],
                'file_name': metadata['file_name'],
                'title': metadata.get('title', metadata['file_name']),
                'summary': metadata.get('summary', ''),
                'tags': metadata.get('tags', []),
                'categories': metadata.get('categories', []),
                'language': metadata.get('language', ''),
                'mime_type': metadata.get('mime_type', ''),
                'extension': metadata.get('extension', ''),
                'created_at': metadata.get('created_at', datetime.utcnow()).isoformat(),
                'file_size': metadata.get('file_size', 0)
            }
            
            # Add embeddings if available
            if 'embeddings' in metadata and 'embedding' in metadata['embeddings']:
                search_document['_vectors'] = {
                    'default': metadata['embeddings']['embedding']
                }
            
            # Index in Meilisearch
            self.meilisearch.index('documents').add_documents([search_document])
            
            # 5. Create graph relationships in Neo4j
            with self.neo4j.session() as session:
                # Create file node
                session.run("""
                MERGE (f:File {id: $file_id})
                SET f.dewey_decimal = $dewey_decimal,
                    f.title = $title,
                    f.file_name = $file_name,
                    f.mime_type = $mime_type,
                    f.created_at = datetime($created_at)
                """, {
                    'file_id': file_id,
                    'dewey_decimal': metadata['dewey_decimal'],
                    'title': metadata.get('title', metadata['file_name']),
                    'file_name': metadata['file_name'],
                    'mime_type': metadata.get('mime_type', ''),
                    'created_at': metadata.get('created_at', datetime.utcnow()).isoformat()
                })
                
                # Create tags and relationships
                for tag in metadata.get('tags', []):
                    session.run("""
                    MERGE (t:Tag {name: $name})
                    MERGE (f:File {id: $file_id})
                    MERGE (f)-[:TAGGED]->(t)
                    """, {
                        'name': tag,
                        'file_id': file_id
                    })
                
                # Create category relationships
                for category in metadata.get('categories', []):
                    session.run("""
                    MERGE (c:Category {name: $name})
                    MERGE (f:File {id: $file_id})
                    MERGE (f)-[:BELONGS_TO]->(c)
                    """, {
                        'name': category,
                        'file_id': file_id
                    })
                
                # Create entity relationships
                for entity in metadata.get('entities', []):
                    if isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                        session.run("""
                        MERGE (e:Entity {name: $name, type: $type})
                        MERGE (f:File {id: $file_id})
                        MERGE (f)-[:MENTIONS]->(e)
                        """, {
                            'name': entity['name'],
                            'type': entity['type'],
                            'file_id': file_id
                        })
            
            # 6. Cache frequently accessed metadata in Redis
            cache_data = {
                'id': file_id,
                'dewey_decimal': metadata['dewey_decimal'],
                'title': metadata.get('title', metadata['file_name']),
                'tags': metadata.get('tags', []),
                'categories': metadata.get('categories', []),
                'storage': {
                    'bucket': bucket_name,
                    'object': object_name
                }
            }
            
            # Cache for 1 hour
            self.redis.setex(f"file:{file_id}", 3600, json.dumps(cache_data))
            self.redis.setex(f"dewey:{metadata['dewey_decimal']}", 3600, json.dumps(cache_data))
            
            self.logger.info(f"Successfully stored processed data for {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing processed data: {str(e)}")
            return False
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/pipeline.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('PipelineProcessor')
    
    def load_config(self):
        """Load pipeline configuration"""
        config_path = "/configs/services/pipeline.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "processors": {
                    "text": ["pdf", "txt", "doc", "docx", "odt", "rtf"],
                    "image": ["jpg", "jpeg", "png", "gif", "bmp", "tiff"],
                    "audio": ["mp3", "wav", "flac", "ogg", "m4a"],
                    "video": ["mp4", "avi", "mkv", "mov", "webm"]
                },
                "embedding_models": {
                    "text": "text-embedding-ada-002",
                    "image": "clip-vit-base-patch32"
                },
                "categories": [
                    "finance", "legal", "medical", "personal", "work", 
                    "academic", "creative", "technical", "correspondence"
                ],
                "dewey_ranges": {
                    "000-099": "Computer science, information & general works",
                    "100-199": "Philosophy & psychology",
                    "200-299": "Religion",
                    "300-399": "Social sciences",
                    "400-499": "Language",
                    "500-599": "Science",
                    "600-699": "Technology",
                    "700-799": "Arts & recreation",
                    "800-899": "Literature",
                    "900-999": "History & geography"
                }
            }
    
    def setup_connections(self):
        """Set up database connections"""
        # Redis connection
        self.redis = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379'))
        
        # MongoDB connection
        self.mongo = MongoClient(os.getenv('MONGODB_URL', 'mongodb://aiuser:password@mongodb:27017'))
        self.db = self.mongo.aiplatform
        
        # Neo4j connection
        self.neo4j = GraphDatabase.driver(
            os.getenv('NEO4J_URL', 'bolt://neo4j:7687'),
            auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
        )
        
        # Meilisearch connection
        self.meilisearch = MeiliSearchClient(
            os.getenv('MEILISEARCH_URL', 'http://meilisearch:7700'),
            os.getenv('MEILISEARCH_KEY', 'masterKey')
        )
        
        # InfluxDB connection
        self.influxdb = InfluxDBClient(
            url=os.getenv('INFLUXDB_URL', 'http://influxdb:8086'),
            token=os.getenv('INFLUXDB_TOKEN', 'aiuser:password'),
            org=os.getenv('INFLUXDB_ORG', 'aiorg')
        )
        self.write_api = self.influxdb.write_api(write_options=SYNCHRONOUS)
        
        # PostgreSQL connection
        self.postgres = psycopg2.connect(os.getenv('POSTGRES_URL', 'postgresql://aiuser:password@postgres:5432/aiplatform'))
        
        # MinIO connection
        self.minio = Minio(
            os.getenv('MINIO_URL', 'minio:9000').replace('http://', ''),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'aiuser'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'password'),
            secure=False
        )
        
        # Ensure buckets exist
        buckets = ['raw-data', 'processed-data', 'embeddings', 'thumbnails']
        for bucket in buckets:
            if not self.minio.bucket_exists(bucket):
                self.minio.make_bucket(bucket)
                self.logger.info(f"Created MinIO bucket: {bucket}")
    
    def process_file(self, file_path: str, priority: str) -> Dict:
        """Process a single file through the AI pipeline"""
        try:
            # Generate a unique ID for this file
            file_id = str(uuid.uuid4())
            
            # Record the start of processing
            self.record_processing_start(file_id, file_path, priority)
            
            # 1. Extract basic metadata
            metadata = self.extract_basic_metadata(file_path)
            metadata['file_id'] = file_id
            metadata['priority'] = priority
            
            # 2. Content extraction based on file type
            content = self.extract_content(file_path, metadata['mime_type'])
            
            # 3. Generate embeddings
            if content:
                metadata['embeddings'] = self.generate_embeddings(content, metadata['mime_type'])
            
            # 4. AI analysis and classification
            if content:
                ai_metadata = self.analyze_content(content, metadata['mime_type'], file_path)
                metadata.update(ai_metadata)
            
            # 5. Generate Dewey Decimal classification
            metadata['dewey_decimal'] = self.generate_dewey_classification(
                metadata.get('title', os.path.basename(file_path)),
                content or "",
                metadata.get('tags', [])
            )
            
            # 6. Store data in databases
            self.store_processed_data(file_id, file_path, metadata, content)
            
            # 7. Record completion of processing
            self.record_processing_complete(file_id, metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            self.record_processing_error(file_id, file_path, str(e))
            raise
    
    def extract_basic_metadata(self, file_path: str) -> Dict:
        """Extract basic metadata from file"""
        metadata = {}
        
        try:
            file_stat = os.stat(file_path)
            
            # Basic file information
            metadata['file_name'] = os.path.basename(file_path)
            metadata['file_path'] = file_path
            metadata['file_size'] = file_stat.st_size
            metadata['created_at'] = datetime.fromtimestamp(file_stat.st_ctime)
            metadata['modified_at'] = datetime.fromtimestamp(file_stat.st_mtime)
            
            # Determine MIME type
            mime = magic.Magic(mime=True)
            metadata['mime_type'] = mime.from_file(file_path)
            
            # File extension
            _, extension = os.path.splitext(file_path)
            metadata['extension'] = extension.lower().lstrip('.')
            
            # Calculate hash for deduplication
            metadata['md5_hash'] = self.calculate_file_hash(file_path)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting basic metadata: {str(e)}")
            raise
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for deduplication"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in 1MB chunks
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def extract_content(self, file_path: str, mime_type: str) -> str:
        """Extract content from file based on MIME type"""
        try:
            # Text-based documents
            if mime_type.startswith('text/') or mime_type in ['application/pdf', 'application/msword', 
                                                              'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                # Use textract for most text documents
                return textract.process(file_path).decode('utf-8', errors='ignore')
                
            # Images - return None for now as we'll use image analysis
            elif mime_type.startswith('image/'):
                # For images, we'll analyze using the image itself, not extracted text
                return None
                
            # Audio - return None, will use specialized audio processing
            elif mime_type.startswith('audio/'):
                return None
                
                # Video - extract metadata and keyframes
            elif mime_type.startswith('video/'):
                try:
                    # Extract metadata using OpenCV
                    cap = cv2.VideoCapture(file_path)
                    
                    # If we can't open the file, return None
                    if not cap.isOpened():
                        self.logger.warning(f"Could not open video file: {file_path}")
                        
                    # Extract metadata using OpenCV
                    cap = cv2.VideoCapture(file_path)
                    
                    # If we can't open the file, return None
                    if not cap.isOpened():
                        self.logger.warning(f"Could not open video file: {file_path}")
                        return None
                    
                    # Extract video metadata
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Extract keyframes (1 frame every 10 seconds)
                    keyframes = []
                    keyframe_interval = int(fps * 10)  # Every 10 seconds
                    
                    # Process only up to 10 keyframes to avoid excessive processing
                    max_keyframes = min(10, int(frame_count / keyframe_interval) + 1)
                    
                    for i in range(max_keyframes):
                        frame_pos = i * keyframe_interval
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                        ret, frame = cap.read()
                        if ret:
                            # Save keyframe to temporary file
                            temp_keyframe = f"/tmp/keyframe_{i}.jpg"
                            cv2.imwrite(temp_keyframe, frame)
                            keyframes.append(temp_keyframe)
                    
                    cap.release()
                    
                    # Process keyframes with OCR or image analysis if needed
                    # For now, we'll just return video metadata as a JSON string
                    video_metadata = {
                        "duration": duration,
                        "fps": fps,
                        "frame_count": frame_count,
                        "width": width,
                        "height": height,
                        "keyframe_count": len(keyframes),
                        "keyframes": keyframes
                    }
                    
                    return json.dumps(video_metadata)
                except Exception as e:
                    self.logger.error(f"Error extracting video content: {str(e)}")
                    return None
