
// Initialize MongoDB for raw data storage and processing queues

db = db.getSiblingDB('aiplatform');

// Create collections for raw data
db.createCollection('raw_files');
db.createCollection('processed_files');
db.createCollection('metadata');

// Create collections for queues
db.createCollection('processing_queue');
db.createCollection('embedding_queue');
db.createCollection('indexing_queue');
db.createCollection('failed_jobs');

// Create indexes
db.raw_files.createIndex({ "dewey_decimal": 1 }, { unique: true });
db.raw_files.createIndex({ "file_type": 1 });
db.raw_files.createIndex({ "priority": 1 });
db.raw_files.createIndex({ "created_at": 1 });

db.processed_files.createIndex({ "dewey_decimal": 1 }, { unique: true });
db.processed_files.createIndex({ "processing_status": 1 });

db.metadata.createIndex({ "dewey_decimal": 1 }, { unique: true });
db.metadata.createIndex({ "tags": 1 });

db.processing_queue.createIndex({ "priority": 1, "created_at": 1 });
db.processing_queue.createIndex({ "status": 1 });

db.embedding_queue.createIndex({ "model": 1 });
db.embedding_queue.createIndex({ "status": 1 });

db.indexing_queue.createIndex({ "target_store": 1 });
db.indexing_queue.createIndex({ "status": 1 });

db.failed_jobs.createIndex({ "job_type": 1 });
db.failed_jobs.createIndex({ "failed_at": 1 });

// Create admin user
db.createUser({
  user: "aiuser",
  pwd: "your_secure_mongodb_password",
  roles: [
    { role: "readWrite", db: "aiplatform" },
    { role: "dbAdmin", db: "aiplatform" }
  ]
});

// Create nodebb database and user
db = db.getSiblingDB('nodebb');
db.createUser({
  user: "aiuser",
  pwd: "your_secure_mongodb_password",
  roles: [
    { role: "readWrite", db: "nodebb" },
    { role: "dbAdmin", db: "nodebb" }
  ]
});

// Create sample document schemas
db = db.getSiblingDB('aiplatform');

// Example document for raw_files
db.raw_files.insertOne({
  "dewey_decimal": "000.001",
  "file_name": "example.pdf",
  "file_path": "/raw-data/high-priority/example.pdf",
  "file_type": "pdf",
  "file_size": 1024000,
  "priority": "high",
  "created_at": new Date(),
  "status": "pending",
  "checksum": "5eb63bbbe01eeed093cb22bb8f5acdc3"
});

// Example document for processing_queue
db.processing_queue.insertOne({
  "dewey_decimal": "000.001",
  "processor": "pdf_extractor",
  "priority": "high",
  "status": "pending",
  "created_at": new Date(),
  "attempts": 0,
  "max_attempts": 3
});

// Remove example documents (keeping as templates only)
db.raw_files.deleteOne({ "dewey_decimal": "000.001" });
db.processing_queue.deleteOne({ "dewey_decimal": "000.001" });

