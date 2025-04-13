-- Initialize PostgreSQL database for AI platform

-- Create schemas
CREATE SCHEMA IF NOT EXISTS metadata;
CREATE SCHEMA IF NOT EXISTS indexes;
CREATE SCHEMA IF NOT EXISTS config;

-- Create metadata tables
CREATE TABLE IF NOT EXISTS metadata.files (
    id SERIAL PRIMARY KEY,
    dewey_decimal VARCHAR(20) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dewey_decimal)
);

CREATE TABLE IF NOT EXISTS metadata.tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (name, category)
);

CREATE TABLE IF NOT EXISTS metadata.file_tags (
    file_id INTEGER REFERENCES metadata.files(id),
    tag_id INTEGER REFERENCES metadata.tags(id),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (file_id, tag_id)
);

-- Create index tables
CREATE TABLE IF NOT EXISTS indexes.dewey_catalog (
    dewey_decimal VARCHAR(20) PRIMARY KEY,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS indexes.vector_locations (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES metadata.files(id),
    vector_store VARCHAR(50),
    vector_id VARCHAR(255),
    embedding_model VARCHAR(100),
    dimensions INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create configuration tables
CREATE TABLE IF NOT EXISTS config.agent_configs (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS config.processing_pipelines (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    steps JSONB NOT NULL,
    file_types VARCHAR[] NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_files_dewey ON metadata.files(dewey_decimal);
CREATE INDEX idx_files_type ON metadata.files(file_type);
CREATE INDEX idx_tags_name ON metadata.tags(name);
CREATE INDEX idx_tags_category ON metadata.tags(category);
CREATE INDEX idx_file_tags_file_id ON metadata.file_tags(file_id);
CREATE INDEX idx_file_tags_tag_id ON metadata.file_tags(tag_id);
CREATE INDEX idx_dewey_category ON indexes.dewey_catalog(category);
CREATE INDEX idx_vector_locations_file_id ON indexes.vector_locations(file_id);
CREATE INDEX idx_agent_configs_name ON config.agent_configs(agent_name);
CREATE INDEX idx_processing_pipelines_name ON config.processing_pipelines(name);
CREATE INDEX idx_processing_pipelines_priority ON config.processing_pipelines(priority);
CREATE INDEX idx_processing_pipelines_active ON config.processing_pipelines(active);

-- Create paperless database
CREATE DATABASE paperless;

-- Create users and set permissions
CREATE USER aiuser WITH PASSWORD 'your_secure_postgres_password';
GRANT ALL PRIVILEGES ON DATABASE aiplatform TO aiuser;
GRANT ALL PRIVILEGES ON DATABASE paperless TO aiuser;
GRANT ALL PRIVILEGES ON SCHEMA metadata TO aiuser;
GRANT ALL PRIVILEGES ON SCHEMA indexes TO aiuser;
GRANT ALL PRIVILEGES ON SCHEMA config TO aiuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metadata TO aiuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA indexes TO aiuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA config TO aiuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metadata TO aiuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA indexes TO aiuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA config TO aiuser;

