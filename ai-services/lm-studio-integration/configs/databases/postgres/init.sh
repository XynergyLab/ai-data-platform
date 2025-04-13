#!/bin/bash
set -e

# Create required schemas
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS metadata;
    CREATE SCHEMA IF NOT EXISTS indexes;
EOSQL

# Load and execute schema definitions from config
python3 <<-EOPYTHON
import json
import psycopg2
import os

# Load configuration
with open('/docker-entrypoint-initdb.d/config.json', 'r') as f:
    config = json.load(f)

# Connect to database
conn = psycopg2.connect(
    dbname=os.environ['POSTGRES_DB'],
    user=os.environ['POSTGRES_USER'],
    password=os.environ['POSTGRES_PASSWORD'],
    host='localhost'
)
conn.autocommit = True
cur = conn.cursor()

# Create schemas and tables
for schema in config['schema_init']:
    schema_name = schema['name']
    for table_name, table_sql in schema['tables'].items():
        cur.execute(table_sql)

# Create indexes
for index_sql in config['indexes']:
    cur.execute(index_sql)

cur.close()
conn.close()
EOPYTHON
