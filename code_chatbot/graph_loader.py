# graph_loader.py
"""
Neo4j Graph Database Loader for Standardized Drug Reviews
Creates only Drug, Condition, and SideEffect nodes from standardized_info column
Establishes relationships from standardized_relations column
"""
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jGraphLoader:
    """Handles loading standardized drug review data into Neo4j graph database"""
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.verify_connection()
    def verify_connection(self):
        """Verify Neo4j database connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")
    def clear_database(self):
        """Clear all data from the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for optimal performance"""
        constraints_and_indexes = [
            # Unique constraints for the three node types only
            "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT condition_name_unique IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT side_effect_name_unique IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
            # Indexes for performance
            "CREATE INDEX drug_generic_name_index IF NOT EXISTS FOR (d:Drug) ON (d.generic_name)",
            "CREATE INDEX condition_severity_index IF NOT EXISTS FOR (c:Condition) ON (c.severity)",
            "CREATE INDEX side_effect_severity_index IF NOT EXISTS FOR (s:SideEffect) ON (s.severity)",
        ]
        with self.driver.session() as session:
            for constraint in constraints_and_indexes:
                try:
                    session.run(constraint)
                    logger.info(f"Applied: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint/Index may already exist: {e}")
    def safe_json_parse(self, json_str: str) -> Optional[Dict]:
        """Safely parse JSON string, return None if invalid"""
        if pd.isna(json_str) or not json_str:
            return None
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse JSON: {json_str[:100]}... Error: {e}")
            return None
    def create_drug_node(self, session, drug_info: Dict, brand_name: str, drug_name: str):
        """Create or merge Drug node with properties from standardized_info"""
        query = """
        MERGE (d:Drug {name: $brand_name})
        SET d.generic_name = $drug_name,
            d.dosage = $dosage,
            d.dosage_form = $dosage_form,
            d.duration = $duration,
            d.continued_use = $continued_use,
            d.alternative_drug_considered = $alternative_drug_considered
        RETURN d
        """
        session.run(query, 
                   brand_name=brand_name,
                   drug_name=drug_name,
                   dosage=drug_info.get('dosage'),
                   dosage_form=drug_info.get('dosage_form'),
                   duration=drug_info.get('duration'),
                   continued_use=drug_info.get('continued_use'),
                   alternative_drug_considered=drug_info.get('alternative_drug_considered'))
    def create_condition_node(self, session, condition_info: Dict):
        """Create or merge Condition node with properties from standardized_info"""
        if not condition_info.get('name'):
            return
        query = """
        MERGE (c:Condition {name: $name})
        SET c.severity = $severity
        RETURN c
        """
        session.run(query,
                   name=condition_info.get('name'),
                   severity=condition_info.get('severity'))
    def create_side_effect_nodes(self, session, side_effects: List[Dict]):
        """Create or merge SideEffect nodes with properties from standardized_info"""
        for side_effect in side_effects:
            if not side_effect.get('name'):
                continue
            query = """
            MERGE (s:SideEffect {name: $name})
            SET s.severity = $severity,
                s.associated_drug = $associated_drug
            RETURN s
            """
            session.run(query,
                       name=side_effect.get('name'),
                       severity=side_effect.get('severity'),
                       associated_drug=side_effect.get('associated_drug'))
    def create_relationships_from_standardized_data(self, session, relations: List[Dict]):
        """Create relationships between Drug, Condition, and SideEffect nodes based on standardized_relations"""
        for relation in relations:
            start_label = relation['start']['label']
            end_label = relation['end']['label']
            relation_type = relation['relation'].upper()
            # Map standardized_relations labels to our three node types
            label_mapping = {
                'Medication': 'Drug',
                'Disease': 'Condition',
                'SideEffect': 'SideEffect'
            }
            start_mapped = label_mapping.get(start_label, start_label)
            end_mapped = label_mapping.get(end_label, end_label)
            # Only create relationships between our three allowed node types
            allowed_labels = {'Drug', 'Condition', 'SideEffect'}
            if start_mapped not in allowed_labels or end_mapped not in allowed_labels:
                continue
            # Create relationship query dynamically
            query = f"""
            MATCH (start:{start_mapped} {{name: $start_name}})
            MATCH (end:{end_mapped} {{name: $end_name}})
            MERGE (start)-[r:{relation_type}]->(end)
            SET r += $properties
            RETURN r
            """
            try:
                session.run(query,
                           start_name=relation['start']['properties']['name'],
                           end_name=relation['end']['properties']['name'],
                           properties=relation.get('properties', {}))
            except Exception as e:
                logger.warning(f"Failed to create relationship: {e}")
    def load_csv_data(self, csv_file_path: str):
        """Main method to load CSV data into Neo4j - creates only Drug, Condition, SideEffect nodes"""
        logger.info(f"Loading data from {csv_file_path}")
        # Read CSV file
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            return
        # Create constraints and indexes
        self.create_constraints_and_indexes()
        # Process each row to extract nodes and relationships from standardized columns
        processed_count = 0
        error_count = 0
        with self.driver.session() as session:
            for index, row in df.iterrows():
                try:
                    # Parse standardized data
                    standardized_info = self.safe_json_parse(row.get('standardized_info'))
                    standardized_relations = self.safe_json_parse(row.get('standardized_relations'))
                    if standardized_info:
                        # Extract the three types of nodes from standardized_info
                        drug_info = standardized_info.get('drug', {})
                        condition_info = standardized_info.get('condition', {})
                        side_effects = standardized_info.get('side_effects', [])
                        brand_name = row.get('Brand Name')
                        drug_name = row.get('Drug Name')
                        # Create only the three allowed node types
                        if drug_info and brand_name:
                            self.create_drug_node(session, drug_info, brand_name, drug_name)
                        if condition_info:
                            self.create_condition_node(session, condition_info)
                        if side_effects:
                            self.create_side_effect_nodes(session, side_effects)
                    # Create relationships from standardized_relations between the three node types
                    if standardized_relations:
                        self.create_relationships_from_standardized_data(session, standardized_relations)
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} rows...")
                except Exception as e:
                    logger.error(f"Error processing row {index}: {e}")
                    error_count += 1
                    continue
        logger.info(f"Data loading completed: {processed_count} rows processed, {error_count} errors")
    def get_database_stats(self):
        """Get statistics about the loaded data - only three node types"""
        queries = {
            "Total Nodes": "MATCH (n) RETURN count(n) as count",
            "Total Relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "Drug Nodes": "MATCH (d:Drug) RETURN count(d) as count",
            "Condition Nodes": "MATCH (c:Condition) RETURN count(c) as count",
            "SideEffect Nodes": "MATCH (s:SideEffect) RETURN count(s) as count",
            "TREATS Relationships": "MATCH ()-[r:TREATS]->() RETURN count(r) as count",
            "CAUSES Relationships": "MATCH ()-[r:CAUSES]->() RETURN count(r) as count",
        }
        stats = {}
        with self.driver.session() as session:
            for stat_name, query in queries.items():
                try:
                    result = session.run(query)
                    stats[stat_name] = result.single()['count']
                except:
                    stats[stat_name] = 0
        return stats
    
def main():
    """Main execution function"""
    # Configuration - Update these values according to your Neo4j setup
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "weightloss"  
    CSV_FILE_PATH = "data_standardized/standardized_reviews_all.csv"
    # Verify CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"CSV file not found: {CSV_FILE_PATH}")
        return
    # Initialize loader
    loader = None
    try:
        loader = Neo4jGraphLoader(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        # # Optional: Clear existing data (uncomment if needed)
        # logger.info("Clearing existing database...")
        # loader.clear_database()
        # Load data - creates only Drug, Condition, SideEffect nodes
        start_time = datetime.now()
        loader.load_csv_data(CSV_FILE_PATH)
        end_time = datetime.now()
        # Print statistics
        stats = loader.get_database_stats()
        logger.info("Database Statistics (Three Node Types Only):")
        for stat_name, count in stats.items():
            logger.info(f"  {stat_name}: {count}")
        logger.info(f"Total processing time: {end_time - start_time}")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if loader:
            loader.close()

if __name__ == "__main__":
    main()