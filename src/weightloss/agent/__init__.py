"""Evidence-grounded question answering over the knowledge graph (project document Section 3.7).

tools  graph tools: parameterized Cypher over the local Neo4j graph (link_terms, query_graph, get_evidence,
       compare_faers), no LLM
agent  Responses API tool loop, final-answer schema and the citation and quote verifier
"""
VERSION = 'agent-0.3.0'
