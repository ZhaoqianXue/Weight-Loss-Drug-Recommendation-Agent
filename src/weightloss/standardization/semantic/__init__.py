"""Semantic standardization of KG side effects and purposes (design: kg-fda-design-20261001-01a0fad5,
docs/dual-mapping-implementation-plan.md).

Two mappings share one lookup machinery: a SideEffect node maps to a current MedDRA LLT (its PT derived from the
same frozen file); an Indication node maps to an exact concept (MedDRA LLT or a project registry concept) plus a
purpose intent. The KG is read only; mapping status never changes review, identity, occurrence or eligibility.
Importing this package reads no data and creates no client.
"""
VERSION = 'semantic-map-0.1.4'
