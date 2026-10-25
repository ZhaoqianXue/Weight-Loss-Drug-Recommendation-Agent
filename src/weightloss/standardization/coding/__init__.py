"""MedDRA coding of KG side-effect and purpose mentions (plan: docs/meddra-coding-plan.md).

Retrieve candidate LLTs with a local BioLORD-2023 index, let gpt-6-luna choose one candidate or "none" through the
University of Arizona AI Assistant (AWS Bedrock), then map LLT -> PT by table lookup. MedDRA content may only be sent
to the UA endpoint (docs/meddra-processing-compliance.md).
"""
VERSION = 'meddra-coding-0.1.0'
