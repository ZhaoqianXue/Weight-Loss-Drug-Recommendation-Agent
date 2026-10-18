"""Website backend: a FastAPI service over the knowledge graph and the AskGLP1 agent (see apps/web for the frontend).

kg.py shapes the agent's graph-tool outputs into page views; ask.py runs the agent and streams its progress; app.py only
exposes them over HTTP. Importing this package connects to nothing; FastAPI is imported only by app.py.
"""
