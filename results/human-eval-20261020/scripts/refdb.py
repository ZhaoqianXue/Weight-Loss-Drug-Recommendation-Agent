"""Read-only connections for writing the reference answers of the human evaluation.

    import sys; sys.path.insert(0, 'results/human-eval-20261020/scripts'); from refdb import db, kg, q

db  the analysis database (weightloss.graph.analysis_db: statement, faers_report, faers_stratum, faers_reaction, meddra,
    approval) with wilson_low(k, n), wilson_high(k, n) and REGEXP, plus the temporary views that apply the frozen RQ2
    comparison rules: faers_scope (FAERS reports of a scope such as 'brand:Ozempic', 'ingredient:tirzepatide', 'all')
    and comparable_statement (Reddit/WebMD side-effect statements of the same scope inside the comparison window,
    clinical events only).
kg  the knowledge graph file with the cited source sentences (edge tables: text = wording, evidence = JSON list).
q   q(sql, *args, con=db) -> pandas DataFrame.

The views are the ones the frozen RQ2 tables are reproduced with (weightloss.agent.code_baseline.AGENT_VIEWS); nothing
here reads any system's answers.
"""
from weightloss.agent import code_baseline

_ns = {}
exec(code_baseline.SETUP + code_baseline.AGENT_VIEWS, _ns)
db, kg, pd = _ns['db'], _ns['kg'], _ns['pd']


def q(sql, *args, con=None):
    return pd.read_sql_query(sql, con or db, params=args or None)
