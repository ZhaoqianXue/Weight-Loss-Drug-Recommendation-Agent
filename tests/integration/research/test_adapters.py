"""Run explicitly with the research lock installed; all provider/model calls are mocked."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

ROOT=Path(__file__).resolve().parents[3]

class AdapterTests(unittest.TestCase):
    def test_imports_do_not_start_models_read_csv_or_download(self):
        script='''
from unittest.mock import patch
import pandas, openai, sentence_transformers
with patch.object(pandas, 'read_csv', side_effect=AssertionError('CSV at import')), patch.object(openai, 'OpenAI', side_effect=AssertionError('Client at import')), patch.object(sentence_transformers, 'SentenceTransformer', side_effect=AssertionError('Model at import')):
 import weightloss.extraction.schema_extraction
 import weightloss.standardization.baseline
 import weightloss.retrieval.table_loader
 import weightloss.retrieval.graph_loader
 import weightloss.retrieval.chatbot
'''
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=300)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_structured_response_and_parse_failure(self):
        from weightloss.extraction import schema_extraction as module
        fake={'structured_info':{'drug':{'name':'Wegovy'},'condition':{'name':'Other'},'side_effects':[]},'relations':[]}
        parsed=SimpleNamespace(model_dump=lambda:fake)
        chain=Mock();chain.invoke.return_value={'raw':None,'parsed':parsed,'parsing_error':None}
        with patch.object(module,'get_extraction_chain',return_value=chain):
            result=module.safe_extract_structured('synthetic text','Wegovy','Other','1 month')
            self.assertIsNotNone(result)
            self.assertEqual(result['structured_info']['drug']['name'],'Wegovy')
            chain.invoke.return_value={'raw':None,'parsed':None,'parsing_error':ValueError()}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(module.safe_extract_structured('synthetic','Wegovy'))

    def test_standardization_restricts_result_to_candidates(self):
        from weightloss.standardization import baseline
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='nausea'))])
        client=Mock();client.chat.completions.create.return_value=response
        with patch.object(baseline,'get_client',return_value=client):
            self.assertEqual(baseline.gpt_resolve_match('queasy',['nausea']),'nausea')
            self.assertIsNone(baseline.gpt_resolve_match('queasy',['headache']))
            client.chat.completions.create.side_effect=RuntimeError('synthetic failure')
            with self.assertRaises(RuntimeError):baseline.gpt_resolve_match('queasy',['nausea'])

    def test_index_builder_propagates_failure(self):
        from weightloss.retrieval import table_loader
        with patch.object(table_loader.pd,'read_csv',side_effect=FileNotFoundError('synthetic')):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):table_loader.build_database(table_loader.CONFIG)

    def test_baseline_no_effects_does_not_call_model(self):
        import pandas as pd
        import numpy as np
        from weightloss.standardization import baseline
        info={'drug':{'name':'Wegovy'},'condition':{'name':'Other'},'side_effects':[]}
        row=pd.Series({'structured_info':json.dumps(info),'relations':'[]'})
        model=Mock();model.encode.side_effect=AssertionError('unnecessary model call')
        result=baseline.process_row(row,model,[],np.empty((0,768)))
        self.assertEqual(json.loads(result['standardized_info'])['side_effects'],[])

if __name__=='__main__':unittest.main()
