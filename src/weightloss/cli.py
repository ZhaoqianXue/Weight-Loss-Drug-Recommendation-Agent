"""One explicit entry point for offline checks and opt-in research workloads."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from .settings import configure, get_settings

def main(argv=None):
    parser = argparse.ArgumentParser(prog='weightloss')
    parser.add_argument('--root', help='Research checkout, independent of cwd')
    parser.add_argument('--config', help='Pipeline config JSON')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('validate','build-web','refresh','freeze','import-graph','stats'):
        sub.add_parser(name)
    for name in ('extract','standardize'):
        sub.add_parser(name).add_argument('--limit',type=int)
    new = sub.add_parser('new-run')
    new.add_argument('--run-id',required=True)
    new.add_argument('--raw-dir')
    collect = sub.add_parser('collect')
    collect.add_argument('--output-dir',required=True)
    collect.add_argument('--run-dir')
    fixture = sub.add_parser('reproduce-fixture')
    fixture.add_argument('--output-dir',required=True)
    serve = sub.add_parser('serve', help='Run the website: FastAPI API plus the built frontend (apps/web/dist)')
    serve.add_argument('--port',type=int,default=5001)
    serve.add_argument('--reload',action='store_true',help='Restart when Python sources change (development)')
    args = parser.parse_args(argv)
    configure(args.root,args.config)
    try:
        return dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'{error}\n')

def dispatch(args):
    command = args.command
    if command == 'reproduce-fixture':
        from .pipeline.fixture import reproduce
        print(reproduce(Path(args.output_dir).expanduser().resolve()))
        return 0
    if command == 'new-run':
        from .runs import new_run
        print(new_run(args.run_id,args.raw_dir))
        return 0
    if command == 'validate':
        from .pipeline.validate_data import main
        main()
        return 0
    if command == 'build-web':
        from .pipeline.build_web import build_web
        print(build_web())
        return 0
    if command == 'serve':
        import uvicorn
        from .web.kg import graph_running
        paths = get_settings()
        if not graph_running():
            print('The knowledge graph database is not running (python -m weightloss.graph.neo4j_db start); '
                  'the API will answer 503 until it is.',flush=True)
        if not (paths.path('web_source')/'dist'/'index.html').is_file():
            print('Frontend not built (make web-build); serving the API only.',flush=True)
        print(f'http://127.0.0.1:{args.port}',flush=True)
        if args.reload:
            uvicorn.run('weightloss.web.app:create_app',factory=True,host='127.0.0.1',port=args.port,reload=True,reload_dirs=[str(Path(__file__).parent)])
        else:
            from .web.app import create_app
            uvicorn.run(create_app(),host='127.0.0.1',port=args.port)
        return 0
    if command in ('refresh','extract','standardize','freeze'):
        get_settings().require_writable()
    if command in ('extract','standardize'):
        if args.limit is not None and args.limit < 0:
            raise ValueError('--limit must be nonnegative')
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is required; no data changed.')
    from .runs import operation, hashes
    with operation(command,{k:v for k,v in vars(args).items() if k not in ('root','config')}) as report:
        if command == 'freeze':
            from .runs import freeze
            freeze()
        elif command == 'collect':
            from .ingestion.batch_scraper import main
            options = ['--output-dir',args.output_dir]
            if args.run_dir: options += ['--run-dir',args.run_dir]
            main(options)
            report["artifacts_sha256"] = hashes(p for p in Path(args.output_dir).rglob("*") if p.is_file())
        elif command in ('extract','standardize'):
            module = importlib.import_module('weightloss.'+('extraction.extract_all' if command == 'extract' else 'standardization.standardize_all'))
            module.main([] if args.limit is None else ['--limit',str(args.limit)])
        elif command == 'refresh':
            from .pipeline.refresh_data import main
            main()
        elif command == 'import-graph':
            from .pipeline.validate_data import validate
            validate()
            from .retrieval.graph_loader import main
            main()
        elif command == 'stats':
            from .evaluation.standardization_stat import main
            main()
            report["artifacts_sha256"] = hashes([get_settings().path('results')/'metrics.json'])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
