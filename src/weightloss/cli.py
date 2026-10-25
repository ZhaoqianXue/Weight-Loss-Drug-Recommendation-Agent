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
    reddit = sub.add_parser('collect-reddit', help='Bounded, resumable Arctic Shift pilot')
    reddit.add_argument('--subreddit', required=True)
    reddit.add_argument('--output-dir', required=True)
    reddit.add_argument('--max-requests', type=int, default=12)
    batch = sub.add_parser('collect-reddit-batch', help='Resumable source inventory with one shared snapshot')
    batch.add_argument('--output-dir', required=True)
    batch.add_argument('--max-requests', type=int, default=100)
    batch.add_argument('--subreddits', nargs='+')
    production = sub.add_parser('collect-reddit-production', help='Continue the full-scope production queue')
    production.add_argument('--output-dir', required=True)
    production.add_argument('--max-requests', type=int, default=1200)
    production.add_argument('--per-community', type=int, default=200)
    worker = sub.add_parser('reddit-production-worker', help='Continuously run resumable production rounds')
    worker.add_argument('--output-dir', required=True)
    worker.add_argument('--report-file', required=True)
    worker.add_argument('--max-requests', type=int, default=1200)
    worker.add_argument('--per-community', type=int, default=200)
    worker.add_argument('--max-rounds', type=int)
    reddit_status = sub.add_parser('reddit-production-status', help='Read live collection progress without downloading')
    reddit_status.add_argument('--output-dir', required=True)
    reddit_status.add_argument('--report-file')
    remove = sub.add_parser('reddit-remove-communities', help='Explicitly delete excluded community data from a stopped production run')
    remove.add_argument('--output-dir', required=True)
    selection = remove.add_mutually_exclusive_group(required=True)
    selection.add_argument('--subreddits', nargs='+', help='Communities to delete')
    selection.add_argument('--retain-subreddits', nargs='+', help='Keep exactly these communities; delete all other current sources')
    remove.add_argument('--reason', required=True)
    clean = sub.add_parser('clean-reddit', help='Resumable offline cleaning of verified immutable Reddit exports')
    clean.add_argument('--production-dir', required=True)
    clean.add_argument('--output-dir', required=True)
    clean.add_argument('--report-file', required=True)
    clean.add_argument('--language-model', required=True)
    clean.add_argument('--policy', help='Defaults to configs/reddit-cleaning.json')
    clean.add_argument('--subreddits', nargs='+')
    clean.add_argument('--max-records', type=int)
    clean.add_argument('--workers', type=int, choices=range(1,5), default=4)
    archive = sub.add_parser('import-reddit-archive', help='Verify and stream selected communities from JSONL/ZST dumps')
    archive.add_argument('--manifest', required=True)
    archive.add_argument('--output-dir', required=True)
    fixture = sub.add_parser('reproduce-fixture')
    fixture.add_argument('--output-dir',required=True)
    site = sub.add_parser('setup-site', help='Set up the website on a fresh checkout from the site data bundle')
    site.add_argument('--bundle', help='Path to askglp1-site-data-*.zip (from the project Google Drive folder)')
    site.add_argument('--make-bundle', metavar='ZIP', help='Write the bundle from this machine\'s data instead')
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
    if command == 'clean-reddit':
        from .cleaning.reddit_unified import main as clean_main
        root = get_settings().root
        argv=['v1','--source-dir',str(args.production_dir),'--output-dir',str(args.output_dir),
              '--report-file',str(args.report_file),'--language-model',str(args.language_model),
              '--workers',str(args.workers),'--config',str(root/'configs/reddit.json'),
              '--policy',str(args.policy or root/'configs/reddit-cleaning.json')]
        if args.subreddits:argv+=['--subreddits',*args.subreddits]
        if args.max_records is not None:argv+=['--max-records',str(args.max_records)]
        clean_main(argv)
        return 0
    if command == 'reddit-remove-communities':
        from .ingestion.reddit_scope import remove_communities
        print(json.dumps(remove_communities(args.output_dir, args.subreddits, args.reason,
                                           retained=args.retain_subreddits), indent=2))
        return 0
    if command == 'reddit-production-worker':
        from .ingestion.reddit_worker import run_worker
        print(json.dumps(run_worker(args.output_dir, args.report_file,
                                    args.max_requests, args.per_community,
                                    max_rounds=args.max_rounds), indent=2))
        return 0
    if command == 'reddit-production-status':
        from .ingestion.reddit_production import production_status
        print(json.dumps(production_status(args.output_dir, args.report_file), indent=2))
        return 0
    if command == 'collect-reddit-production':
        from .ingestion.reddit_production import collect_production
        print(json.dumps(collect_production(args.output_dir, args.max_requests, args.per_community), indent=2))
        return 0
    if command == 'collect-reddit-batch':
        from .ingestion.reddit import collect_batch
        if args.max_requests < 1:
            raise ValueError('--max-requests must be positive')
        print(json.dumps(collect_batch(args.output_dir, args.max_requests, args.subreddits), indent=2))
        return 0
    if command == 'import-reddit-archive':
        from .ingestion.reddit_archive import import_archives
        scope = json.loads((get_settings().root / 'configs/reddit.json').read_text())
        print(json.dumps(import_archives(args.manifest, args.output_dir, scope), indent=2))
        return 0
    if command == 'collect-reddit':
        from .ingestion.reddit import collect_pilot
        print(json.dumps(collect_pilot(args.subreddit, args.output_dir, args.max_requests), indent=2))
        return 0
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
    if command == 'setup-site':
        from .site_setup import make_bundle, setup
        if args.make_bundle:
            make_bundle(args.make_bundle)
            return 0
        return setup(args.bundle)
    if command == 'serve':
        import uvicorn
        from .graph import neo4j_db
        from .web.kg import graph_running
        paths = get_settings()
        if not graph_running() and neo4j_db.MANIFEST.is_file() and neo4j_db.installation() is not None:
            print('Starting the knowledge graph database...',flush=True)  # imported earlier by setup-site
            neo4j_db.start()
        if not graph_running():
            print('The knowledge graph is not set up: run make site-setup BUNDLE=<site data zip>; '
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
