"""Write the API schema used to generate the frontend types: python -m weightloss.web.openapi OUTPUT."""
import json
from pathlib import Path
import sys

from .app import openapi_document


def render() -> str:
    return json.dumps(openapi_document(), indent=2, ensure_ascii=False) + '\n'


def main(argv=None):
    output = Path((sys.argv[1:] if argv is None else argv)[0])
    output.write_text(render())


if __name__ == '__main__':
    main()
