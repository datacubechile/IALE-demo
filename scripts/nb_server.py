"""
Publish a notebook "live" as HTML on a public server/port using streamlit. The
content gets re-rendered and published every REFRESH seconds.
"""

from os import environ
from pathlib import Path
from time import sleep

import click
import nbformat
import streamlit as st
import streamlit.components.v1 as components
from streamlit.script_request_queue import RerunData
from streamlit.script_runner import RerunException
from nbconvert import HTMLExporter
from streamlit import config

# Refresh rate in seconds.
REFRESH = 2.0

# Base URL of this JupyterLab session.
ROOT_URL = "https://hub.sg-dev.easi-eo.solutions/"


def rerun():
    """Force the app to re-run.

    https://discuss.streamlit.io/t/remove-custom-components/6993/2
    """
    raise RerunException(RerunData(None))


def serve_notebook(path):
    """Convert the current notebook to HTML and serve."""
    html_exporter = HTMLExporter()
    html_exporter.template_name = "classic"

    with path.open() as fh:
        response = fh.read()
    nb = nbformat.reads(response, as_version=4)
    body, _ = html_exporter.from_notebook_node(nb)
    # Use a large height value as it crops notebook contents
    render = components.html(body, width=800, height=50000)

    # Sleep and rerun the whole notebook, otherwise the data doesn't seem to get
    # refreshed
    sleep(REFRESH)
    rerun()


@click.command()
@click.argument(
    "notebook",
    type=click.Path(exists=True),
)
def main(notebook):
    """Publish NOTEBOOK as HTML on a public server/port.

    usage: streamlit run nb_server.py -- <path/to/notebook.ipynb>
    """
    port = config.get_option("server.port")
    link = f'{ROOT_URL}{environ["JUPYTERHUB_SERVICE_PREFIX"]}proxy/{port}/'
    print(f"Serving at {link}")
    serve_notebook(Path(notebook))


if __name__ == "__main__":
    main()
