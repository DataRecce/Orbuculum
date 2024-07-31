import os

import click
from rich.console import Console
from rich.prompt import Prompt

from orbuculum.database import load_documents, split_documents, add_to_chroma, clear_database
from orbuculum.rag import query_orbuculum

console = Console()
DEBUG_MODE = False


def get_version():
    version_file = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'VERSION'))
    with open(version_file) as fh:
        version_name = fh.read().strip()
        return version_name


__version__ = get_version()


@click.group(context_settings={'help_option_names': ['-h', '--help']}, invoke_without_command=True)
@click.option('--debug', is_flag=True, help='Enable debug mode.')
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = kwargs.get('debug', False)

    if ctx.invoked_subcommand is None:
        ctx.invoke(ask)


@cli.command(help='Show the version of orbuculum.')
def version():
    console.print(f'[bold blue]{__version__}[/bold blue]')


@cli.command(help='Ask a question based on the provided PDF documents.')
@click.argument('query_text', required=False)
@click.pass_context
def ask(ctx, query_text: str = None, **kwargs):
    if query_text is None:
        query_text = Prompt.ask('[bold green]Enter your query[/bold green]')

    response_answer = query_orbuculum(query_text)
    console.print(f'[[bold green]Answer[/bold green]]\n{response_answer}')


@cli.command(help='Recharge the orbuculum with new PDF documents.')
@click.option('--reset', is_flag=True, help='Reset the orbuculum.')
@click.pass_context
def recharge(ctx, **kwargs):
    if kwargs.get('reset'):
        console.print('Resetting...')
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    console.print(f'Loaded {len(documents)} documents.')
    if documents and ctx.obj.get('DEBUG'):
        console.rule('Documents')
        console.print(documents)

    chunks = split_documents(documents)
    console.print(f'Split into {len(chunks)} chunks.')
    if chunks and ctx.obj.get('DEBUG'):
        console.rule('Chunks')
        console.print(chunks)

    add_to_chroma(chunks)


if __name__ == '__main__':
    cli()
