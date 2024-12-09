from pathlib import Path
import click


def merge_codebase(
    root_dir: str | Path, include_suffixes: tuple[str] = ("py",), recursive: bool = True
) -> str:
    """Merge all code files in the given directory into a single string."""
    cwd = Path().cwd()
    root_dir = Path(root_dir)
    code = ""
    pattern = "**/*" if recursive else "*"
    for path in root_dir.glob(pattern):
        path = path.resolve()
        if path.is_file() and path.suffix[1:] in include_suffixes:
            name = path.relative_to(cwd)
            code += f"### File: {name}\n ##" + path.read_text() + "\n\n"
    return code


@click.command()
@click.argument("root_dir", type=click.Path(exists=True))
@click.option(
    "--include-suffix",
    "-i",
    default=["py"],
    multiple=True,
    help="File suffixes to include in the merge.",
)
@click.option("--recursive", "-r", is_flag=True, help="Recursively search for files.")
def cli_main(root_dir: str, include_suffix: tuple[str], recursive: bool) -> None:
    """Merge all code files in the given directory into a single string."""
    code = merge_codebase(root_dir, include_suffix, recursive)
    print(code)
