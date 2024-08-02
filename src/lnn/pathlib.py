import json
from pathlib import Path as _OrigPath


class Path(_OrigPath):
    def read_json(self, *args, **kwargs) -> dict | list:
        """Read a JSON file and returns its Python-representation.

        Returns:
            dict | list: JSON Data
        """
        return json.loads(self.read_text(*args, **kwargs))

    def write_json(self, *args, **kwargs) -> None:
        """Serializes a object as JSON and writes it to disk."""
        indent = kwargs.pop("indent", 2)
        self.write_text(json.dumps(args[0], indent=indent), *args[1:], **kwargs)
