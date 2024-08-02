import json


def patch_path():
    def read_json(self, *args, **kwargs):
        return json.loads(self.read_text(*args, **kwargs))

    setattr(Path, "read_json", read_json)

    def write_json(self, *args, **kwargs):
        indent = kwargs.pop("indent", 2)
        self.write_text(json.dumps(args[0], indent=indent), *args[1:], **kwargs)

    setattr(Path, "write_json", write_json)


patch_path()
