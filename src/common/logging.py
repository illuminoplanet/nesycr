import os


class Logger:
    def __init__(self):
        self.log_path = None
        self._lines = []

    def set_path(self, log_path):
        self.log_path = log_path

    def log(self, title, content):
        self._lines.append(f"[{title}]\n{content}\n")
        self._save(self.log_path)

    def split_line(self):
        self._lines.append("-" * 50 + "\n")

    def _save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._to_string())

    def _to_string(self):
        return "\n".join(self._lines)


logger = Logger()
