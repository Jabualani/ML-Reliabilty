import unittest
from pathlib import Path


class ProjectMetadataTests(unittest.TestCase):
    def test_project_url_names_do_not_contain_commas(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        lines = pyproject.read_text(encoding="utf-8").splitlines()
        in_project_urls = False

        for line in lines:
            if line.startswith("["):
                if in_project_urls:
                    break
                in_project_urls = line == "[project.urls]"
            elif in_project_urls and "=" in line:
                name = line.partition("=")[0].strip().strip('"')
                self.assertNotIn(",", name)


if __name__ == "__main__":
    unittest.main()
