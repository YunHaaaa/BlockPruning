import unittest
import regex as re

# From project root dir:
# python -m unittest

class DatasetUtilsTest(unittest.TestCase):

    def setUp(self) -> None:

        self.model_name = 'distilbert-base-uncased'
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # noqa E501


if __name__ == '__main__':
    unittest.main()
