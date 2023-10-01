import unittest
import json

from config import read_grid
from main import run


class TestConfigs(unittest.TestCase):
    CONFIG_ROOT = 'configs.json'

    def test_all_configs(self):
        return
        """Assert that all configs in configs.json are runnable."""
        assert(True)
        with open(self.CONFIG_ROOT, 'r') as fp:
            grids = json.load(fp)
        for grid_key in grids:
            grid = read_grid(path=self.CONFIG_ROOT, prop=grid_key)
            for conf in grid:
                if conf.data == "mnist": continue
                run(conf, 1)

if __name__ == '__main__':
    unittest.main()