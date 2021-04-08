import unittest
import tensorflow as tf
import dlf.core
from dlf.data_generators.fs_pair_reader import FsRandomUnpairedReader


class TestFsUnpairedReader(unittest.TestCase):
    preprocessing = {'resize': {'width': 512, 'height': 512}}

    def setUp(self):
        self.reader = FsRandomUnpairedReader(
            'data/imagenet_like/Mug/*.jpg', 'data/imagenet_like/Bottle/*.jpg', preprocess_list=self.preprocessing)

    def test_init(self):
        with self.assertRaises(TypeError):
            FsRandomUnpairedReader()

        with self.assertRaises(FileNotFoundError):
            FsRandomUnpairedReader(
                paths_lhs='data', paths_rhs='datas')

        with self.assertRaises(FileNotFoundError):
            FsRandomUnpairedReader(
                paths_lhs='datas', paths_rhs='data')

        with self.assertRaises(FileNotFoundError):
            FsRandomUnpairedReader(
                paths_lhs='dlf/*.txt', paths_rhs='data')

        with self.assertRaises(FileNotFoundError):
            FsRandomUnpairedReader(
                paths_lhs='data', paths_rhs='dlf/*.txt')

        with self.assertRaises(FileNotFoundError,
                               msg='Should raise an because the second path in paths_lhs doesn\'t exists'):
            FsRandomUnpairedReader(
                paths_lhs=['dlf/*', 'abc'], paths_rhs=['dlf/*'])

        with self.assertRaises(FileNotFoundError,
                               msg='Should raise an because the second path in paths_rhs doesn\'t exists'):
            FsRandomUnpairedReader(
                paths_lhs=['dlf/*'], paths_rhs=['dlf/*', 'abc'])

        with self.assertRaises(ValueError,
                               msg='Should raise a ValueError because resize preprocessing is missing'):
            FsRandomUnpairedReader(
                paths_lhs='dlf/*', paths_rhs='dlf/*')

    def test_iter(self):
        item = next(iter(self.reader.dataset))
        self.assertTrue(len(item) == 2)
        self.assertTrue('x_batch' in item)
        self.assertTrue('y_batch' in item)

        for batch in [4, 32, 128]:
            lhs, rhs = next(iter(self.reader.dataset.batch(batch))).values()
            self.assertTrue(lhs.shape[0] == batch)
            self.assertTrue(rhs.shape[0] == batch)
