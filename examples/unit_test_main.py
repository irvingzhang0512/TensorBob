import unittest
from tensorbob.dataset.base_dataset_test import BaseDatasetTest
from tensorbob.dataset.imagenet_test import ImageNetTest
from tensorbob.dataset.voc2012_test import Voc2012Test
from tensorbob.dataset.ade2016_test import Ade2016Test
from tensorbob.dataset.camvid_test import CamVidTest

from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(logging.DEBUG)


suite = unittest.TestSuite()
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(BaseDatasetTest))
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ImageNetTest))
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Voc2012Test))
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Ade2016Test))
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CamVidTest))

# 获取suite中测试用例个数
# 可以看出，测试用例可以重复添加
print(suite.countTestCases())

# 普通console runner
runner = unittest.TextTestRunner(verbosity=1)
runner.run(suite)

