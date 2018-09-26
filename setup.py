from setuptools import setup

setup(
      name='tensorbob',
      version='0.0.1',
      description="TensorFlow tools",
      keywords='python tensorflow',
      author='ZhangYiYang',
      author_email='irvingzhang0512@gmail.com',
      url='https://github.com/irvingzhang0512/tensorbob',
      packages=['tensorbob', 'tensorbob.utils', 'tensorbob.dataset',
                'tensorbob.evaluating', 'tensorbob.models', 'tensorbob.training'],
      install_requires=['tensorflow', 'numpy']
)
