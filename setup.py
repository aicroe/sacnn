from setuptools import setup, find_packages

setup(
    name='sacnn',
    version='0.1.2',
    # url='',
    # project_urls={ },
    author='Diego Garcia',
    author_email='qtimpot@gmail.com',
    description='A Sentiment Analysis Convolutional Neural Network',
    keywords=[
        'machine learning',
        'deep learning',
        'convolutional neural network',
        'sentiment analysis',
        'thesis',
        'final project',
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow',
        'matplotlib',
        'pandas',
        'gensim',
        'flask',
    ],
    dependency_links=[
        'git+git@github.com:aicroe/mlscratch.git@master#egg=mlscratch'
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'sacnn_train = sacnn.train:main',
            'sacnn_process_data = sacnn.process_data:main',
            'sacnn_reduce_labels = sacnn.reduce_labels:main',
            'sacnn_eval = sacnn.eval:main',
            'sacnn_server = sacnn.server:main'
        ],
    },
)
