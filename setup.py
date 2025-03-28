from setuptools import setup, find_packages

setup(
    name="web_rag",
    version="0.1",
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=[
        'langchain-openai>=0.0.5',
        'langchain-core>=0.1.17',
        'langgraph>=0.0.20',
        'faiss-cpu>=1.7.4',
        'selenium>=4.16.0',
        'duckduckgo-search>=4.1.1',
        'python-dotenv>=1.0.0',
        'openai>=1.12.0',
        'twilio',
        'annoy',
        'webdriver-manager'
    ],
)
