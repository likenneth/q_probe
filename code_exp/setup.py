from setuptools import setup, find_packages

setup(
    name="mcts_llm",  # Replace with your own package name
    version="0.1",  # Replace with your package version
    packages=find_packages(),
    # entry_points={
    #     "console_scripts": [
    #         "evaluate_functional_correctness = mcts-llm.mcts_llm.code_gen.human_eval.evaluate_functional_correctness",
    #     ]
    # }
)