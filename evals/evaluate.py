#!/usr/bin/env python3

import csv
import os
import sys

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import EvaluatorType

script_dir = os.path.dirname(os.path.abspath(__file__))

backend_dir = os.path.join(script_dir, "../backend")
sys.path.insert(0, str(backend_dir))
from rag import run_rag

correctness_criteria = {
    "correctness": "Given the reference, does the submission convey the same primary information without introducing any contradictions?"
}

llm = ChatOpenAI(temperature=0, model="gpt-4")

labeled_criteria_evaluator = load_evaluator(
    EvaluatorType.LABELED_CRITERIA, criteria=correctness_criteria, llm=llm)

def main():
    tsv_path = os.path.join(script_dir, "dataset.tsv")
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            question, reference_answer = row
            actual_answer = run_rag(question)
            result = labeled_criteria_evaluator.evaluate_strings(
                input=question,
                reference=reference_answer,
                prediction=actual_answer)
            result['question'] = question
            result['reference_answer'] = reference_answer
            result['actual_answer'] = actual_answer
            print(result)

if __name__ == "__main__":
    main()
