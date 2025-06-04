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
    dataset_tsv_path = os.path.join(script_dir, "dataset.tsv")
    result_tsv_path = os.path.join(script_dir, "result.tsv")
    with open(dataset_tsv_path, newline='', encoding='utf-8') as f_in, open(result_tsv_path, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in, delimiter='\t')
        fieldnames = ['Question', 'Reference Answer', 'Actual Answer', 'Correct?', 'Score', 'Reasoning']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in reader:
            question = row['Question']
            reference_answer = row['Reference Answer']
            actual_answer = run_rag(question)
            result = labeled_criteria_evaluator.evaluate_strings(
                input=question,
                reference=reference_answer,
                prediction=actual_answer)
            write_dict = {
                    'Question': question,
                    'Reference Answer': reference_answer,
                    'Actual Answer': actual_answer,
                    'Correct?': result['value'],
                    'Score': result['score'],
                    'Reasoning': result['reasoning'],
            }
            writer.writerow(write_dict)

if __name__ == "__main__":
    main()
