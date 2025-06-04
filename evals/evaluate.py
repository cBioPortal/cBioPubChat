#!/usr/bin/env python3

import csv
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsv_path = os.path.join(script_dir, "dataset.tsv")
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            print(row)

if __name__ == "__main__":
    main()
