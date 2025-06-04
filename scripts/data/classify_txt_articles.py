import os
import json

def is_full_text_by_structure(text):
    """Heuristic: consider a file full-text if 2+ major section headings are present."""
    headings = ['introduction', 'methods', 'results', 'discussion', 'conclusion']
    lower_text = text.lower()
    found = sum(1 for h in headings if h in lower_text)
    return found >= 2

def classify_txt_folder(txt_dir):
    """Classify each .txt file in a directory as full_text or abstract_only based on structure."""
    results = {}
    counts = {"full_text": 0, "abstract_only": 0}

    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            path = os.path.join(txt_dir, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if is_full_text_by_structure(content):
                classification = "full_text"
                counts["full_text"] += 1
            else:
                classification = "abstract_only"
                counts["abstract_only"] += 1

            results[filename] = classification
            print(f"{filename}: {classification}")

    results["_summary"] = {
        "total_files": counts["full_text"] + counts["abstract_only"],
        "full_text": counts["full_text"],
        "abstract_only": counts["abstract_only"]
    }

    return results

if __name__ == "__main__":
    input_dir = "/Users/bsatravada/Desktop/cBioPubChat/data/data_raw/txt"  # Directory of .txt files
    output_json = "text_classification_by_structure.json"

    print(f" Analyzing files in: {input_dir}")
    classified = classify_txt_folder(input_dir)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(classified, f, indent=4)

    summary = classified["_summary"]
    print(f"\n Classification Summary:")
    print(f"Total files     : {summary['total_files']}")
    print(f"Full-text files : {summary['full_text']}")
    print(f"Abstract-only   : {summary['abstract_only']}")
    print(f"\n Results saved to: {output_json}")
