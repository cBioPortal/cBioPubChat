import requests
from diskcache import Cache
import os
import argparse
import json

# Ensure cache directory exists
os.makedirs('cache', exist_ok=True)
cache = Cache('cache')


def check_pmcid(pmid, email="anuvijay126@gmail.com"):
    key = f"pmcid_{pmid}"
    cached_value = cache.get(key)

    if cached_value is not None and not isinstance(cached_value, dict):
        print(f" {pmid}: Unexpected cache format")
        del cache[key]
        cached_value = None

    if cached_value:
        print(f" {pmid}: {cached_value['pmcid'] or 'Not found'}")
        return cached_value

    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": pmid,
        "format": "json",
        "tool": "return_pmcid",
        "email": email
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "records" in data and data["records"]:
            record = data["records"][0]
            pmcid_data = {
                "doi": record.get("doi"),
                "pmid": record.get("pmid", pmid),
                "pmcid": record.get("pmcid")
            }
        else:
            pmcid_data = {
                "doi": None,
                "pmid": pmid,
                "pmcid": None
            }

        cache[key] = pmcid_data
        print(f" {pmid}: {pmcid_data['pmcid'] or 'Not found'}")
        return pmcid_data

    except requests.exceptions.RequestException as e:
        print(f" Error looking up PMCID for {pmid}: {e}")
        pmcid_data = {
            "doi": None,
            "pmid": pmid,
            "pmcid": None
        }
        cache[key] = pmcid_data
        return pmcid_data


def process_pmid_file(input_file, email, output_file=None):
    with open(input_file, 'r') as f:
        pmids = [line.strip().strip('"') for line in f if line.strip()]

    found = []
    not_found = []

    for pmid in pmids:
        result = check_pmcid(pmid, email=email)
        if result["pmcid"]:
            found.append(result)
        else:
            not_found.append(result)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(found, f, indent=2)
        print(f"\n Found entries written to: {output_file}")

        no_pmcid_file = os.path.splitext(output_file)[0] + "_no_pmcid.json"
        with open(no_pmcid_file, 'w') as nf:
            json.dump(not_found, nf, indent=2)
        print(f"⚠️ Missing PMCID entries written to: {no_pmcid_file}")
    else:
        print(json.dumps(found, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check PMCIDs for a list of PMIDs.")
    parser.add_argument("-f", "--file", required=True, help="Path to the .txt file with PMIDs")
    parser.add_argument("-e", "--email", required=True, help="Email address for NCBI API usage")
    parser.add_argument("-o", "--output", required=False, help="Path to save output JSON")

    args = parser.parse_args()
    process_pmid_file(args.file, args.email, args.output)
