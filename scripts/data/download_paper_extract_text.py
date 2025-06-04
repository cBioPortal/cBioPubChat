import os
import argparse
from bs4 import BeautifulSoup
from download_pmc_s3 import download_pmc_s3
from indra_nxml_extraction import id_lookup, extract_text


def read_and_extract_xml_data(filename):
    """Load and extract text from an XML file.

    This function reads an XML file, parses it using BeautifulSoup,
    and extracts plain text content using the extract_text function.

    Parameters
    ----------
    filename : str
        Path to the XML file to be processed.

    Returns
    -------
    str
        Extracted plain text content from the XML file.
    """
    with open(filename, 'r') as f:
        data = f.read()
    xml_data = BeautifulSoup(data, "xml")
    text = extract_text(xml_data)
    return text


def download_paper_extract_text(pmid, xml_dir="data/data_raw/xml", txt_dir="data/data_raw/txt"):
    """Download and extract text from a PubMed paper.

    This function takes a PubMed ID, looks up the corresponding PubMed Central ID,
    downloads the XML file from PMC S3 storage, extracts the text content,
    and saves it to a text file in the specified output directory.

    Parameters
    ----------
    pmid : str
        PubMed ID of the paper to download and process.
    xml_dir : str, optional
        Directory to store downloaded XML files (default: data/data_raw/xml).
    txt_dir : str, optional
        Directory to store extracted text files (default: data/data_raw/txt).

    Returns
    -------
    None
        This function does not return a value but saves the extracted text
        to a file.
    """
    try:
        pmcid = id_lookup(pmid).get("pmcid")
        print(pmcid)
        try:
            download_pmc_s3(pmcid, output_dir=xml_dir, cache_dir=xml_dir)
            filename_xml = os.path.join(xml_dir, f"{pmcid}.xml")

            try:
                # Read and extract from XML data
                text = read_and_extract_xml_data(filename_xml)

                # Write a TXT file of extracted XML content
                filename_txt = os.path.join(txt_dir, f"{pmcid}.txt")

                with open(filename_txt, 'w') as f:
                    f.write(str(text))

            except Exception as e:
                print(f"ERROR: Reading and extracting PMCID: {pmcid}: {e}")
        except Exception as e:
            print(f"ERROR: PMCID cannot be downloaded: {pmcid}: {e}")

    except Exception as e:
        print(f"ERROR : PMID: {pmid} has no PMCID: {e}")


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Download PubMed paper and extract text")
    parser.add_argument('-p', '--pmid', default="36517593",
                        help='PubMed ID of the paper to download (default: 36517593)')
    parser.add_argument('-x', '--xml-dir', default="data/data_raw/xml",
                        help='Directory to store XML files (default: data/data_raw/xml)')
    parser.add_argument('-t', '--txt-dir', default="data/data_raw/txt",
                        help='Directory to store TXT files (default: data/data_raw/txt)')

    args = parser.parse_args()

    # Download paper using provided PMID and directories
    download_paper_extract_text(args.pmid, args.xml_dir, args.txt_dir)
