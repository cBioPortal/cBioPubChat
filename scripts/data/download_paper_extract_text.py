import os
import argparse
import requests
from bs4 import BeautifulSoup
from lxml import etree # For robust XML parsing
import logging
import time
import json # Added for JSON file processing

# --- Logging Setup ---
# Setup logging to file and console
log_file = "bulk_pmc_download.log"
logging.basicConfig(
    filename=log_file,
    filemode='a', # Append to log file
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# --- NCBI E-utilities Functions ---

def id_lookup(pmid, email, tool_name="bulk_pmc_downloader"):
    """
    Converts a PubMed ID (PMID) to a PubMed Central ID (PMCID) using NCBI's ID Converter API.
    
    Args:
        pmid (str): The PubMed ID.
        email (str): User's email address (mandatory for NCBI E-utilities).
        tool_name (str): Name of the tool making the request.

    Returns:
        str or None: The corresponding PMCID if found, otherwise None.
    """
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": pmid,
        "format": "json",
        "tool": tool_name,
        "email": email
    }
    try:
        # NCBI rate limits: 3 requests per second without API key, 10 with.
        time.sleep(0.4) # Adhere to rate limits
        response = requests.get(base_url, params=params, timeout=20) # Increased timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        data = response.json()
        if "records" in data and data["records"]:
            record = data["records"][0]
            if "pmcid" in record:
                logging.info(f"PMID {pmid} -> PMCID {record['pmcid']}")
                return record['pmcid']
            else:
                logging.warning(f"No PMCID found for PMID {pmid} in API response: {record.get('status', 'N/A')}")
                return None
        logging.warning(f"No records found for PMID {pmid} in idconv API response.")
        return None
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error looking up PMCID for PMID {pmid}: {http_err} - Response: {response.text[:200] if response else 'No response'}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error looking up PMCID for PMID {pmid}: {e}")
    except json.JSONDecodeError as json_err: # Catch JSON decoding errors
        logging.error(f"JSON decode error for PMID {pmid} idconv response: {json_err} - Response: {response.text[:200] if response else 'No response'}")
    return None

def download_pmc_xml_direct(pmc_id, output_dir, email, tool_name="bulk_pmc_downloader"):
    """
    Downloads the full XML of a paper from PMC using E-utilities.
    
    Args:
        pmc_id (str): The PubMed Central ID (e.g., "PMC123456").
        output_dir (str): Directory to save the XML file.
        email (str): User's email address.
        tool_name (str): Name of the tool making the request.

    Returns:
        str or None: Path to the downloaded XML file if successful, otherwise None.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize pmc_id for filename, removing "PMC" prefix if present for consistency
    filename_pmc_id = pmc_id.replace("PMC", "").strip()
    file_path = os.path.join(output_dir, f'PMC{filename_pmc_id}.xml')

    if os.path.exists(file_path):
        logging.info(f"XML file for {pmc_id} already exists at {file_path}, skipping download.")
        return file_path

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    # Efetch uses the numeric part of PMCID or the full PMCID
    params = {
        "db": "pmc",
        "id": pmc_id, 
        "rettype": "full",
        "retmode": "xml",
        "tool": tool_name,
        "email": email
    }

    try:
        logging.info(f"Attempting to download XML for {pmc_id}...")
        time.sleep(0.4) # Adhere to rate limits
        response = requests.get(base_url, params=params, timeout=60) # Increased timeout for large files
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if not ('xml' in content_type or response.text.strip().startswith('<?xml')):
            logging.warning(f"Response for {pmc_id} does not appear to be XML. Content-Type: {content_type}. Snippet: {response.text[:200]}")
            # Check for common error messages or unavailable article pages
            if "unavailable" in response.text.lower() or "not found" in response.text.lower():
                 logging.error(f"Article {pmc_id} may be unavailable or access restricted.")
            return None

        # Pretty-print and save XML
        try:
            # Use response.content for bytes, then decode to string for etree
            xml_bytes = response.content
            parsed_xml = etree.fromstring(xml_bytes) 
            pretty_xml = etree.tostring(parsed_xml, pretty_print=True, encoding='utf-8').decode('utf-8')
        except etree.XMLSyntaxError as xml_err:
            logging.error(f"XMLSyntaxError parsing XML for {pmc_id}: {xml_err}. Saving raw content instead.")
            pretty_xml = response.text # Save raw text if parsing fails

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        logging.info(f"Successfully downloaded and saved XML for {pmc_id} to {file_path}")
        return file_path

    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error downloading XML for {pmc_id}: {errh} - Response: {response.text[:500] if response else 'No response'}")
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Connection Error downloading XML for {pmc_id}: {errc}")
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error downloading XML for {pmc_id}: {errt}")
    except requests.exceptions.RequestException as err:
        logging.error(f"An unexpected error occurred downloading XML for {pmc_id}: {err}")
    return None

# --- Text Extraction ---

def extract_text_from_xml(xml_file_path):
    """
    Extracts plain text content from a PMC XML file, including article title and abstract.
    """
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        soup = BeautifulSoup(xml_content, "xml")
        text_parts = []

        # --- Extract Article Title ---
        title_tag = soup.find("article-title")
        if title_tag:
            article_title = title_tag.get_text(strip=True)
            text_parts.append(article_title.upper())
            text_parts.append("")  # blank line after title

        # --- Extract Abstract ---
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            text_parts.append("ABSTRACT")
            for child in abstract_tag.find_all(['p', 'sec'], recursive=False):
                if child.name == 'p':
                    text_parts.append(child.get_text(separator=' ', strip=True))
                elif child.name == 'sec':
                    sec_title = child.find('title', recursive=False)
                    if sec_title:
                        text_parts.append(sec_title.get_text(separator=' ', strip=True).upper())
                    for p in child.find_all('p', recursive=True):
                        text_parts.append(p.get_text(separator=' ', strip=True))
            text_parts.append("")

        # --- Extract Main Body ---
        body = soup.find("body")
        if body:
            for elem in body.find_all(['sec', 'p'], recursive=False):
                if elem.name == 'sec':
                    title = elem.find('title', recursive=False)
                    if title:
                        text_parts.append(title.get_text(separator=' ', strip=True).upper())
                    for p in elem.find_all('p', recursive=True):
                        text_parts.append(p.get_text(separator=' ', strip=True))
                elif elem.name == 'p':
                    text_parts.append(elem.get_text(separator=' ', strip=True))

        if not text_parts:
            logging.warning(f"No specific text extracted from {xml_file_path}. Fallback to raw text.")
            return soup.get_text(separator='\n\n', strip=True)

        return "\n\n".join(filter(None, text_parts))

    except FileNotFoundError:
        logging.error(f"File not found: {xml_file_path}")
    except Exception as e:
        logging.error(f"Failed to extract text from {xml_file_path}: {e}")
    return None


# --- Main Processing Function ---

def bulk_download_and_extract(pmid_list, xml_dir, txt_dir, email):
    """
    Processes a list of PMIDs to download XML and extract text.

    Args:
        pmid_list (list): A list of PubMed IDs (strings).
        xml_dir (str): Directory to save downloaded XML files.
        txt_dir (str): Directory to save extracted TXT files.
        email (str): User's email for NCBI E-utilities.
    """
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    succeeded_pmids = []
    failed_pmids_no_pmcid = []
    failed_pmids_xml_download = []
    failed_pmids_text_extraction = []

    # Remove duplicates from pmid_list while preserving order (if Python 3.7+)
    # For older Python, use list(dict.fromkeys(pmid_list))
    unique_pmid_list = list(dict.fromkeys(pmid_list))
    if len(unique_pmid_list) < len(pmid_list):
        logging.info(f"Removed {len(pmid_list) - len(unique_pmid_list)} duplicate PMIDs from the input list.")
    
    logging.info(f"Starting bulk processing for {len(unique_pmid_list)} unique PMIDs.")

    for pmid in unique_pmid_list:
        pmid = str(pmid).strip() # Ensure pmid is a string and stripped
        if not pmid:
            logging.warning("Encountered an empty PMID. Skipping.")
            continue
        
        logging.info(f"--- Processing PMID: {pmid} ---")
        
        pmcid = id_lookup(pmid, email)
        
        if not pmcid:
            logging.warning(f"Could not find PMCID for PMID: {pmid}. Skipping.")
            failed_pmids_no_pmcid.append(pmid)
            continue
        
        # Define PMCID specific filenames
        # Ensure PMCID in filename starts with PMC for consistency
        filename_base = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        # Further sanitize filename_base in case PMCID itself has odd characters (though unlikely)
        filename_base = "".join(c if c.isalnum() else "_" for c in filename_base) # Replace non-alphanumeric with underscore

        xml_file_path = os.path.join(xml_dir, f"{filename_base}.xml")
        txt_file_path = os.path.join(txt_dir, f"{filename_base}.txt")

        # Check if TXT file already exists, if so, skip all processing for this PMCID
        if os.path.exists(txt_file_path):
            logging.info(f"TXT file for {pmcid} (from PMID {pmid}) already exists at {txt_file_path}. Skipping.")
            succeeded_pmids.append(pmid) # Count as success if final output exists
            continue

        # Download XML
        # The download function itself checks if XML exists, but we ensure TXT is the ultimate skip condition
        downloaded_xml_path = download_pmc_xml_direct(pmcid, xml_dir, email)
        
        if not downloaded_xml_path:
            logging.error(f"Failed to download XML for PMCID: {pmcid} (from PMID: {pmid}). Skipping text extraction.")
            failed_pmids_xml_download.append({"pmid": pmid, "pmcid": pmcid})
            continue
            
        # Extract Text
        extracted_text = extract_text_from_xml(downloaded_xml_path)
        
        if extracted_text:
            try:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                logging.info(f"Successfully extracted text for {pmcid} (PMID: {pmid}) and saved to {txt_file_path}")
                succeeded_pmids.append(pmid)
            except IOError as e:
                logging.error(f"IOError saving TXT file {txt_file_path} for {pmcid} (PMID: {pmid}): {e}")
                failed_pmids_text_extraction.append({"pmid": pmid, "pmcid": pmcid, "error": str(e)})
        else:
            logging.error(f"Failed to extract text for {pmcid} (PMID: {pmid}) from {downloaded_xml_path}.")
            failed_pmids_text_extraction.append({"pmid": pmid, "pmcid": pmcid, "error": "Extraction returned no text"})

    logging.info("--- Bulk Processing Summary ---")
    logging.info(f"Total unique PMIDs to process: {len(unique_pmid_list)}")
    logging.info(f"Successfully processed (TXT created or existed): {len(succeeded_pmids)}")
    logging.info(f"Failed - No PMCID found: {len(failed_pmids_no_pmcid)}")
    if failed_pmids_no_pmcid: logging.info(f"  PMIDs: {', '.join(failed_pmids_no_pmcid)}")
    logging.info(f"Failed - XML download error: {len(failed_pmids_xml_download)}")
    if failed_pmids_xml_download: logging.info(f"  Details: {failed_pmids_xml_download}")
    logging.info(f"Failed - Text extraction or save error: {len(failed_pmids_text_extraction)}")
    if failed_pmids_text_extraction: logging.info(f"  Details: {failed_pmids_text_extraction}")
    logging.info("--- End of Summary ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk download PMC XML articles and extract text from a list of PMIDs.")
    parser.add_argument('-e', '--email', required=True, help='Your email address for NCBI E-utilities.')

    # Mutually exclusive group for PMID input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pmids', nargs='+', help='A list of PubMed IDs separated by spaces.')
    group.add_argument('-f', '--pmid_file', help='Path to a text file containing one PMID per line.')
    group.add_argument('-u', '--unique_pmid_file', help='Path to a text file (e.g., unique_pmi.txt) with one PMID per line.')

    parser.add_argument('-x', '--xml_dir', default="downloaded_xml", help='Directory to store downloaded XML files (default: downloaded_xml).')
    parser.add_argument('-t', '--txt_dir', default="extracted_txt", help='Directory to store extracted TXT files (default: extracted_txt).')

    args = parser.parse_args()

    input_pmids = []

    if args.pmids:
        input_pmids = args.pmids
    elif args.pmid_file:
        try:
            with open(args.pmid_file, 'r') as f:
                input_pmids = [line.strip().strip('"') for line in f if line.strip()]
            if not input_pmids:
                logging.error(f"PMID file {args.pmid_file} is empty or contains no valid PMIDs.")
                print(f"Error: PMID file {args.pmid_file} is empty or contains no valid PMIDs.")
                exit(1)
        except FileNotFoundError:
            logging.error(f"PMID file not found: {args.pmid_file}")
            print(f"Error: PMID file not found: {args.pmid_file}")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading PMID file {args.pmid_file}: {e}")
            print(f"Error reading PMID file {args.pmid_file}: {e}")
            exit(1)
    elif args.unique_pmid_file:
        try:
            with open(args.unique_pmid_file, 'r') as f:
                input_pmids = [line.strip().strip('"') for line in f if line.strip()]
            if not input_pmids:
                logging.error(f"Unique PMID file {args.unique_pmid_file} is empty or contains no valid PMIDs.")
                print(f"Error: Unique PMID file {args.unique_pmid_file} is empty or contains no valid PMIDs.")
                exit(1)
        except FileNotFoundError:
            logging.error(f"Unique PMID file not found: {args.unique_pmid_file}")
            print(f"Error: Unique PMID file not found: {args.unique_pmid_file}")
            exit(1)
        except Exception as e:
            logging.error(f"Error reading unique PMID file {args.unique_pmid_file}: {e}")
            print(f"Error reading unique PMID file {args.unique_pmid_file}: {e}")
            exit(1)

    if not input_pmids:
        logging.info("No PMIDs provided to process after attempting all input methods.")
        print("No PMIDs provided. Exiting.")
        exit(0)

    logging.info(f"Email: {args.email}, XML Dir: {args.xml_dir}, TXT Dir: {args.txt_dir}")
    logging.info(f"Total PMIDs collected for processing: {len(input_pmids)}")

    bulk_download_and_extract(input_pmids, args.xml_dir, args.txt_dir, args.email)

    logging.info("Script finished.")
    print("Script finished. Check bulk_pmc_download.log for details.")
