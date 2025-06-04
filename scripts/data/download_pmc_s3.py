import os
import shutil
import logging

import boto3
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_pmc_s3(pmc_id, file_type='xml', output_dir='pmc', cache_dir='pmc', bucket_name='pmc-oa-opendata'):
    """Download PMC files from AWS S3 bucket.

    Downloads PMC files from the specified AWS S3 bucket. The function will try to
    download from different paths in the bucket if the initial attempt fails.
    If the file exists in the cache directory, it will be copied to the output directory.

    Parameters
    ----------
    pmc_id : str
        PubMed Central ID.
    file_type : str
        File type (xml or txt). Default is 'xml'.
    output_dir : str
        Output directory. Default is 'pmc'.
    cache_dir : str
        Cache directory. Default is 'pmc'.
    bucket_name : str
        S3 bucket name. Default is 'pmc-oa-opendata'.

    Returns
    -------
    None
        The function does not return a value.

    Examples
    --------
    >>> download_pmc_s3('PMC3898398')
    """

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{pmc_id}.{file_type}')
    cache_path = os.path.join(cache_dir, f'{pmc_id}.{file_type}')

    if not os.path.exists(output_path):
        if os.path.exists(cache_path):
            shutil.copy(cache_path, output_path)
        else:
            logger.info(f"Attempting to download {pmc_id}.{file_type} to {output_path}")

            try:
                file_key = f'oa_comm/{file_type}/all/{pmc_id}.{file_type}'
                s3.download_file(bucket_name, file_key, cache_path)
                shutil.copy(cache_path, output_path)
            except Exception as e:
                try:
                    file_key = f'oa_noncomm/{file_type}/all/{pmc_id}.{file_type}'
                    s3.download_file(bucket_name, file_key, cache_path)
                    shutil.copy(cache_path, output_path)
                except Exception as e:
                    try:
                        file_key = f'author_manuscript/{file_type}/all/{pmc_id}.{file_type}'
                        s3.download_file(bucket_name, file_key, cache_path)
                        shutil.copy(cache_path, output_path)
                    except Exception as e:
                        if not os.path.exists(cache_path):
                            logger.error(e)

    if os.path.exists(cache_path):
        logger.info(f"DONE: File: {output_path}")
