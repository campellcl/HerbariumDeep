"""
ImageScraper.py
A multi-threaded image scraper.
"""

__author__ = 'Chris Campell'
__created__ = '7/7/2018'

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from time import time
import argparse
import pandas as pd
import urllib3, certifi, requests

'''
Command Line Argument Parsers: 
'''

parser = argparse.ArgumentParser(description='SERNEC web scraper command line interface.')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')


def main():
    if os.path.isfile(args.STORE + '\collections\df_meta.pkl'):
        df_meta = pd.read_pickle(path=args.STORE  + '\collections\df_meta.pkl')
        if df_meta.empty:
            print('ERROR: Could not read df_meta.pkl. Re-run the metadata WebScraper.')
        else:
            print('Loaded metadata dataframe: \'df_meta.pkl\'.')
    else:
        print('ERROR: Could not locate df_meta.pkl on the local hard drive at %s\collections\. '
              'Have you run the metadata WebScraper?' % args.STORE)
        exit(-1)
    return df_meta


def create_storage_dirs(targets_and_urls):
    """
    create_storage_dirs: Creates storage directories for every unique class under args.STORE\images.
    :param targets_and_urls: A list of target class labels and the associated image URLs.
    :return:
    """
    write_dir = args.STORE + '\images'
    if not os.path.isdir(write_dir):
        print('\tThis script detects no existing image folder: %s. Instantiating class storage directories...')
        os.mkdir(write_dir)
    print('\tNow instantiating class image storage directories...')
    pruned_targets_and_urls = targets_and_urls.copy()
    num_failed_dirs = 0
    num_created_dirs = 0
    for i, (target, url) in enumerate(targets_and_urls):
        target_dir = write_dir + '\%s' % target
        if os.path.isdir(target_dir):
            pass
            # if args.verbose:
            #     print('\t\tTarget class label directory %s already exists.' % target_dir)
        else:
            if args.verbose:
                print('\t\tCreating storage dir %s for target %s' % (target_dir, target))
            try:
                os.mkdir(target_dir)
                num_created_dirs += 1
            except OSError as err:
                print('\t\tERROR: Received the following error during directory creation:')
                print('\t\t\t %s' % err)
                pruned_targets_and_urls.remove((target, url))
                num_failed_dirs += 1

    print('\tFinished instantiating target label directories. There were %d directories created this '
          'time and %d failed attempts.' % (num_created_dirs, num_failed_dirs))
    return pruned_targets_and_urls


if __name__ == '__main__':
    # Declare global vars:
    global args, verbose
    args = parser.parse_args()
    verbose = args.verbose
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    if args.verbose:
        print('Loading metadata dataframe. This file is several GBs, please be patient...')
    df_meta = main()

    # Discard scientificNames that contain the unicode replacement character '?':
    # len(df_meta[df_meta['scientificName'].str.contains('\?')])
    df_meta = df_meta[~df_meta['scientificName'].str.contains('\?')]

    targets_and_urls = None
    # If there are no null values in goodQualityAccessURI use that for a URL:
    if not df_meta.goodQualityAccessURI.isnull().values.any():
        targets_and_urls = list(zip(df_meta.scientificName, df_meta.goodQualityAccessURI))

    # Create storage directories for every class:
    targets_and_urls = create_storage_dirs(targets_and_urls)

    http = urllib3.PoolManager(
        cert_reqs = 'CERT_REQUIRED',
        ca_certs = certifi.where()
    )
    for i in range(10):
        download_path = args.STORE + '\images\%s' % targets_and_urls[i][0]
        # download_path / os.path.basename(link)
        logger.info('Downloading %s', targets_and_urls[i][1])
         # Instantiate http object per urllib3:
        ts = time()
        response = http.request('GET', targets_and_urls[i][1])
        logging.info('Took %s seconds', time() - ts)
        if not response.status == 200:
            print('ERROR: Recieved http response %d' % response.status)
        else:
            with download_path.open('wb') as fp:
                fp.write(response.data)



    # Instantiate thread pool:
    # executor = ThreadPoolExecutor(max_workers=6)
    pass
