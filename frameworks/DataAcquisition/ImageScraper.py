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

    # Drop rows that have scientificName of NaN:
    num_samples_pre_drop = df_meta.shape[0]
    df_meta = df_meta.dropna(axis=0, how='any', subset=['scientificName'])
    if args.verbose:
        print('Warning: Data Lost! Dropped %d records from df_meta that had no discernible scientificName.'
              % (num_samples_pre_drop - df_meta.shape[0]))

    # Drop rows that have no image URLS in either goodQualityAccessURI, accessURI, associatedSpecimenReference, or thumbnailAccessURI:
    num_samples_pre_drop = df_meta.shape[0]
    df_meta = df_meta.dropna(axis=0, how='all',
                             subset=['accessURI', 'goodQualityAccessURI', 'identifier', 'associatedSpecimenReference'])
    if args.verbose:
        if (num_samples_pre_drop - df_meta.shape[0]) > 0:
            print('Warning: Data Lost! Dropped %d records from df_meta that had no associated image URL.'
                  % (num_samples_pre_drop - df_meta.shape[0]))
    return df_meta


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
    pool = ThreadPoolExecutor(max_workers=6)
    # TODO: Need to iterate over every row of the dataframe. If the accessURI is not null use that.
    #   If the accessURI is null try goodQualityAccessURI. If that is null try 'identifier' if that is null try:
    #   associatedSpecimenReference. If that is null then it shouldn't still be in the dataframe at this point.
    targets, urls = list(zip(df_meta.scientificName, df_meta.accessURI))
