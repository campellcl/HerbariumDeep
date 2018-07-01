"""
WebScraper.py
A urllib and Scrapy Selector based web scraper.
"""

import os
import argparse
from collections import OrderedDict
from scrapy import Selector
from scrapy.http import HtmlResponse
import urllib3
import certifi
import requests
import xml.etree.ElementTree as ET
import lxml
import pandas as pd
import numpy as np
import json
import zipfile, io

'''
Command line argument parsers:
'''

parser = argparse.ArgumentParser(description='SERNEC web scraper command line interface.')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-s', '--source-url', dest='source_url', metavar='URL',
                    default='https://bisque.cyverse.org/data_service/image?value=*/iplant/home/shared/sernec/*',
                    help='Source URL for web scraper.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help='Enable verbose print statements (yes, no)?')


def init_storage_dir(inst, code):
    """
    init_storage_dir: Initializes a storage directory and metadata skeletons for the provided institution. Scraped web content
        regarding the provided institution will be stored here.
    :param inst: The institution to create a directory for (the name of the herbarium).
    :param code: The unique identifier for the provided institution (the code associated with the herbarium). This value
        is also used to assemble web URL's during web scraping.
    :return:
    """
    inst_store_dir_path = args.STORE + '\\' + code
    # Check to see if storage directory exists:
    if not os.path.isdir(inst_store_dir_path):
        if verbose:
            print('Storage directory at: %s for institution: %s with code: %s does not currently exist. Creating.' % (inst_store_dir_path, inst, code))
        # Create a new directory to store this institution's metadata in:
        os.mkdir(inst_store_dir_path)
    if verbose:
        print('Storage directory: %s for institution: %s with code: %s already exists. No need to create new one.' % (inst_store_dir_path, inst, code))
    # Ensure the storage directory is writeable:
    if not os.access(inst_store_dir_path, os.W_OK):
        print('This script does not have write permissions for institution code: %s\'s supplied directory: %s. Terminating.' % (code, inst_store_dir_path))
        exit(-1)
    # Check to see if metadata file is present:
    if not os.path.isfile(inst_store_dir_path + '/metadata.csv'):
        if verbose:
            print('No existing metadata found for institution: %s with code: %s. Creating a new metadata file.' % (inst, code))
        with open(inst_store_dir_path + '/metadata.csv', 'w') as fp:
            fp.write('institution,code\n"%s",%s\n' % (inst, code))


def main():
    """
    main: Scrapes and assembles a meta dataframe containing the collection identifiers for every SERNEC collection and
        other additional information for each collection such as:
        * Collection Identifier (collid): A three digit numeric unique identifier for the collection.
        * Institution Code (inst): The institution code associated with the collection (i.e. BOON).
        * Description (desc): The full name of the collection (i.e. Appalachian State University Herbarium).
        * Emllink (emllink): A link to the sernec portal containing collection stats and more metadata.
        * Darwin Core Archive (dwca): A link to the metadata for this collection in the form of a DwC-A export.
    :return:
    """
    # Scrapes the collection identifiers for every SERNEC collection.
    df_collids = pd.DataFrame(columns={'collid', 'inst', 'desc', 'emllink', 'dwca'})
    num_added = 0
    num_rejected = 0
    # Instantiate http object per urllib3:
    http = urllib3.PoolManager(
        cert_reqs = 'CERT_REQUIRED',
        ca_certs = certifi.where()
    )
    # Obtain list of SERNEC collection identifiers (collids):
    if args.verbose:
        print('Issuing GET request to global source URL: %s' % args.source_url)
    collids_response = http.request('GET', 'http://sernecportal.org/portal/collections/datasets/rsshandler.php')
    if args.verbose:
        print('Received HTTP response code: %d' % collids_response.status)
        # print(collids_response.data)
    rss_root = ET.fromstring(collids_response.data)
    root = rss_root.getchildren()[0]
    for child in root:
        if child.tag == 'item':
            collid = int(child.attrib['collid'])
            if args.verbose:
                print('Parsing XML response for child %s' % collid)
            for property in child.getchildren():
                if property.tag == 'title':
                    title = property.text
                elif property.tag == 'description':
                    description = property.text
                elif property.tag == 'emllink':
                    emllink = property.text
                elif property.tag == 'link':
                    link = property.text
            # Get the DWCA link for this collection:
            if '.zip' in link:
                if args.verbose:
                    print('\tLocated DwC-A link for COLLID: %s, INST: %s, at: %s' % (collid, title, link))
                coll_series = pd.Series(
                    {'collid': collid, 'inst': title, 'desc': description, 'emllink': emllink, 'dwca': link}
                )
                df_collids = df_collids.append(coll_series, ignore_index=True)
                num_added += 1
            else:
                # NOTE: Tried web scraping for DwC-A, but if it isn't present under 'link' than the data is a EML File.
                if args.verbose:
                    print('\tThis collection: %s, INST: %s has no publicly available DwC-A. '
                          'This collection will be omitted from the global data frame.' % (collid, title))
                num_rejected += 1
    print('Metadata Scraping Completed. Obtained %d collections with accessible DwC-A\'s. '
          'Discarded %d collections without accessible DwC-A\'s.' % (num_added, num_rejected))
    return df_collids


def download_and_extract_zip_files(df_collids):
    """
    download_and_extract_zip_files: Goes through every collection in df_collids and downloads the DwC-A zip file for the
        collection extracting all files to the respective directory.
    :param df_collids: The global metadata dataframe containing SERNEC collections.
    :return None: Upon completion, local directories will be created for each collection which will house said
        collection's extracted DwC-A zip files.
    """
    # Download the DwC-A zip file for every collection:
    for index, row in df_collids.iterrows():
        # Create a storage directory for this collection if it doesn't already exist:
        write_dir = args.STORE + '/' + row['inst']
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
            # Download the DwC-A zip file:
            zip_response = requests.get(row['dwca'])
            if zip_response.status_code == 200:
                if args.verbose:
                    print('Downloaded DwC-A zipfile for COLLID: %s, INST: %s, with HTTP response 200 (OK)'
                          % (row['collid'], row['inst']))
                # Convert raw byte response to zip file:
                dwca_zip = zipfile.ZipFile(io.BytesIO(zip_response.content))
                # Extract all the zip files to the specified directory.
                if args.verbose:
                    print('Extracting zip files to relative directory: %s' % write_dir)
                    dwca_zip.printdir()
                dwca_zip.extractall(path=write_dir)
            else:
                if args.verbose:
                    print('Data Lost! Failed to download DwC-A zipfile for COLLID: %s, INST: %s, with HTTP response: %s'
                          % (row['collid'], row['inst'], zip_response.status_code))
                    print('Removing empty directory: %s' % write_dir)
                    os.rmdir(write_dir)


def aggregate_occurrences_and_images():
    """
    aggregate_occurrences_and_images: Aggregates data found in images.csv and occurrences.csv into a single data frame
        for each subdirectory of args.STORE.
    :return:
    """
    for i, (subdir, dirs, files) in enumerate(os.walk(args.STORE)):
        # print(subdir)
        print(i, subdir)
        if i != 0:
            # Ignore zero (the root directory \SERNEC).
            with open(subdir + '/occurrences.csv', 'r') as fp:
                df_occurr = pd.read_csv(fp)
            with open(subdir + '/images.csv', 'r') as fp:
                df_imgs = pd.read_csv(fp)
            # TODO: Concat df_occurr and df_imgs into a single metadata dataframe. Discard unwanted columns.
            # df_meta = pd.concat(...)
            pass



if __name__ == '__main__':
    # Declare global vars:
    global args, verbose
    args = parser.parse_args()
    verbose = args.verbose
    # Ensure storage directory exists:
    if not os.path.isdir(args.STORE):
        print('Specified data storage directory %s does not exist! Terminating.' % args.STORE)
        exit(-1)
        # Ensure storage directory is writeable:
        if not os.access(args.STORE, os.W_OK):
            print('This script does not have write permission for the supplied directory %s. Terminating.' % args.STORE)
            exit(-1)
    # Check existence of global metadata data frame:
    if not os.path.isfile('../../../data/SERNEC/df_collids.pkl'):
        # Perform first pass of global metadata scrape. Obtain collection codes and DwC-A URLs for all collections:
        df_collids = main()
        # Save the dataframe to the hard drive.
        df_collids.to_pickle(path='../../../data/SERNEC/df_collids.pkl')
    else:
        if args.verbose:
            print('Global metadata dataframe \'df_collids\' already exists. Will not re-scrape unless deleted.')
        df_collids = pd.read_pickle('../../../data/SERNEC/df_collids.pkl')
    ''' Uncomment the following method call to attempt a re-download of DwC-A's for empty collection directories '''
    # download_and_extract_zip_files(df_collids)
    ''' Uncomment the following method call to re-aggregate csv data for each collection's local directory '''
    aggregate_occurrences_and_images()
    pass
