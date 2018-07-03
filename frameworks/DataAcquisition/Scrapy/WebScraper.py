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
        print('\tIssuing GET request to SERNEC collections source URL: %s. Please be patient...'
              % args.source_url)
    collids_response = http.request('GET', 'http://sernecportal.org/portal/collections/datasets/rsshandler.php')
    if args.verbose:
        print('\tReceived HTTP response code: %d. Now parsing xml data...' % collids_response.status)
        # print(collids_response.data)
    rss_root = ET.fromstring(collids_response.data)
    root = rss_root.getchildren()[0]
    for child in root:
        if child.tag == 'item':
            collid = int(child.attrib['collid'])
            if args.verbose:
                print('\tParsing XML response for child %s' % collid)
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
                    print('\t\tLocated DwC-A link for COLLID: %s, INST: %s, at: %s' % (collid, title, link))
                coll_series = pd.Series(
                    {'collid': collid, 'inst': title, 'desc': description, 'emllink': emllink, 'dwca': link}
                )
                df_collids = df_collids.append(coll_series, ignore_index=True)
                num_added += 1
            else:
                # NOTE: Tried web scraping for DwC-A, but if it isn't present under 'link' than the data is a EML File.
                if args.verbose:
                    print('\t\tThis collection: %s, INST: %s has no publicly available DwC-A. '
                          'This collection will be omitted from the global data frame.' % (collid, title))
                num_rejected += 1
    print('STAGE_ONE: Pipeline STAGE_ONE complete. Collection metadata downloaded and parsed successfully.'
          '\nSTAGE_ONE: Obtained %d collections with accessible DwC-A\'s. '
          'Discarded %d collections without accessible DwC-A\'s.' % (num_added, num_rejected))
    print('=' * 100)
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
        write_dir = args.STORE + '/collections/' + row['inst']
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
                    print('\tData Lost! Failed to download DwC-A zipfile for COLLID: %s, INST: %s, with HTTP response: %s'
                          % (row['collid'], row['inst'], zip_response.status_code))
                    print('\tRemoving empty directory: %s and proceeding without this collection' % write_dir)
                    os.rmdir(write_dir)


def aggregate_occurrences_and_images():
    """
    aggregate_occurrences_and_images: Aggregates data found in images.csv and occurrences.csv into a single data frame
        for each subdirectory of args.STORE. If unreadable unicode errors are encountered they are replaced with the
        unicode character for unknown '?'. Revisit this method if instead unicode errors should result in an omitted
        field or record.
    :return:
    """
    for i, (subdir, dirs, files) in enumerate(os.walk(args.STORE + '/collections')):
        # print(subdir)
        if i != 0:
            # If already merged don't re-merge:
            if not os.path.isfile(subdir + '\df_meta.csv'):
                # Ignore zero (the root directory \SERNEC).
                print('\tPerforming Aggregation: %d %s' % (i, subdir))
                # Attempt to load the occurrences.csv file but prepare to encounter unrecognized unicode characters:
                try:
                    with open(subdir + '/occurrences.csv', 'r') as fp:
                        df_occurr = pd.read_csv(fp, encoding='utf8')
                except UnicodeDecodeError:
                    # Unicode encoding errors during load, try to resolve them by replacing unknown chars w/ ?
                    print('\t\tWarning: Detected UnicodeEncodingError(s) during load of occurrences.csv. '
                          'Attempting automated resolution by replacing unknown characters with ? replacement character...')
                    try:
                        # First pass just detects the corrupted lines for more advanced error handling later:
                        with open(subdir + '/occurrences.csv', 'r', errors='replace') as fp:
                            raw = fp.read()
                            lines = raw.splitlines()
                            error_lines = {}
                            for i, line in enumerate(lines):
                                if '\ufffd' in line:
                                    # print("\t\t\tUnicode encoding error with record %d: %s" % (i+1, line))
                                    error_lines[line.split(',')[0]] = line
                        print('\t\tWarning: Analysis reports %d corrupted records which may be omitted.' % len(error_lines))
                    except UnicodeDecodeError:
                        # If this executes, for some reason replacing unknown unicode chars with ? didn't work.
                        print('\t\tError: Replacement unable to resolve UnicodeEncodingError. '
                              'PLEASE SPECIFY RESOLUTION STRATEGY')
                    print('\t\tWarning: Attempting final read of corrupted occurrences.csv with ? replacement chars...')
                    try:
                        with open(subdir + '/occurrences.csv', 'r', errors='replace') as fp:
                            df_occurr = pd.read_csv(fp, header=0, error_bad_lines=True)
                    except UnicodeDecodeError:
                        print('\t\tError: Replacement characters insufficient fix. PLEASE SPECIFY RESOLUTION STRATEGY')
                    print('\t\tSuccess: occurrences.csv read with replacement chars. Up to %d records are now corrupted.'
                          % len(error_lines))
                # Rename id column to coreid for inner merge:
                df_occurr = df_occurr.rename(index=str, columns={'id': 'coreid'})
                with open(subdir + '/images.csv', 'r') as fp:
                    df_imgs = pd.read_csv(fp)
                # Perform inner mrege on coreid
                df_meta = pd.merge(df_occurr, df_imgs, how='inner', on=['coreid'])
                df_meta.to_csv(subdir + '\df_meta.csv')


def download_high_res_images():
    """
    download_high_res_images: Goes through
    :return:
    """
    # metadata_dtypes = [np.int64, np.int64, np.object]
    for i, (subdir, dirs, files) in enumerate(os.walk(args.STORE)):
        # Skip i==0 which is the root directory.
        if i != 0:
            if args.verbose:
                 print('\tTargeting: %d %s' % (i, subdir))
            with open(subdir + '\df_meta.csv', 'r') as fp:
                df_meta = pd.read_csv(fp, header=0)
            # Correct dtypes for mixed data columns:
            # Discard all columns but the following:
            df_meta = df_meta[[
                'coreid', 'institutionCode', 'occurrenceID', 'catalogNumber',
                'kingdom', 'phylum', 'class', 'order', 'family', 'scientificName',
                'genus', 'specificEpithet', 'country', 'stateProvince', 'county',
                'locality', 'recordId', 'references', 'identifier', 'accessURI',
                'thumbnailAccessURI', 'goodQualityAccessURI', 'format', 'associatedSpecimenReference',
                'type', 'subtype'
            ]]
            # Everything should really have dtype object or be OneHotEncoded.
            # df_meta['coreid'] = df_meta['coreid'].astype('int')
            # df_meta['institutionCode'] = df_meta['institutionCode'].astype('str')
            # df_meta['occurrenceID'] = df_meta['occurrenceID'].astype('str')
            for i, row in df_meta.iterrows():
                img_response = requests.get(row['goodQualityAccessURI'])
                if img_response.status_code == 200:
                    # Create class directory:
                    return NotImplementedError


def aggregate_institution_metadata_by_species():
    """
    aggregate_institution_metadata_by_species: Aggregates all institution metadata into one dataframe characterized by
        species.
    :return df_species:
    """
    for i, (subdir, dirs, files) in enumerate(os.walk(args.STORE)):
        # Skip i==0 which is the root directory.
        if i != 0:
            if args.verbose:
                 print('\tTargeting: %d %s' % (i, subdir))

def aggregate_institution_metadata():
    """
    aggregate_institution_metadata: Combines all institution df_meta files into one
    :return:
    """
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
            print('This script does not have write permission for the supplied write directory %s. Terminating.'
                  % args.STORE)
            exit(-1)
    # Check existence of global metadata data frame:
    if not os.path.isfile(args.STORE + '/collections/df_collids.pkl'):
        print('STAGE_ONE: Failed to detect an existing df_collids.pkl. I will have to create a new one.')
        os.mkdir(args.STORE + '/collections')
        # Perform first pass of global metadata scrape. Obtain collection codes and DwC-A URLs for all collections:
        df_collids = main()
        # Save the dataframe to the hard drive.
        df_collids.to_pickle(path=args.STORE + '\collections\df_collids.pkl')
    else:
        if args.verbose:
            print('STAGE_ONE: Now loading collections dataframe \'df_collids.pkl\'. '
                  'To update this file, remove it from the HDD and run this script again...')
            # print('Global metadata dataframe \'df_collids\' already exists. Will not re-scrape unless deleted.')
        df_collids = pd.read_pickle('../../../data/SERNEC/df_collids.pkl')
        print('STAGE_ONE: Loaded collections dataframe. No need to re-download. Pipeline STAGE_ONE complete.')
        print('=' * 100)

    ''' Uncomment the following method call to attempt a re-download of DwC-A's for empty collection directories '''
    if args.verbose:
        print('Now creating local subdirectories for each collection, downloading and extracting zipped DwC-A files...')
    download_and_extract_zip_files(df_collids)

    ''' Uncomment the following method call to re-aggregate csv data for each collection's local directory '''
    if args.verbose:
        print('Now aggregating occurrence.csv and image.csv for every collection. Standby...')
    aggregate_occurrences_and_images()

    ''' Uncomment the following call to '''
    df_global = aggregate_institution_metadata()


    ''' Uncomment the following method call to re-create metadata-to-hdd links and hdd class subdirectory instantiation'''
    df_classes = aggregate_institution_metadata_by_species()

    ''' Uncomment the following method call to re-download images for every collection. '''
    if args.verbose:
        print('Now downloading high resolution images for every sample in every collection. This will take quite a while.......')
    download_high_res_images()
    pass
