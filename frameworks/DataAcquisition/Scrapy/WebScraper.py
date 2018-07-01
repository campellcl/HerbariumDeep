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
            else:
                # NOTE: Tried web scraping for DwC-A, but if it isn't present under 'link' than the data is a EML File.
                if args.verbose:
                    print('\tThis collection: %s, INST: %s has no publicly available DwC-A. '
                          'This collection will be omitted from global data frame.')
                # if args.verbose:
                #     print('\tIssuing GET request to DwC-A source URL: %s' % link)
                #     print('\tAttempting to parse HTML for DwC-A source URL')
                # dwca_response = requests.get(link)
                # Selector(response=dwca_response).xpath('//body').extract()
                # Convert html byte repsonse to string type and then to a dict for json encoding:
                # json_res = json.loads(dwca_response.text)

            df_collids = df_collids.append(coll_series, ignore_index=True)
    pass
    # # Open a csv of SERNEC institutions and institution codes:
    # with open('../../../data/SERNEC/InstitutionCodes/InstitutionCodes.csv', 'r') as fp:
    #     institution_codes = pd.read_csv(fp, header=0, dtype={'institution': str, 'code': str})
    #     # Remove any trailing or starting white spaces in the institution codes:
    #     institution_codes['code'] = institution_codes['code'].str.strip()
    # # Go through every institution:
    # for index, row in institution_codes.iterrows():
    #     inst = row[0]
    #     code = row[1]
    #     # Setup directory and metadata for institution if not already present:
    #     init_storage_dir(inst, code)
    # # ?


    # http = urllib3.PoolManager(
    #     cert_reqs = 'CERT_REQUIRED',
    #     ca_certs=certifi.where()
    # )
    # if args.verbose:
    #     print('Issuing GET request to source URL: %s' % args.source_url)
    # r = http.request('GET', args.source_url)
    # if args.verbose:
    #     print('Received HTTP response code: %d' % r.status)
    #     print(r.data)
    #     # TODO: Need a way to retrieve all the herbaria 'instcode' that bisque is using. Try my institutionCodes.csv
    #     # InstitutionCodes.csv created using: http://sernecportal.org/portal/collections/index.php
    # if r.status != 200:
    #     print('Recieved HTTP response code: %d. Now terminating.' % r.status)
    #     exit(-1)





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
    main()
