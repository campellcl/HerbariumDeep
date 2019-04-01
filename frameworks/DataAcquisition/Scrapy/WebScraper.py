"""
WebScraper.py
A urllib and Scrapy Selector based web scraper.
"""

import os
import argparse
import urllib3
import certifi
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile, io
import shutil
'''
Command line argument parsers:
'''

parser = argparse.ArgumentParser(description='SERNEC web scraper command line interface.')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-s', '--source-url', dest='source_url', metavar='URL',
                    default='https://bisque.cyverse.org/data_service/image?value=*/iplant/home/shared/sernec/*',
                    help='Source URL for web scraper.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')


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
        collection extracting all files to the respective directory. Checks the extracted images.csv file for data
        besides the csv header. If no data is found the parent directory for the collection is removed and the
        collection is dropped from the returned df_collids.
    :param df_collids: The global metadata dataframe containing SERNEC collections.
    :return df_collids: The updated collids dataframe with collections containing no image data removed.
    :return num_dl_requested: The number of DwC-A zip files that the script attempted to download.
    :return num_dl_recieved: The number of DwC-A zip files that the script downloaded successfully.
    :return num_new_collections_added: The number of new collections that were added. These collections:
        * Have DwC-A's that downloaded and extracted with no errors.
        * Have an images.csv file that contained data other than the header.
    """
    flagged_for_removal = {}
    num_dl_requested = 0
    num_dl_recieved = 0
    num_dl_rejected = 0
    # Download the DwC-A zip file for every collection:
    for index, row in df_collids.iterrows():
        write_dir = args.STORE + '\collections\\' + row['inst']
        # Create a storage directory for this collection if it doesn't already exist:
        if not os.path.isdir(write_dir):
            num_dl_requested += 1
            os.mkdir(write_dir)
            # Download the DwC-A zip file:
            zip_response = requests.get(row['dwca'])
            if zip_response.status_code == 200:
                num_dl_recieved += 1
                if args.verbose:
                    print('\tDownloaded DwC-A zipfile for COLLID: %s, INST: %s, with HTTP response 200 (OK)'
                          % (row['collid'], row['inst']))
                # Convert raw byte response to zip file:
                dwca_zip = zipfile.ZipFile(io.BytesIO(zip_response.content))
                dwca_zip.printdir()
                # Extract all the zip files to the specified directory.
                if args.verbose:
                    print('\t\tExamining the obtained zip file for image data prior to extraction...')
                for zip_info in dwca_zip.infolist():
                    f_name = zip_info.filename
                    if f_name == 'images.csv':
                        if zip_info.file_size == 218:
                            num_dl_rejected += 1
                            print('\t\tData Lost! Although \'images.csv\' does exist, it has no data other '
                                  'than the header. The subdirectory: %s will now be removed.' % write_dir)
                            shutil.rmtree(write_dir)
                            flagged_for_removal[row['collid']] = row['inst']
                            print('\t\tThe rest of the pipeline will proceed without collection %s. '
                                  'Flagged this collection for removal from \'df_collids\'.' % (row['inst']))
                        else:
                            print('\t\tDetected an \'images.csv\' file with data. Proceeding to extraction...')
                            print('\t\tExtracting zip files to relative directory: %s' % write_dir)
                            dwca_zip.extractall(path=write_dir)
            else:
                if args.verbose:
                    print('\t\tData Lost! Failed to download DwC-A zipfile for COLLID: %s, INST: %s, with HTTP response: %s'
                          % (row['collid'], row['inst'], zip_response.status_code))
                    print('\t\tRemoving empty directory: %s and proceeding without this collection' % write_dir)
                os.rmdir(write_dir)
    print('\tFinished analysis. Notifying \'df_collids\' to remove the collections with no image data: %s'
          % list(flagged_for_removal.values()))
    for collid, inst in flagged_for_removal.items():
        df_collids = df_collids[df_collids.collid != collid]
    num_new_collections_added = (num_dl_recieved - num_dl_rejected)
    return df_collids, num_dl_requested, num_dl_recieved, num_dl_rejected, num_new_collections_added


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
                with open(subdir + '/images.csv', 'r') as fp:
                    df_imgs = pd.read_csv(fp)
                # Check to ensure this collection has image data:
                if df_imgs.empty:
                    print('\t\tError: %s\images.csv was read improperly with no errors!!!!! EXAMINE!' % subdir)
                    # shutil.rmtree(subdir)
                    # print('\t\tError: Data Lost! Removed %s from hard drive' % subdir)
                else:
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
                                for i_l, line in enumerate(lines):
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
                    # Perform inner mrege on coreid
                    df_meta = pd.merge(df_occurr, df_imgs, how='inner', on=['coreid'])
                    ''' Data Reduction Techniques to reduce memory footprint '''
                    # Drop columns of all NaN values:
                    df_meta = df_meta.dropna(axis=1, how='all')

                    # Columns that were desired but not all datasets contained:
                    # intraspecificEpithet
                    # class
                    # catalogNumber

                    # Keep only these columns:
                    columns_to_retain = [
                                'institutionCode', 'collectionID', 'occurrenceID',
                                'kingdom', 'phylum', 'order', 'family', 'scientificName',
                                'scientificNameAuthorship', 'genus', 'specificEpithet',
                                'recordId', 'references', 'identifier', 'accessURI', 'thumbnailAccessURI',
                                'goodQualityAccessURI', 'format', 'associatedSpecimenReference', 'type', 'subtype'
                            ]
                    if 'recordedBy' in df_meta.columns:
                        columns_to_retain.append('recordedBy')
                        if 'recordEnteredBy' in df_meta.columns:
                            columns_to_retain.append('recordEnteredBy')

                    # Drop everything but the specified columns:
                    df_meta = df_meta[columns_to_retain]
                    # Convert object dtype to categorical where appropriate:
                    df_meta.kingdom = df_meta.kingdom.astype('category')
                    df_meta.phylum = df_meta.phylum.astype('category')
                    # df_meta['class'] = df_meta['class'].astype('category')
                    df_meta.order = df_meta.order.astype('category')
                    df_meta.family = df_meta.family.astype('category')
                    df_meta.scientificName = df_meta.scientificName.astype('category')
                    df_meta.genus = df_meta.genus.astype('category')
                    df_meta.specificEpithet = df_meta.specificEpithet.astype('category')
                    # df_meta.infraspecificEpithet = df_meta.infraspecificEpithet.astype('category')
                    df_meta.format = df_meta.format.astype('category')
                    df_meta.type = df_meta.type.astype('category')
                    df_meta.subtype = df_meta.subtype.astype('category')
                    if 'recordedBy' in df_meta.columns:
                        df_meta.recordedBy = df_meta.recordedBy.astype('category')
                    if 'recordEnteredBy' in df_meta.columns:
                        df_meta.recordEnteredBy = df_meta.recordEnteredBy.astype('category')
                    # Reduce integer 64 bit and float 64 bit representations to 32 bit representations where appropriate.
                    df_meta.to_csv(subdir + '\df_meta.csv')


def aggregate_collection_metadata():
    """
    aggregate_institution_metadata: Combines all collection df_meta files into one global df_meta file which contains
        the information of every collection.
    :return df_meta: A global metadata dataframe containing the meatadata of every SERNEC collection.
    """
    # Don't re-aggregate if the file already exists.
    if not os.path.isfile(args.STORE + '\collections\df_meta.pkl'):
        df_meta = pd.DataFrame()
        for i, (subdir, dirs, files) in enumerate(os.walk(args.STORE + '\collections')):
            # Skip i==0 which is the root directory.
            if i != 0:
                if args.verbose:
                    print('\tTargeting: %d %s' % (i, subdir))
                with open(subdir + '\df_meta.csv', 'r', errors='replace') as fp:
                    df_coll_meta = pd.read_csv(fp, index_col=0)

                df_meta = df_meta.append(df_coll_meta)
                if args.verbose:
                    print('\t\tAppended metadata successfully. New size of df_meta: (%d, %d).'
                          % (df_meta.shape[0], df_meta.shape[1]))

        df_meta.to_pickle(path=args.STORE + '\collections\df_meta.pkl')
    else:
        print('\tMetadata already aggregated as %s exists. Delete file from hard drive to re-aggregate.\n'
              'Loading global metadata file...' % (args.STORE + '\collections\df_meta.pkl'))
        df_meta = pd.read_pickle(path=args.STORE + '\collections\df_meta.pkl')
    return df_meta


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
    ''' Data Pipeline STAGE_ONE: Download unique collection Id's and their associated DwC-A links '''
    # Check existence of global metadata data frame:
    if not os.path.isfile(args.STORE + '\collections\df_collids.pkl'):
        print('STAGE_ONE: Failed to detect an existing df_collids.pkl. I will have to create a new one.')
        if not os.path.isdir(args.STORE + '/collections'):
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
        df_collids = pd.read_pickle(args.STORE + '\collections\df_collids.pkl')
        print('STAGE_ONE: Loaded collections dataframe. No need to re-download. Pipeline STAGE_ONE complete.')
        print('=' * 100)

    ''' Data Pipeline STAGE_TWO: Download and extract zipped DwC-A files for each collection '''
    if args.verbose:
        print('STAGE_TWO: Downloading and extracting zipped DwC-A files for each collection. Checking contents for '
              'actual image data. Removing collections with no image data. Creating collection subdirectories otherwise'
              '...')
    df_collids, num_dl_requested, num_dl_recieved, num_dl_rejected, num_collections_added = download_and_extract_zip_files(df_collids)
    print('STAGE_TWO: Requested %d new DwC-A downloads. Received %d new DwC-A downloads successfully. Rejected %d '
          'new downloads with no real image data. Final relevant collections added: %d. '
          'Now updating saved version of \'df_collid\' with omitted data...'
          % (num_dl_requested, num_dl_recieved, num_dl_rejected, num_collections_added))
    df_collids.to_pickle(args.STORE + '\collections\df_collids.pkl')
    print('STAGE_TWO: Pipeline STAGE_TWO complete. Updated saved df_collids. Removed collections with no real image '
          'data. Created directories and extracted zip files for all relevant collections.')
    print('=' * 100)

    ''' Data Pipeline STAGE_THREE: Aggregate the occurrences.csv and images.csv files for every collection '''
    if args.verbose:
        print('STAGE_THREE: Stepping through every collection aggregating occurrence.csv and image.csv files. Standby...')
    aggregate_occurrences_and_images()
    print('STAGE_THREE: Pipeline STAGE_THREE complete. Aggregated every collection\'s occurrence and image data.')
    print('=' * 100)
    # print('DEV-OP: Removing Stage Three programmatically...')
    # Undo stage three:
    # for i, (subdir, dirs, files) in enumerate(os.walk(args.STORE + '/collections')):
    #     # print(subdir)
    #     if i != 0:
    #         # If already merged don't re-merge:
    #         if os.path.isfile(subdir + '\df_meta.csv'):
    #             os.remove(subdir + '\df_meta.csv')

    ''' Data Pipeline STAGE_FOUR: Aggregate every collection's data into one dataframe 'df_meta' '''
    print('STAGE_FOUR: Stepping through every collection aggregating into one global metadata dataframe. Patience...')
    df_meta = aggregate_collection_metadata()

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
    # Update dataframe:
    df_meta.to_pickle(path=args.STORE + '\collections\df_meta.pkl')

    if args.verbose:
        if (num_samples_pre_drop - df_meta.shape[0]) > 0:
            print('Warning: Data Lost! Dropped %d records from df_meta that had no associated image URL.'
                  % (num_samples_pre_drop - df_meta.shape[0]))
    # for i, row in df_meta.iterrows():
    #     target = row['scientificName']
    #     url = None
    #     if row['goodQualityAccessURI']:
    #         url = row['goodQualityAccessURI']
    #     elif row['accessURI']:
    #         url = row['accessURI']
    #     elif row['associatedSpecimenReference']:
    #         url = row['associatedSpecimenReference']
    #     else:
    #         print('ERROR: Data Lost! Record %d does not possess a valid image URL' % i)
    #     targets_and_urls.append((target, url))
    print('STAGE_FOUR: Pipeline STAGE_FOUR complete. Aggregated every collection\'s metadata into one global dataframe.')
    print('=' * 100)
