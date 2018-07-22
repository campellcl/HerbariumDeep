import argparse
import os
import pandas as pd
import logging
import urllib3, certifi, requests
import time
from pandas.api.types import CategoricalDtype
import concurrent.futures
from filelock import Timeout, FileLock


parser = argparse.ArgumentParser(description='SERNEC web scraper command line interface.')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')


def main():
    """
    main: Returns the composite dataframe df_meta created by merging occurrence.csv with media.csv using an inner merge
        on the 'coreid' attribute. This method also eliminates non-essential columns and performs dtype specifications
        to reduce dataframe memory footprint. This method also removes records with no target class label (here deemed,
        'dwc:scientificName') and no associated image data (here attribute 'ac:accessURI').
    :return df_meta: A single dataframe representing the metadata of the entire image collection.
    """
    if os.path.isdir(args.STORE):
        read_dir = args.STORE + '\Herbaria1K_iDIgBio_Pointers\Herbaria1K_iDIgBio_Pointers'
        with open(read_dir + '\occurrence.csv', 'r', errors='replace') as fp:
            df_occurr = pd.read_csv(fp)
        # column_sizes = {column: len(df_occurr[column].unique()) for column in df_occurr.columns}
        with open(read_dir + '\multimedia.csv', 'r', errors='replace') as fp:
            df_imgs = pd.read_csv(fp)
        df_meta = pd.merge(df_occurr, df_imgs, how='inner', on=['coreid'])
        df_meta = df_meta[[
            'coreid', 'dwc:catalogNumber', 'dwc:class', 'dwc:collectionID', 'idigbio:collectionName',
            'dwc:vernacularName', 'idigbio:commonnames', 'dwc:family', 'dwc:genus', 'dwc:group',
            'idigbio:hasImage', 'idigbio:hasMedia', 'dwc:higherClassification', 'dwc:infraspecificEpithet',
            'dwc:institutionCode', 'dwc:institutionID', 'idigbio:institutionName', 'dwc:kingdom',
            'dwc:occurrenceID', 'dwc:order', 'dwc:phylum', 'dwc:recordNumber', 'dwc:specificEpithet',
            'dwc:taxonID', 'dwc:taxonomicStatus', 'dwc:taxonRank', 'dwc:typeStatus', 'ac:accessURI',
            'dwc:scientificName', 'dwc:recordedBy'
        ]]
        # Dtype conversion for memory footprint reduction:
        df_meta['dwc:family'] = df_meta['dwc:family'].astype('category')
        df_meta['dwc:class'] = df_meta['dwc:class'].astype('category')
        df_meta['dwc:genus'] = df_meta['dwc:genus'].astype('category')
        df_meta['dwc:group'] = df_meta['dwc:group'].astype('category')
        df_meta['idigbio:hasImage'] = df_meta['idigbio:hasImage'].astype('category')
        df_meta['idigbio:hasMedia'] = df_meta['idigbio:hasMedia'].astype('category')
        df_meta['idigbio:institutionName'] = df_meta['idigbio:institutionName'].astype('category')
        df_meta['dwc:kingdom'] = df_meta['dwc:kingdom'].astype('category')
        df_meta['dwc:order'] = df_meta['dwc:order'].astype('category')
        df_meta['dwc:phylum'] = df_meta['dwc:phylum'].astype('category')
        df_meta['dwc:specificEpithet'] = df_meta['dwc:specificEpithet'].astype('category')
        df_meta['dwc:recordedBy'] = df_meta['dwc:recordedBy'].astype('category')
        ''' Clean the dataframe'''
        # Drop records with a scientificName of NaN:
        orig_num_samples = df_meta.shape[0]
        df_meta = df_meta.dropna(axis=0, how='any', subset=['dwc:scientificName'])
        new_num_samples = df_meta.shape[0]
        if orig_num_samples - new_num_samples != 0:
            print('Warning: Data Lost! Dropped %d records from df_meta that had no discernible scientificName.'
                  % (orig_num_samples - new_num_samples))
        # Drop rows that have no image URLS in ac:accessURI:
        orig_num_samples = df_meta.shape[0]
        df_meta = df_meta.dropna(axis=0, how='all', subset=['ac:accessURI'])
        new_num_samples = df_meta.shape[0]
        if orig_num_samples - new_num_samples != 0:
            print('Warning: Data Lost! Dropped %d records from df_meta that had no accessURI.'
                  % (orig_num_samples - new_num_samples))
    else:
        return NotImplementedError
    return df_meta


def download_image(url, write_path, lock_path):
    """
    download_image: This method is called asynchronously by executing threads. It performs several useful functions:
        * Determines if the supplied url has already been downloaded.
        * Determines if another thread is currently downloading the supplied URL.
        * Handles the locking and unlocking of the shared resource: image-lock files.
        * Prints HTTP errors received while attempting the download of an image.
        * Stores the downloaded data at the provided write_path if successfully obtained.
    :param url: The image URL which is to be downloaded.
    :param write_path: The path indicating where the downloaded image is to be stored.
    :param lock_path: The path indicating where the .lock file for the corresponding image is to be located.
    :return url: The url that this method was tasked with downloading. Upon completion, this method will have performed
        the tasks listed above or returned None: (indicating to the controlling ThreadPoolExecutor that this thread is
        dead and should be re-allocated with a new URL to download).
    """
    time_stamp = time.time()
    # Does the file already exist?
    if not os.path.isfile(write_path):
        # print('Working on URL: %r' % url)
        # Does the lock file already exist?
        if not os.path.isfile(lock_path):
            # Create an empty lockfile:
            open(lock_path, 'a').close()
            # Lock the lockfile:
            file_lock = FileLock(lock_path, timeout=0.1)
            # print('Just created lockfile: %s The file is locked: %s' % (lock_path, file_lock.is_locked))
        # Try and acquire the file lock to see if another thread is already working on this url:
        try:
            # print('Attempting to acquire lockfile: %s. The file is now locked: %s' % (lock_path, file_lock.is_locked))
            with file_lock.acquire(timeout=0.1):
                # If this code executes the lockfile has been acquired and is now locked by this process instance.
                # print('Acquired lockfile %s. The file is locked: %s' % (lock_path, file_lock.is_locked))
                # Instantiate http object per urllib3:
                http = urllib3.PoolManager(
                    cert_reqs='CERT_REQUIRED',
                    ca_certs=certifi.where()
                )
                dl_response = http.request('GET', url)
                if dl_response.status == 200:
                    with open(write_path, 'wb') as fp:
                        fp.write(dl_response.data)
                    # print('Downloaded file: %s' % write_path)
                    return url, time_stamp
                else:
                    print('Error downloading accessURI %s. Received http response %d' % (url, dl_response.status))
        except Timeout:
            print('Attempt to acquire the file lock timed out. Perhaps another instance of this application '
                  'currently holds the lock: %s' % lock_path)
        finally:
            # NOTE: This code is guaranteed to run before any return statements are executed,
            #   see: https://stackoverflow.com/questions/11164144/weird-try-except-else-finally-behavior-with-return-statements
            '''
            NOTE: Exiting from a 'with FileLock.acquire:' block will automatically release the lock. So uncomment the,
                following lines of code only if you believe that the file lock should be released by all threads if a 
                Timeout exception occurs as well as a success:
            '''
            file_lock.release()
            # print('Released file lock: %s. The file is now locked: %s.' % (lock_path, file_lock.is_locked))
    else:
        print('The requested url: %r has already been downloaded!' % write_path)


def create_image_storage_dirs(df_meta):
    """
    create_image_storage_dirs: Creates a storage directory for every target class label in the df_meta dataframe.
    :param df_meta: The global metadata dataframe.
    :return:
    """
    for i, row in df_meta.iterrows():
        write_path = args.STORE + '\\images\\' + row['dwc:scientificName']
        if not os.path.isdir(write_path):
            os.mkdir(write_path)
            print('\tCreated new target directory: %s' % write_path)


def purge_lock_files():
    """
    purge_lock_files: Removes all lock files from the storage directory.
    :param write_dir: The directory for which lock files are to be recursively removed.
    :return:
    """
    root_folder = args.STORE + '\\images\\'
    for item in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, item)):
            for the_file in os.listdir(os.path.join(root_folder, item)):
                if the_file.endswith('.lock'):
                    file_path = os.path.join(root_folder, item)
                    file_path = os.path.join(file_path, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            # os.unlink(file_path)
                    except OSError as exception:
                        print(exception)


if __name__ == '__main__':
    global args, verbose
    args = parser.parse_args()
    verbose = args.verbose
    # Create the images storage directory if it doesn't exist already:
    if not os.path.isdir(args.STORE + '\images'):
        print('INIT: It appears the global storage directory was removed. Recreating storage directories...')
        os.mkdir(args.STORE + '\images')
    # Load the df_meta dataframe if it exists, otherwise create it:
    if os.path.isfile(args.STORE + '\images\df_meta.pkl'):
        df_meta = pd.read_pickle(args.STORE + '\images\df_meta.pkl')
    else:
        df_meta = main()
        # Add binary flags to dataframe:
        df_meta['downloaded'] = pd.Series([False for i in range(df_meta.shape[0])])
        cat_type = CategoricalDtype(categories=[True, False], ordered=False)
        df_meta['downloaded'] = df_meta['downloaded'].astype(cat_type)
        df_meta['has_error'] = pd.Series([False for i in range(df_meta.shape[0])])
        df_meta['has_error'] = df_meta['has_error'].astype(cat_type)
        df_meta['filename'] = pd.Series(['' for i in range(df_meta.shape[0])])
        df_meta.to_pickle(args.STORE + '\images\df_meta.pkl')
        # Create image storage directories:
        print('Creating image storage directories...')
        create_image_storage_dirs(df_meta)
    print('Purging left over lock files before spooling up threads...')
    purge_lock_files()
    print('Downloading Images...')
    # get list of urls and their associated labels:
    urls_and_labels = list(zip(df_meta['ac:accessURI'].tolist(), df_meta['dwc:scientificName'].tolist()))
    # get a list of urls and their associated file paths:
    urls_and_write_paths = []
    for url, label in urls_and_labels:
        f_name = os.path.basename(url)
        if str.lower(f_name).endswith('.jpg'):
            urls_and_write_paths.append((url, args.STORE + '\\images\\%s\\%s' % (label, f_name)))
        else:
            urls_and_write_paths.append((url, args.STORE + '\\images\\%s\\%s.jpg' % (label, f_name)))
    # urls_and_write_paths = [(url, args.STORE + '\\images\\%s\\%s' % (label, os.path.basename(url))) for url, label in urls_and_labels]
    # get list of lock files:
    urls_and_lock_files = [(url, write_path + '.lock') for url, write_path in urls_and_write_paths]
    # max_workers is initially the number of processor cores:
    max_workers = 6
    # FileLock timeouts:
    lock_timeout = 0.1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_image, url, write_path, lock_path): url for
                         (url, write_path), (_, lock_path) in zip(urls_and_write_paths, urls_and_lock_files)}
        # TODO: The following code is not being executed and I am unsure why:
        # for future in concurrent.futures.as_completed(concurrent.futures.FIRST_COMPLETED, future_to_url):
        #     print('DEBUG: I care about my future\'s! ;)')
        #     url, start_time = future_to_url[future]
        #     ts = time.time()
        #     print('It took %s seconds to download URL: %r.' % (start_time - ts, url))
