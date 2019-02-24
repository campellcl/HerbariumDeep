"""
SERNECImageDownloader.py
A multi-threaded image downloader for use with SERNEC data. This program requires an existing df_meta.pkl file to be
created under the data/SERNEC/collections directory. If this is not the case then WebScraper.py should be run.
"""

__author__ = 'Chris Campell'
__created__ = '7/7/2018'
__updated__ = '11/17/2018'

import logging
import os
import argparse
import pandas as pd
import urllib3, certifi, requests
import time
import concurrent.futures
from filelock import Timeout, FileLock


'''
Command Line Argument Parsers: 
'''

parser = argparse.ArgumentParser(description='SERNEC web scraper command line interface.')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
# parser.add_argument('WRITE', metavar='DIR', help='Image storage directory')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')


def _manually_clean_malformed_class_labels(df_meta):
    orig_df = df_meta.copy()
    anticipated_sample_loss = 0
    for iloc, row in orig_df.iterrows():
        scientific_name = row.scientificName
        write_dir = args.STORE + '\\images\\' + scientific_name
        if not os.path.exists(write_dir):
            try:
                os.mkdir(write_dir)
            except OSError as err:
                # print('\t\tERROR: Received the following error during directory creation:')
                # print('\t\t\t %s' % err)
                print('%s' % err)
                num_samples_of_class_in_data =  orig_df[orig_df['scientificName'] == scientific_name].shape[0]
                anticipated_sample_loss += num_samples_of_class_in_data
                print('\tclass: \'%s\' count: %d' % (scientific_name, num_samples_of_class_in_data))
    print('Total Anticipated Sample Losses: %d' % anticipated_sample_loss)
    #
    # for class_label, count in df_meta.scientificName.value_counts().items():
    #     if count < 10:
    #         print('class_label (scientific name): \'%s\'\t\tCount: %d' % (class_label, count))
    return df_meta


def _remove_samples_with_invalid_directory_names_and_create_valid_directories(df_meta):
    # Drop classes whose names form invalid windows directories:
    updated_df_meta = df_meta.copy()
    violating_classes = []
    for sci_name in df_meta.scientificName.values:
        class_dir = args.STORE + '\\images\\' + sci_name
        if not os.path.exists(class_dir):
            try:
                os.mkdir(class_dir)
                # os.rmdir(class_dir)
            except OSError as err:
                violating_classes.append(sci_name)
    print('Dropping %d classes with invalid Window\'s directory names.' % len(violating_classes))
    print('Dropping: %s' % violating_classes)
    init_num_samples = updated_df_meta.shape[0]
    updated_df_meta = updated_df_meta[~updated_df_meta['scientificName'].isin(violating_classes)]
    print('Dropped %d samples.' % (init_num_samples - updated_df_meta.shape[0]))
    return updated_df_meta


def _remove_classes_with_less_than_x_samples(df_meta, x=20):
    updated_df_meta = df_meta.copy()
    # Drop classes who have less than x samples:
    class_labels_to_discard = set()
    for sci_name, count in df_meta.scientificName.value_counts().items():
        if count < x:
            class_labels_to_discard.add(sci_name)
    print('Dropping %d classes with less than %d samples.' % (len(class_labels_to_discard), x))
    # print('Dropping: %s' % class_labels_to_discard)
    init_num_samples = updated_df_meta.shape[0]
    updated_df_meta = updated_df_meta[~updated_df_meta['scientificName'].isin(class_labels_to_discard)]
    # updated_df_meta = updated_df_meta[updated_df_meta.scientificName not in class_labels_to_discard]
    print('Dropped %d samples.' % (init_num_samples - updated_df_meta.shape[0]))
    return updated_df_meta


def _perform_manual_class_label_multiplexing(df_meta):
    """
    _perform_manual_class_label_multiplexing: Merges the metadata class records with different taxonomic scope into the
        same named entity. For example, records: 'Acacia farnesiana','Acacia farnesiana var. farnesiana', and
        'Acacia farnesiana native to southern coastal...' should all be treated as the same class
        label: 'Acacia farnesiana'.
    :param df_meta:
    :return:
    """
    # TODO: Globally replace the 'subsp.' word with the 'var.' word:
    # classes_manually_flagged_for_removal = {
    #     '(a) Potamogeton pusillus subsp. tenuissimus (b) Najas guadalupensis subsp. guadalupensis',
    #     '(Broom Comparison)',
    #     '(Elymus lanceolatus x Elymus glaucus) Elymus trachycaulus'
    # }
    return df_meta


def main():
    metadata_file_path = 'C:\\Users\\ccamp\\Documents\\GitHub\\HerbariumDeep\\data\\SERNEC\\collections\\df_meta.pkl'
    df_meta = None
    if os.path.isfile(metadata_file_path):
        df_meta = pd.read_pickle(metadata_file_path)
        if df_meta.empty:
            print('ERROR: Could not read df_meta.pkl. Re-run the metadata WebScraper.')
        else:
            print('Loaded metadata dataframe: \'df_meta.pkl\'.')
    else:
        print('ERROR: Could not locate df_meta.pkl on the local hard drive at \'%s\'. '
              'Have you run the metadata WebScraper (WebScraper.py)?' % metadata_file_path)
        exit(-1)
    # Discard scientificNames that contain the unicode replacement character '?':
    # len(df_meta[df_meta['scientificName'].str.contains('\?')])
    init_num_samples = df_meta.shape[0]
    df_meta = df_meta[~df_meta['scientificName'].str.contains('\?')]
    df_meta = _remove_classes_with_less_than_x_samples(df_meta, x=100)
    df_meta = _remove_samples_with_invalid_directory_names_and_create_valid_directories(df_meta)
    print('Finished first pass of data cleaning stage. Removed %d/%d samples. There are %d samples remaining.' % ((init_num_samples - df_meta.shape[0]), init_num_samples, df_meta.shape[0]))

    # TODO: resolve class labels that should be the same.
    df_meta = _perform_manual_class_label_multiplexing(df_meta)
    #
    return df_meta


# def create_storage_dirs(targets_and_urls):
#     """
#     create_storage_dirs: Creates storage directories for every unique class under args.STORE\images.
#     :param targets_and_urls: A list of target class labels and the associated image URLs.
#     :return:
#     """
#     write_dir = args.STORE + '\images'
#     if not os.path.isdir(write_dir):
#         print('\tThis script detects no existing image folder: %s. Instantiating class storage directories...')
#         os.mkdir(write_dir)
#     print('\tNow instantiating class image storage directories...')
#     pruned_targets_and_urls = targets_and_urls.copy()
#     num_failed_dirs = 0
#     num_created_dirs = 0
#     for i, (target, url) in enumerate(targets_and_urls):
#         target_dir = write_dir + '\%s' % target
#         if os.path.isdir(target_dir):
#             pass
#             # if args.verbose:
#             #     print('\t\tTarget class label directory %s already exists.' % target_dir)
#         else:
#             if args.verbose:
#                 print('\t\tCreating storage dir %s for target %s' % (target_dir, target))
#             try:
#                 os.mkdir(target_dir)
#                 num_created_dirs += 1
#             except OSError as err:
#                 print('\t\tERROR: Received the following error during directory creation:')
#                 print('\t\t\t %s' % err)
#                 pruned_targets_and_urls.remove((target, url))
#                 num_failed_dirs += 1
#
#     print('\tFinished instantiating target label directories. There were %d directories created this '
#           'time and %d failed attempts.' % (num_created_dirs, num_failed_dirs))
#     return pruned_targets_and_urls


def _purge_lock_files():
    """
    _purge_lock_files: Removes all lock files from the storage directory.
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

    urls_and_labels = None
    # If there are no null values in goodQualityAccessURI use that for a URL:
    if not df_meta.goodQualityAccessURI.isnull().values.any():
        urls_and_labels = list(zip(df_meta.goodQualityAccessURI, df_meta.scientificName))

    print('Purging left over lock files before spooling up threads...')
    _purge_lock_files()
    print('Setting up speculative write paths...')
    urls_and_write_paths = []
    for url, label in urls_and_labels:
        f_name = os.path.basename(url)
        if str.lower(f_name).endswith('.jpg'):
            urls_and_write_paths.append((url, args.STORE + '\\images\\%s\\%s' % (label, f_name)))
        else:
            urls_and_write_paths.append((url, args.STORE + '\\images\\%s\\%s.jpg' % (label, f_name)))
    # get list of lock files:
    urls_and_lock_files = [(url, write_path + '.lock') for url, write_path in urls_and_write_paths]
    # max_workers is initially the number of processor cores:
    max_workers = 6
     # FileLock timeouts:
    lock_timeout = 0.1
    print('Downloading files, spooling threads...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_image, url, write_path, lock_path): url for
                         (url, write_path), (_, lock_path) in zip(urls_and_write_paths, urls_and_lock_files)}
    # http = urllib3.PoolManager(
    #     cert_reqs = 'CERT_REQUIRED',
    #     ca_certs = certifi.where()
    # )
    # for i in range(10):
    #     download_path = args.STORE + '\images\%s' % targets_and_urls[i][0]
    #     # download_path / os.path.basename(link)
    #     logger.info('Downloading %s', targets_and_urls[i][1])
    #      # Instantiate http object per urllib3:
    #     ts = time()
    #     response = http.request('GET', targets_and_urls[i][1])
    #     logging.info('Took %s seconds', time() - ts)
    #     if not response.status == 200:
    #         print('ERROR: Recieved http response %d' % response.status)
    #     else:
    #         with download_path.open('wb') as fp:
    #             fp.write(response.data)



    # Instantiate thread pool:
    # executor = ThreadPoolExecutor(max_workers=6)

