import os
import urllib3, certifi, requests
import zipfile, io
import pandas as pd
import time
import concurrent.futures
from filelock import Timeout, FileLock

def aggregate_occurrences_and_images(dwca_dir):
    if not os.path.isfile(os.path.join(dwca_dir, 'df_meta.csv')):
        with open(os.path.join(dwca_dir, 'images.csv'), 'r') as fp:
            df_imgs = pd.read_csv(fp)
        if df_imgs.empty:
            print('Error: \'%s\\images.csv\' was empty.' % dwca_dir)
            exit(-1)
        try:
            with open(os.path.join(dwca_dir, 'occurrences.csv'), 'r') as fp:
                df_occurr = pd.read_csv(fp, encoding='utf8')
        except UnicodeDecodeError:
            # Unicode encoding errors during load, try to resolve them by replacing unknown chars w/ ?
            print('\tWarning: Detected UnicodeEncodingError(s) during load of occurrences.csv. '
                  'Attempting automated resolution by replacing unknown characters with ? replacement character...')
            try:
                # First pass just detects the corrupted lines for more advanced error handling later:
                with open(os.path.join(dwca_dir, 'occurrences.csv'), 'r', errors='replace') as fp:
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
                with open(os.path.join(dwca_dir, 'occurrences.csv'), 'r', errors='replace') as fp:
                    df_occurr = pd.read_csv(fp, header=0, error_bad_lines=True)
            except UnicodeDecodeError:
                print('\t\tError: Replacement characters insufficient fix. PLEASE SPECIFY RESOLUTION STRATEGY')
            print('\t\tSuccess: occurrences.csv read with replacement chars. Up to %d records are now corrupted.'% len(error_lines))
        # Rename id column to coreid for inner merge:
        df_occurr = df_occurr.rename(index=str, columns={'id': 'coreid'})
        # Perform inner mrege on coreid
        df_meta = pd.merge(df_occurr, df_imgs, how='inner', on=['coreid'])
        ''' Data Reduction Techniques to reduce memory footprint '''
        # Drop columns of all NaN values:
        df_meta = df_meta.dropna(axis=1, how='all')
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
        df_meta.to_csv(os.path.join(dwca_dir, 'df_meta.csv'))


def download_and_extract_zip_file(dwca_url):
    root_dir = 'D:\\data\\BOON\\DwCA'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    zip_response = requests.get(dwca_url)
    if zip_response.status_code == 200:
        print('Downloaded DwC-A zipfile for BOON')
        dwca_zip = zipfile.ZipFile(io.BytesIO(zip_response.content))
        print('\tExamining the obtained zip file for image data prior to extraction...')
        dwca_zip.printdir()
        for zip_info in dwca_zip.infolist():
            f_name = zip_info.filename
            if f_name == 'images.csv':
                if zip_info.file_size == 218:
                    print('\tData Lost! Although \'images.csv\' does exist, it has no data other '
                          'than the header.')
                    exit(-1)
                else:
                    print('Detected an \'images.csv\' file with data. Proceeding to extraction...')
                    print('\tExtracting zip files to relative directory: %s' % root_dir)
                    dwca_zip.extractall(path=root_dir)
    else:
        print('Failed to download DwC-A zipfile for BOON, with HTTP response: %s' % zip_response.status_code)

def aggregate_collection_metadata(dwca_dir):
    """
    aggregate_institution_metadata: Combines all collection df_meta files into one global df_meta file which contains
        the information of every collection.
    :return df_meta: A global metadata dataframe containing the meatadata of every SERNEC collection.
    """
    # Don't re-aggregate if the file already exists.
    if not os.path.isfile(os.path.join(dwca_dir, 'df_meta.pkl')):
        df_meta = pd.DataFrame()
        with open(os.path.join(dwca_dir, 'df_meta.csv'), 'r', errors='replace') as fp:
            df_coll_meta = pd.read_csv(fp, index_col=0)
        df_meta = df_meta.append(df_coll_meta)
        print('\tAppended metadata successfully. New size of df_meta: (%d, %d).'
              % (df_meta.shape[0], df_meta.shape[1]))
        df_meta.to_pickle(path=os.path.join(dwca_dir, 'df_meta.pkl'))
    else:
        print('\tMetadata already aggregated as %s exists. Delete file from hard drive to re-aggregate.\n'
              'Loading BOON metadata file...' % (os.path.join(dwca_dir, 'df_meta.pkl')))
        df_meta = pd.read_pickle(path=os.path.join(dwca_dir, 'df_meta.pkl'))
    return df_meta


def _purge_lock_files(boon_image_folder):
    """
    _purge_lock_files: Removes all lock files from the storage directory.
    :return:
    """
    root_folder = boon_image_folder
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


def main():
    boon = {
        'collid': 196,
        'desc': 'Appalachian State University, I. W. Carpenter, Jr. Herbarium',
        'dwca': 'http://sernecportal.org/portal/content/dwca/BOON_DwC-A.zip',
        'inst': 'BOON',
        'emllink': 'http://sernecportal.org/portal/collections/datasets/emlhandler.php?collid=196'
    }
    download_and_extract_zip_file(boon['dwca'])
    aggregate_occurrences_and_images(dwca_dir='D:\\data\\BOON\\DwCA')
    df_meta = aggregate_collection_metadata(dwca_dir='D:\\data\\BOON\\DwCA')

    # Drop rows that have scientificName of NaN:
    num_samples_pre_drop = df_meta.shape[0]
    df_meta = df_meta.dropna(axis=0, how='any', subset=['scientificName'])
    print('Warning: Data Lost! Dropped %d records from df_meta that had no discernible scientificName.'
          % (num_samples_pre_drop - df_meta.shape[0]))

    # Drop rows that have no image URLS in either goodQualityAccessURI, accessURI, associatedSpecimenReference, or thumbnailAccessURI:
    num_samples_pre_drop = df_meta.shape[0]
    df_meta = df_meta.dropna(axis=0, how='all',
                             subset=['accessURI', 'goodQualityAccessURI', 'identifier', 'associatedSpecimenReference'])
    # Update dataframe:
    df_meta.to_pickle(path=os.path.join('D:\\data\\BOON\\DwCA', 'df_meta.pkl'))

    if (num_samples_pre_drop - df_meta.shape[0]) > 0:
        print('Warning: Data Lost! Dropped %d records from df_meta that had no associated image URL.'
              % (num_samples_pre_drop - df_meta.shape[0]))

    urls_and_labels = None
    # If there are no null values in goodQualityAccessURI use that for a URL:
    if not df_meta.goodQualityAccessURI.isnull().values.any():
        urls_and_labels = list(zip(df_meta.goodQualityAccessURI, df_meta.scientificName))

    print('Purging left over lock files before spooling up threads...')
    _purge_lock_files(boon_image_folder='D:\\data\\BOON\\images')
    print('Setting up speculative write paths...')
    urls_and_write_paths = []
    for url, label in urls_and_labels:
        f_name = os.path.basename(url)
        if str.lower(f_name).endswith('.jpg'):
            urls_and_write_paths.append((url, 'D:\\data\\BOON\\images\\%s\\%s' % (label, f_name)))
        elif str.lower(f_name).endswith('jpeg'):
            urls_and_write_paths.append((url, 'D:\\data\\BOON\\images\\%s\\%s' % (label, f_name)))
        else:
            urls_and_write_paths.append((url, 'D:\\data\\BOON\\images\\%s\\%s.jpg' % (label, f_name)))
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

if __name__ == '__main__':
    main()
