import argparse
import os
import pandas as pd
import logging
import urllib3, certifi, requests
import time
import signal
from pandas.api.types import CategoricalDtype
import concurrent.futures
from functools import partial


parser = argparse.ArgumentParser(description='SERNEC web scraper command line interface.')
parser.add_argument('STORE', metavar='DIR', help='Data storage directory.')
parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                    help='Enable verbose print statements (yes, no)?')


def main():
    if os.path.isdir(args.STORE):
        read_dir = args.STORE + '\Herbaria1K_iDIgBio_Pointers\Herbaria1K_iDIgBio_Pointers'
        with open(read_dir + '\occurrence.csv', 'r', errors='replace') as fp:
            df_occurr = pd.read_csv(fp)
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
            'dwc:scientificName'
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


def download_images(df_meta):
    download_times = []
    # Instantiate http object per urllib3:
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
    )
    for i, row in df_meta.iterrows():
        # Check to see if this file has been downloaded already:
        if not row['downloaded']:
            target = row['dwc:scientificName']
            write_path = args.STORE + '\images\%s' % target
            if not os.path.isdir(write_path):
                os.mkdir(write_path)
            url = row['ac:accessURI']
            time_stamp = time.time()
            dl_response = http.request('GET', url)
            elapsed_time = time.time() - time_stamp
            print('\tResponse received in %s seconds.' % elapsed_time)
            download_times.append(elapsed_time)
            print('\tAverage download time %s seconds for %d records.'
                  % ((sum(download_times)/ len(download_times)), len(download_times)))
            # Check if the download was successful:
            if not dl_response.status == 200:
                print('Error: Received http response %d while attempting to download record %d (%s).'
                      % (dl_response.status, i, row['dwc:scientificName']))
                # has_err_flag_updates.append((i, True))
            else:
                # Get the number of files in the target class directory to decide what the name of the image should be:
                num_files_in_write_dir = len([name for name in os.listdir(write_path)])
                f_name = write_path + '\%06d.jpg' % num_files_in_write_dir
                with open(f_name, 'wb') as fp_out:
                    fp_out.write(dl_response.data)
                # Update dataframe copy:
                df_meta_updated.iloc[i, -2] = True
                # Release connection:
                dl_response.release_conn()
                # Update flags:
                df_meta_updated.iloc[i, -3] = True
                # df_meta_updated.iloc[i]['downloaded'] = df_meta_updated.iloc[i]['downloaded'] = True
                df_meta_updated.iloc[i, -1] = f_name
                # df_meta_updated.iloc[i]['filename'] = f_name
                # downloaded_flag_updates.append((i, True))
                # file_name_updates.append((i, f_name))
                print('\t\tDownloaded record %d (%s) successfully and saved to: %s' % (i, row['dwc:scientificName'], f_name))


def download_image(accessURI):
    # Instantiate http object per urllib3:
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
    )
    dl_response = http.request('GET', accessURI)
    if dl_response.status == 200:
        return dl_response.data
    else:
        print('Error downloading accessURI %s. Received http response %d' % (accessURI, dl_response.data))



def signal_handler(signum, frame):
    # print('TERMINAL: Signal handler called with signal %d.' % signum)
    if signum == 2:
        print('TERMINAL: Kill request SIGINT received. Saving state information. '
              'Updating metadata flags then exiting gracefully...')
        df_meta_updated.to_pickle(args.STORE + '\images\df_meta.pkl')
        exit(1)


if __name__ == '__main__':
    global args, verbose
    args = parser.parse_args()
    verbose = args.verbose
    # Kill signal handler:
    print('Attaching SIGINT listener to process. Ensure this program is run with the option to \'emulate terminal '
          'in output console\' in PyCharm.')
    print('IMPORTANT: DO NOT KILL THIS PROGRAM in any way other than using ctrl+C in the terminal. Otherwise the '
          'program state will be corrupted.')
    signal.signal(signal.SIGINT, handler=signal_handler)
    # signal.signal(signal.SIGKILL, handler=signal_handler)
    # signal.signal(signal.SIGTERM, handler=signal_handler)
    # Create the images storage directory if it doesn't exist already:
    if not os.path.isdir(args.STORE + '\images'):
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
    df_meta_updated = df_meta.copy(deep=True)
    print('Downloading Images...')
    # download_images(df_meta)
    ''' MultiThreading (see: https://docs.python.org/3/library/concurrent.futures.html) '''
    max_workers = 6
    urls = df_meta['ac:accessURI'].tolist()
    urls = [urls[i:i + max_workers] for i in range(0, len(urls), max_workers)]
    batch_number = 0
    for i in range(len(urls)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(download_image, url): url for url in urls[i]}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                print('url: %s' % url)
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    print('%r page is %d bytes' % (url, len(data)))


    pass

