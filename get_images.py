"""

    Author: siddhi bajracharya
    Date: 10/26/2022
    Description: This script is used to download neutrino dataset and save the images in a folder
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
from concurrent.futures import ProcessPoolExecutor

def download_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download csv files from the internet and return them as pandas dataframes"""
    # Download csv files from the internet
    er_sig = pd.read_csv('https://raw.githubusercontent.com/siehienp20/topanga/main/raw_er.csv', header = None)
    nr_sig = pd.read_csv('https://raw.githubusercontent.com/siehienp20/topanga/main/raw_nr.csv', header= None)
    return er_sig, nr_sig

def get_nr_er_numpy():
    er_sig, nr_sig = download_csv()
    er_sig = er_sig.drop([1876], axis=1)
    nr_sig = nr_sig.drop([1876], axis=1)
    return np.array(er_sig), np.array(nr_sig)


def get_charts(signals):
    fs = 25e7
    for i in range(0, len(signals)):
        x = signals[i]
        f, t, Sxx = signal.spectrogram(x, fs,return_onesided=False)
        plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
        plt.ylim(0,2.5e7)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.axis('off')

        yield plt

def save_image(signals,directory, image_number):
    fs = 25e7
    x = signals[image_number]
    f, t, Sxx = signal.spectrogram(x, fs,return_onesided=False)
    plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    plt.ylim(0,2.5e7)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.axis('off')
    plt.savefig('images/' + directory + 'fig_' + str(image_number) + '.png')

# train test val split a directory
def split_data(SOURCE, TRAINING, TESTING, VALIDATION, SPLIT_SIZE):
    import random
    import os
    from shutil import copyfile

    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length - int(len(files) * 0.05))
    validation_length = int(len(files) - training_length - testing_length)

    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]
    validation_set = shuffled_set[training_length:-testing_length]

    os.makedirs(TRAINING, exist_ok=True)
    os.makedirs(TESTING, exist_ok=True)
    os.makedirs(VALIDATION, exist_ok=True)

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)
    
    for filename in validation_set:
        this_file = SOURCE + filename
        destination = VALIDATION + filename
        copyfile(this_file, destination)

def main():

    save_dir = 'images/'
    print("Downloading csv files...")
    er_sig, nr_sig = get_nr_er_numpy()
    print("Done downloading csv files")
    er_charts = get_charts(er_sig)
    nr_charts = get_charts(nr_sig)
    
    os.makedirs(os.path.join(save_dir, 'NRIMAGE'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ERIMAGE'), exist_ok=True)
    counter = 0

    print("Saving images...")
    with ProcessPoolExecutor() as executor:
        for i in range(0, len(er_sig)):
            executor.submit(save_image, er_sig, 'ERIMAGE/', i)
            executor.submit(save_image, nr_sig, 'NRIMAGE/', i)
            counter += 1
            if counter % 100 == 0:
                print(f"Saved {counter} images")
    split_data('images/ERIMAGE/', 'data/ERIMAGE/TRAIN/', 'data/ERIMAGE/TEST/', 'data/ERIMAGE/VAL/', 0.8)
    
if __name__ == '__main__':
    main()


