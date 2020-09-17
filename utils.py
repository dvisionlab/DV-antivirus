# generate histograms
from matplotlib import pyplot as plt
import csv
import numpy as np
import argparse
import os
import SimpleITK as sitk
import zipfile36 as zipfile

# Images w CM

# -       Separate right and left lung
# -       Select values in the range:  -983 < HU < -740
# -       Plot histogram with interval of 10 HU
# -       Calculate number of pixels with Low perfusion: -983 < HU < -869
# -       Calculate number of pixels with High perfusion: -869 < HU < -740
# -       Calculate the % of voxels of Low perfusion/Total volume


def examine(csv_path):
    print('Loading data...')
    data = np.genfromtxt(csv_path, delimiter=';', dtype=int, names=True)
    print('Data loaded.')
    # define which is left/right
    left_data = data[data['polmone'] == 1]
    right_data = data[data['polmone'] == 2]
    print('LEFT LUNG\tRIGHT LUNG')
    print(len(left_data), '\t', len(right_data))

    left_data_low = left_data[left_data['perfusion'] == 10]['valore_con_mdc']
    left_data_high = left_data[left_data['perfusion'] == 20]['valore_con_mdc']
    right_data_low = right_data[right_data['perfusion']
                                == 10]['valore_con_mdc']
    right_data_high = right_data[right_data['perfusion']
                                 == 20]['valore_con_mdc']
    print(len(left_data_low), '\t', len(left_data_high))
    print(len(right_data_low), '\t', len(right_data_high))

    print(left_data_low)

    # select mdc column
    left_data_mdc = left_data['valore_con_mdc']
    right_data_mdc = right_data['valore_con_mdc']

    # compute ratio on total volume
    left_ratio = len(left_data_low)/(len(left_data_low)+len(left_data_high))
    right_ratio = len(right_data_low) / \
        (len(right_data_low)+len(right_data_high))
    print(left_ratio, '\t', right_ratio)
    # plot hist
    start = -983
    stop = -740
    step = 10
    bins = range(start, stop, step)
    print(start, stop, step, bins)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 10))

    # LEFT
    nl, binsl, patchesl = axes[0].hist(
        left_data_mdc, bins, facecolor='blue', alpha=0.3, label='high perfusion')
    # axes[0].hist(
    #     left_data_high, bins, facecolor='blue', alpha=0.6, label='high perfusion')
    axes[0].hist(
        left_data_low, bins, facecolor='blue', alpha=0.9, label='low perfusion')
    # RIGHT
    nr, binsr, patchesr = axes[1].hist(
        right_data_mdc, bins, facecolor='red', alpha=0.3, label='high perfusion')
    # axes[1].hist(
    #     right_data_high, bins, facecolor='red', alpha=0.6, label='high perfusion')
    axes[1].hist(
        right_data_low, bins, facecolor='red', alpha=0.9, label='low perfusion')

    # CHARTS SETUP
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    fig.suptitle("Image with mc, Arterial 1mm")

    y_max = int(max(nl.max(), nr.max()))

    yticks = range(0, y_max, 100000)
    axes[0].set_title('Left lung low perfusion volume = %s' % left_ratio)
    axes[0].set_xticks(bins)
    axes[0].set_yticks(yticks)
    axes[1].set_title('Right lung low perfusion volume = %s' % right_ratio)
    axes[1].set_xticks(bins)
    axes[1].set_yticks(yticks)
    axes[1].set_title('Right lung low perfusion volume = %s' % right_ratio)

    # save hist
    plt.savefig('plots.png')
    plt.show()

    # save data
    with open('stats.csv', mode='w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['bin', 'val_left', 'val_right'])
        print(bins, len(nl), len(nr))
        for n in range(len(bins)-1):
            print([bins[n], int(nl[n]), int(nr[n])])
            writer.writerow([bins[n], int(nl[n]), int(nr[n])])


def examine_treshold(csv_path, thresholds):
    print('Loading data...')
    data = np.genfromtxt(csv_path, delimiter=';', dtype=int, names=True)
    print('Data loaded.')
    # define which is left/right
    left_data = data[data['polmone'] == 1]
    right_data = data[data['polmone'] == 2]
    print('LEFT LUNG\tRIGHT LUNG')
    print(len(left_data), '\t', len(right_data))

    # select mdc column
    left_data_mdc = left_data['valore_con_mdc']
    right_data_mdc = right_data['valore_con_mdc']
    # select data inside the thresholds
    t1_low = thresholds[0]
    t2_low = thresholds[1]
    t1_high = thresholds[1]
    t2_high = thresholds[2]
    left_data_low = left_data_mdc[(
        t1_low <= left_data_mdc) & (left_data_mdc < t2_low)]
    right_data_low = right_data_mdc[(
        t1_low <= right_data_mdc) & (right_data_mdc < t2_low)]
    left_data_high = left_data_mdc[(
        t1_high <= left_data_mdc) & (left_data_mdc < t2_high)]
    right_data_high = right_data_mdc[(
        t1_high <= right_data_mdc) & (right_data_mdc < t2_high)]

    print(len(left_data_low), '\t', len(left_data_high))
    print(len(right_data_low), '\t', len(right_data_high))

    print(left_data_low)

    # compute ratio on total volume
    left_ratio = len(left_data_low)/(len(left_data_low)+len(left_data_high))
    right_ratio = len(right_data_low) / \
        (len(right_data_low)+len(right_data_high))
    print(left_ratio, '\t', right_ratio)
    # plot hist
    start = -983
    stop = -740
    step = 10
    bins = range(start, stop, step)
    print(start, stop, step, bins)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 10))

    # LEFT
    nl, binsl, patchesl = axes[0].hist(
        left_data_mdc, bins, facecolor='blue', alpha=0.3, label='high perfusion')
    # axes[0].hist(
    #     left_data_high, bins, facecolor='blue', alpha=0.6, label='high perfusion')
    axes[0].hist(
        left_data_low, bins, facecolor='blue', alpha=0.9, label='low perfusion')
    # RIGHT
    nr, binsr, patchesr = axes[1].hist(
        right_data_mdc, bins, facecolor='red', alpha=0.3, label='high perfusion')
    # axes[1].hist(
    #     right_data_high, bins, facecolor='red', alpha=0.6, label='high perfusion')
    axes[1].hist(
        right_data_low, bins, facecolor='red', alpha=0.9, label='low perfusion')

    # CHARTS SETUP
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    fig.suptitle("Image with mc, Arterial 1mm \n Range: {} / {} \n Threshold: {}".format(
                 thresholds[0], thresholds[2], thresholds[1]))

    y_max = int(max(nl.max(), nr.max()))

    yticks = range(0, y_max, 100000)
    axes[0].set_title('Left lung low perfusion volume = %s' % left_ratio)
    axes[0].set_xticks(bins)
    axes[0].set_yticks(yticks)
    axes[1].set_title('Right lung low perfusion volume = %s' % right_ratio)
    axes[1].set_xticks(bins)
    axes[1].set_yticks(yticks)

    # save hist
    folder = os.path.dirname(csv_path)
    filepath = os.path.join(folder, "histogram.png", )
    plt.savefig(filepath)
    plt.show()

    # save data
    with open('stats.csv', mode='w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['bin', 'val_left', 'val_right'])
        print(bins, len(nl), len(nr))
        for n in range(len(bins)-1):
            print([bins[n], int(nl[n]), int(nr[n])])
            writer.writerow([bins[n], int(nl[n]), int(nr[n])])


def plot(csv_path):
    l1 = []
    l2 = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if (row[0] == 'i'):
                continue
            l1.append(row[3])
            l2.append(row[4])

    fig, axis = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

    arr_f = np.array(l1)
    arr_f = arr_f.astype(np.int)

    arr_m = np.array(l2)
    arr_m = arr_m.astype(np.int)

    axis[0].hist(arr_f, bins=256)
    axis[0].set_title('values_f')

    axis[1].hist(arr_m, bins=256)
    axis[1].set_title('values_m')

    plt.show()


def moveIntoFolder(f, name, wdir):
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace(",", "_")
    name = name.replace(".", "_")
    name = name.replace("/", "_")
    dest_folder = os.path.join(wdir, name)
    wdir_path = os.path.join(wdir, f)
    dest_path = os.path.join(dest_folder, f)
    os.makedirs(dest_folder, exist_ok=True)
    print(wdir_path, ' >> ', dest_path)
    os.rename(wdir_path, dest_path)


def getImageSeriesId(file_name, series_list, desc_list):
    print('Reading image...')
    # A file name that belongs to the series we want to read

    # Read the file's meta-information without reading bulk pixel data
    # print('Reading image...')
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)

    try:
        file_reader.ReadImageInformation()
    except:
        print('ERROR while reading: ', file_name)
        print('SKIP file')
        return

    # Get the sorted file names, opens all files in the directory and reads the meta-information
    # without reading the bulk pixel data
    series_ID = file_reader.GetMetaData('0020|000e')
    description = file_reader.GetMetaData('0008|103e')
    # print('seriesId', series_ID, '\t\t descr', description)

    if series_ID not in series_list:
        series_list.append(series_ID)
        desc_list.append(description)

    return series_ID


def organize_series(study_folder_path):
    print(study_folder_path)
    for (root, dirs, files) in os.walk(study_folder_path):
        print('root', root)
        print('dirs', len(dirs))
        print('files', len(files))

        data_directory = root

        series_list = []
        desc_list = []

        for f in range(len(files)):
            print('file', f, '/', len(files))
            path = os.path.join(root, files[f])
            getImageSeriesId(path, series_list, desc_list)

        print('\n\n---------------------------\n\n')
        print(series_list, desc_list)
        print('\n')
        print(len(series_list), len(desc_list))
        print('\n\n---------------------------\n\n')

        for n in range(len(series_list)):
            series_ID = series_list[n]
            description = desc_list[n]
            sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                data_directory, series_ID)
            print('series', series_ID, '\tdesc', description,
                  '\tnumber of files', len(sorted_file_names))
            for file_path in sorted_file_names:
                f = os.path.basename(file_path)
                target_dir = os.path.dirname(file_path)
                moveIntoFolder(f, description, target_dir)


def unzip(zip_file):
    print("[*] Beginning extraction process...")
    # parent = os.path.dirname(zip_file)
    # basename = os.path.splitext(zip_file)[0]
    # out_folder = os.path.join(basename, 'DICOM')
    zip = zipfile.ZipFile(zip_file)
    zip.setpassword(b'ar_unibg')
    for i, f in enumerate(zip.filelist):
        f.filename = os.path.join('DICOM_C2', 'extracted_{0:03}'.format(i))
        zip.extract(f)
        print("--- Extracted '%s'" % (f.filename))

    print("[*] Done")


if __name__ == "__main__":
    # init arg parser

    parser = argparse.ArgumentParser(
        description='Extract lung values from given images and store output in target csv file')

    parser.add_argument('--examine', action='store',
                        help='examine passed csv file')

    parser.add_argument('--tresholds', action='store', nargs='+', type=float,
                        help='array of tresholds')

    parser.add_argument('--plot', action='store',
                        help='plot passed csv file')

    parser.add_argument('--organize', action='store',
                        help='organize passed study folder into series subfolders')

    parser.add_argument('--unzip', action='store',
                        help='organize passed study folder into series subfolders')

    args = parser.parse_args()

    if (args.plot):
        plot(args.plot)
    elif (args.organize):
        organize_series(args.organize)
    elif (args.examine and args.tresholds):
        print(args.tresholds)
        examine_treshold(args.examine, args.tresholds)
    elif (args.examine):
        examine(args.examine)
    elif (args.unzip):
        unzip(args.unzip)
