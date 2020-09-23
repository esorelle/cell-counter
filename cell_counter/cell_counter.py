import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt

from cell_counter import core

VERSION = 'v20200920'


def process_directory_extract_apartments(
        target_directory,
        flip_horizontal=False,
        save_digit_images=False
):
    time_stamp = dt.now().strftime('%Y_%m_%d_%H_%M_%S')

    save_path = 'extracted_data_%s' % time_stamp
    save_path = os.path.join(target_directory, save_path)
    os.mkdir(save_path)

    images = glob.glob(os.path.join(target_directory, '*.tif'))
    img_count = len(images)

    if img_count == 0:
        raise (FileNotFoundError("No TIFF images found in %s" % target_directory))

    if save_digit_images:
        digit_dir = os.path.join(save_path, 'digits')
        os.mkdir(digit_dir)
    else:
        digit_dir = None

    apt_region_dir = os.path.join(save_path, 'apt_regions')
    os.mkdir(apt_region_dir)

    apt_data_list = []

    for i, img_path in enumerate(images):
        img_base_name = os.path.basename(img_path)
        print(i + 1, img_base_name)

        apt_data = core.identify_apartments(
            img_path,
            flip_horizontal=flip_horizontal,
            digit_dir=digit_dir,
            apt_region_dir=apt_region_dir
        )

        apt_data_list.append(apt_data)

    return apt_data_list


def process_directory(
        target_directory,
        min_cell_area,
        max_cell_area,
        flip_horizontal=False,
        save_process_pics=False,
        count_hist=False
):
    time_stamp = dt.now().strftime('%Y_%m_%d_%H_%M_%S')

    save_path = 'analysis_%s' % time_stamp
    save_path = os.path.join(target_directory, save_path)
    os.mkdir(save_path)

    images = glob.glob(os.path.join(target_directory, '*.tif'))
    img_count = len(images)

    if img_count == 0:
        raise(FileNotFoundError("No TIFF images found in %s" % target_directory))

    if save_process_pics:
        apt_fig_dir = os.path.join(save_path, 'apartment_figures')
        os.mkdir(apt_fig_dir)
        fid_fig_dir = os.path.join(save_path, 'fiducial_figures')
        os.mkdir(fid_fig_dir)
    else:
        apt_fig_dir = None
        fid_fig_dir = None

    total_apt_count = 0
    apartments_per_image = []
    apt_data_df_list = []

    for i, img_path in enumerate(images):
        img_base_name = os.path.basename(img_path)
        print(i + 1, img_base_name)

        apt_data = core.identify_apartments(
            img_path,
            flip_horizontal=flip_horizontal,
            fiducial_dir=fid_fig_dir
        )

        apt_count = len(apt_data)
        print('\t%d chambers counted' % apt_count)

        apt_data = core.extract_cell_data(
            apt_data,
            min_cell_area,
            max_cell_area
        )

        if apt_fig_dir is not None:
            for apt in apt_data:
                fig = core.render_apartment(apt)
                fig_name = "_".join(['apt_fig', apt['image_name'], apt['row_address'], apt['col_address']])
                fig_name = '.'.join([fig_name, 'png'])
                fig.savefig(os.path.join(apt_fig_dir, fig_name))
                plt.close()

        total_apt_count += apt_count
        apartments_per_image.append(apt_count)

        df_apt_data = pd.DataFrame(
            apt_data,
            columns=[
                'image_name',
                'row_address',
                'col_address',
                'fid_x',
                'fid_y',
                'edge_blob_area',
                'edge_blob_apt_ratio',
                'edge_blob_count',
                'edge_cell_count_min',
                'edge_cell_count_max',
                'non_edge_blob_area',
                'non_edge_blob_apt_ratio',
                'non_edge_blob_count',
                'non_edge_cell_count_min',
                'non_edge_cell_count_max'
            ]
        )

        apt_data_df_list.append(df_apt_data)

    all_apt_data_df = pd.concat(apt_data_df_list, ignore_index=True)

    if count_hist:
        # TODO: can we show both min & max cell counts from the simple apt capacity method?
        plt.hist(all_apt_data_df['edge_cell_count_min'], color='lightcoral', bins=35)
        plt.title('Simple Cell Count - Min')
        plt.xlabel('cell count')
        plt.ylabel('# of chambers on chip')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '_simple_cell_count_min_histogram.png'))
        plt.close()

        plt.hist(apartments_per_image, color='dodgerblue', bins=18)
        plt.title('Apartments Per Image')
        plt.xlabel('# of detected apartments')
        plt.ylabel('image count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '_summary_apartment_count_histogram.png'))
        plt.close()

    all_apt_data_df.to_csv(
        os.path.join(
            save_path,
            'directory_cell_counts_%s.csv' % VERSION
        )
    )

    # We expect roughly 24 apartments to be fully visible in each image
    naive_chamber_id_rate = np.around(total_apt_count / (img_count * 24.), decimals=4)

    metadata = open(os.path.join(save_path, 'analysis_metadata.txt'), 'w')
    metadata_str = "\n".join(
        [
            'data_location: %s' % target_directory,
            'datetime: %s' % time_stamp,
            'version: %s' % VERSION,
            '',
            'save_processing_pics: %s' % save_process_pics,
            'count_histogram: %s' % count_hist,
            '# of images analyzed: %d' % img_count,
            'total # of chambers detected: %d' % total_apt_count,
            'naive_chamber_id_rate: %f%%' % (naive_chamber_id_rate * 100),
            'min_cell_area: %d' % min_cell_area,
            'max_cell_area: %d' % max_cell_area
        ]
    )
    metadata.write(metadata_str)
    metadata.close()

    print("Naive chamber detection rate for chip: %.2f%%\n" % (naive_chamber_id_rate * 100))

    return all_apt_data_df
