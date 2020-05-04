import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pytesseract
import skimage as sk
from datetime import datetime as dt
from scipy import ndimage as ndi
from skimage import feature, filters, morphology, util
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


apartment_workflow = 'denoise -> norm -> edges -> dilate -> invert -> maxout -> close -> extentout -> open -> minout -> label -> borderout -> rect_mask'
cell_workflow = 'denoise -> norm -> tophat -> blur -> distance -> label -> centroids'


def flip_horizontal(input_img):
    flipped_img = input_img[:,::-1]
    return flipped_img


# cribbed from skimage to create remove_large_objects function
def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


def remove_large_objects(ar, max_size=64, connectivity=1, in_place=False):
    # Raising type error if not int or bool
    _check_dtype_supported(ar)
    if in_place:
        out = ar
    else:
        out = ar.copy()
    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out
    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")
    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0
    return out


# new cell count method that uses chamber blobs to do direct gating
def get_chamber_cell_counts_bf(input_img, img_name, gauss_blur_sigma, window_thresh, scaling_thresh, min_blob_area, max_blob_area, min_blob_extent, tophat_selem, min_cell_area, max_cell_area, save_process_pics):
    # find the chambers
    img_blur = filters.gaussian(input_img, gauss_blur_sigma)
    back_map = filters.threshold_local(img_blur, window_thresh, offset=0)
    img_scaled = np.divide(img_blur, back_map) * 255
    edges_0 = (img_scaled < scaling_thresh)
    edges_1 = morphology.dilation(edges_0, morphology.selem.rectangle(1,3)) #2,3 --> changed to 1,3 to reduce connectivity of dense cells leading to false negative for very full apartments while retaining connection of apartment entry points.
    blobs_0 = util.invert(edges_1)
    blobs_1 = remove_large_objects(blobs_0, max_blob_area)
    blobs_2 = morphology.closing(blobs_1, morphology.selem.rectangle(4,1)) #3,1 --> changed to 4,1 to increase blob grouping of dense cells as contiguous apartment
    blobs_3 = np.zeros(np.shape(input_img))
    N, W, S, E = [], [], [], []
    true_N = []
    for blob in regionprops(label(blobs_2)):
        if blob.extent > min_blob_extent:
            _N, _W, _S, _E = blob.bbox[0], blob.bbox[1], blob.bbox[2], blob.bbox[3]
            if np.logical_and(70 < _E - _W < 100, 180 < _S - _N < 240):         # new condition based on bounding box dimensions -- remove if needed
                N.append(_N)
                W.append(_W - 3)
                S.append(_S)
                E.append(_E + 3)
                true_N.append(_S - 212)
                hull = blob.convex_image
                blobs_3[_N:_S,_W:_E] = hull
    blobs_4 = morphology.opening(blobs_3, morphology.selem.disk(3))
    blobs_5 = blobs_4 > 0
    blobs_6 = morphology.remove_small_objects(blobs_5, min_blob_area)
    blobs_7 = label(blobs_6, connectivity=2)
    refined_blobs = np.zeros(np.shape(input_img))
    N, W, S, E = [], [], [], []
    true_N = []
    for blob in regionprops(blobs_7):
        if abs(blob.centroid[0] - np.shape(input_img)[0]) > 50 and blob.centroid[0] > 50:
            if abs(blob.centroid[1] - np.shape(input_img)[1]) > 50 and blob.centroid[1] > 50:
                _N, _W, _S, _E = blob.bbox[0], blob.bbox[1], blob.bbox[2], blob.bbox[3]
                N.append(_N)
                W.append(_W - 3)
                S.append(_S)
                E.append(_E + 3)
                true_N.append(_S - 212)
                hull = blob.convex_image
                refined_blobs[_N:_S,_W:_E] = hull
    # make rectangles (or insert chamber polygon ndarray with reference to anchor point)
    rect_mask = np.zeros(np.shape(input_img))
    for i in range(len(true_N)):
        rect_mask[S[i]-212:S[i]-15, W[i]:E[i]] = 1
    # count the cells in each chamber
    img_blur = filters.gaussian(input_img, 1)
    back_map = filters.threshold_local(img_blur, window_thresh, offset=0)
    img_scaled = np.divide(img_blur, back_map) * 255
    selem = morphology.disk(tophat_selem)
    img_scaled = img_scaled / 255
    b0 = morphology.white_tophat(img_scaled, selem)
    b1 = b0 * (b0 > 0.035)
    b2 = filters.gaussian(b1, sigma=1.25)
    b3 = b2 > 0.05
    distance = ndi.distance_transform_edt(b3)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((11, 11)), labels=b3)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=b3)
    # optional save of process step images
    if save_process_pics == 1:
        analysis_folder = os.getcwd()
        image_save_path = img_name + '_process_steps'
        os.mkdir(image_save_path)
        os.chdir(image_save_path)
        plt.imshow(edges_0, cmap='gray')
        plt.title('001_initial_edges')
        plt.tight_layout()
        plt.savefig(img_name + '001_initial_edges' + '.png')
        plt.close()
        plt.imshow(edges_1, cmap='gray')
        plt.title('002_dilated_edges')
        plt.tight_layout()
        plt.savefig(img_name + '002_dilated_edges' + '.png')
        plt.close()
        plt.imshow(blobs_0, cmap='gray')
        plt.title('003_initial_blobs')
        plt.tight_layout()
        plt.savefig(img_name + '003_initial_blobs' + '.png')
        plt.close()
        plt.imshow(blobs_1, cmap='gray')
        plt.title('004_max_area_filtered_blobs')
        plt.tight_layout()
        plt.savefig(img_name + '004_max_area_filtered_blobs' + '.png')
        plt.close()
        plt.imshow(blobs_2, cmap='gray')
        plt.title('005_closed_blobs')
        plt.tight_layout()
        plt.savefig(img_name + '005_closed_blobs' + '.png')
        plt.close()
        plt.imshow(blobs_4, cmap='gray')
        plt.title('006_extent_filtered_opened_blobs')
        plt.tight_layout()
        plt.savefig(img_name + '006_extent_filtered_opened_blobs' + '.png')
        plt.close()
        plt.imshow(blobs_6, cmap='gray')
        plt.title('007_min_area_filtered_blobs')
        plt.tight_layout()
        plt.savefig(img_name + '007_min_area_filtered_blobs' + '.png')
        plt.close()
        plt.imshow(blobs_7, cmap='nipy_spectral')
        plt.title('008_labeled_proto_chambers')
        plt.tight_layout()
        plt.savefig(img_name + '008_labeled_proto_chambers' + '.png')
        plt.close()
        plt.imshow(label(refined_blobs, connectivity=2), cmap='nipy_spectral')
        plt.title('009_labeled_refined_chambers')
        plt.tight_layout()
        plt.savefig(img_name + '009_labeled_refined_chambers' + '.png')
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(input_img, cmap='gray')
        for i in range(len(true_N)):
            patch = mpatches.Rectangle((W[i],S[i]), E[i] - W[i], -212, fill=False, edgecolor='turquoise', linewidth=2)
            ax.add_patch(patch)
        plt.title('010_bounded_chambers_overlay')
        plt.tight_layout()
        plt.savefig(img_name + '010_bounded_chambers_overlay' + '.png')
        plt.close()
        os.chdir('..')
    # show detected centroids overlay on input image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(input_img, cmap='gray')
    for i in range(len(true_N)):
        patch = mpatches.Rectangle((W[i],S[i]), E[i] - W[i], -212, fill=False, edgecolor='dodgerblue', linewidth=2)
        ax.add_patch(patch)
    x_coords, y_coords = [], []
    for region in regionprops(labels):
        if min_cell_area < region.area < max_cell_area:
            cent_x, cent_y = region.centroid
            if rect_mask[int(np.round(cent_x)), int(np.round(cent_y))] > 0:
                y_coords.append(int(np.round(cent_x)))
                x_coords.append(int(np.round(cent_y)))
                ax.scatter(cent_y, cent_x, s=4, c='lightcoral', marker='x')
    ax.set_axis_off()
    plt.title('#_of_detected_cells_in_image = ' + str(len(x_coords)))
    plt.tight_layout()
    plt.savefig(img_name + '_detected_cells' + '.png')
    plt.close()
    # tabulate the counted cells by chamber for output
    prefix = img_name[img_name.find('ST_'):img_name.find('ST_') + 14] + '_CHAMBER_'
    address_counts = {}
    chamber_cell_count_array = np.zeros(len(N))
    for p in range(len(x_coords)):
        for c in range(len(true_N)):
            if W[c] < x_coords[p] < E[c]:
                if S[c] - 212 < y_coords[p] < S[c]-15:
                    chamber_cell_count_array[c] += 1
    for chamber in range(len(chamber_cell_count_array)):
        if chamber < 9:
            address_counts[prefix + '00' + str(chamber + 1)] = int(chamber_cell_count_array[chamber])
        else:
            address_counts[prefix + '0' + str(chamber + 1)] = int(chamber_cell_count_array[chamber])
    return address_counts


# directory wrapper
def process_directory_relative_id(flip, gauss_blur_sigma, window_thresh, scaling_thresh, min_blob_area, max_blob_area, min_blob_extent, tophat_selem, min_cell_area, max_cell_area, save_process_pics, count_hist, targetdirectory):
    version = '_v20200424'
    cwd = targetdirectory
    os.chdir(cwd)
    save_path = 'analysis_' + dt.now().strftime('%Y_%m_%d_%H_%M')
    os.mkdir(save_path)
    images = glob.glob('./*.tif')
    images = [r[2:] for r in images]
    num_images = len(images)
    cell_counts_df = pd.DataFrame(columns = ['Folder_Name', 'Image_Name', 'Chamber_ID', 'Detected_Cells'])
    num_chambers_detected = 0
    apartments_per_image = []
    for i in range(0,num_images):
        img_name = images[i]
        raw_img = plt.imread(img_name)
        if flip == True:
            raw_img = flip_horizontal(raw_img)
        os.chdir(save_path)
        address_counts = get_chamber_cell_counts_bf(raw_img, img_name, gauss_blur_sigma, window_thresh, scaling_thresh, min_blob_area, max_blob_area, min_blob_extent, tophat_selem, min_cell_area, max_cell_area, save_process_pics)
        os.chdir('..')
        print(img_name + ': ' + str(len(address_counts)) + ' chambers counted')
        num_chambers_detected += len(address_counts)
        apartments_per_image.append(len(address_counts))
        for chamber_key in address_counts:
            cell_counts_df = cell_counts_df.append({'Folder_Name': cwd, 'Image_Name': img_name, 'Chamber_ID': chamber_key, 'Detected_Cells': address_counts[chamber_key]}, ignore_index=True)
    if count_hist == 1:
        os.chdir(save_path)
        plt.hist(cell_counts_df['Detected_Cells'], color='lightcoral', bins=35)
        plt.title('summary_cell_count_histogram')
        plt.xlabel('number_of_detected_cells')
        plt.ylabel('number_of_chambers_on_chip')
        plt.tight_layout()
        plt.savefig('_summary_cell_count_histogram' + '.png')
        plt.close()
        plt.hist(apartments_per_image, color='dodgerblue', bins=18)
        plt.title('apartments_per_image_in_directory')
        plt.xlabel('number_of_detected_apartments')
        plt.ylabel('number_of_images')
        plt.tight_layout()
        plt.savefig('_summary_apartment_count_histogram' + '.png')
        plt.close()
        os.chdir('..')
    os.chdir(save_path)
    cell_counts_df.to_csv('_directory_cell_counts_relative_chamber_id' + version + '.csv')
    naive_chamber_id_rate = np.around(num_chambers_detected / (num_images * 24), decimals=4)
    metadata = open('analysis_metadata.txt', 'w')
    metadata.write('data_location: ' + targetdirectory + '\n')
    metadata.write('datetime: ' + dt.now().strftime('%Y_%m_%d_%H_%M') + '\n')
    metadata.write('version: ' + version + '\n' + '\n')
    metadata.write('apartment-finding method: ' + apartment_workflow + '\n')
    metadata.write('cell-counting method: ' + cell_workflow + '\n')
    if flip == True:
        metadata.write('\n' + 'image_flip: True')
    else:
        metadata.write('\n' + 'image_flip: False')
    if save_process_pics == 0:
        metadata.write('\n' + 'save_processing_pics: False')
    else:
        metadata.write('\n' + 'save_processing_pics: True')
    if count_hist == 0:
        metadata.write('\n' + 'directory_chamber_cell_count_histogram: False')
    else:
        metadata.write('\n' + 'directory_chamber_cell_count_histogram: True' + '\n')
    metadata.write('\n')
    metadata.write('# of images analyzed: ' + str(num_images) + '\n')
    metadata.write('total # of chambers detected: ' + str(num_chambers_detected) + '\n')
    metadata.write('naive_chamber_id_rate: ' + str(naive_chamber_id_rate * 100) + '%' + '\n')
    metadata.write('\n')
    metadata.write('window_thresh: ' + str(window_thresh) + '\n')
    metadata.write('gauss_blur_sigma: ' + str(gauss_blur_sigma) + '\n')
    metadata.write('scaling_thresh: ' + str(scaling_thresh) + '\n')
    metadata.write('min_blob_area: ' + str(min_blob_area) + '\n')
    metadata.write('max_blob_area: ' + str(max_blob_area) + '\n')
    metadata.write('min_blob_extent: ' + str(min_blob_extent) + '\n')
    metadata.write('tophat_selem: ' + str(tophat_selem) + '\n')
    metadata.write('min_cell_area: ' + str(min_cell_area) + '\n')
    metadata.write('max_cell_area: ' + str(max_cell_area) + '\n')
    metadata.close()
    os.chdir('..')
    print("Naive chamber detection rate for chip: " + str(naive_chamber_id_rate * 100) + '%')
    return cell_counts_df
