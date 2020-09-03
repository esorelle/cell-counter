# import numpy as np
# from sklearn.cluster import spectral_clustering
# from sklearn.feature_extraction import image as sk_image
# import matplotlib.pyplot as plt
# import cv2
#
#
# def split_multi_cell(signal_img, multi_cell_mask, max_cell_area, plot=False):
#     contour_tree, hierarchy = cv2.findContours(
#         multi_cell_mask.astype(np.uint8),
#         cv2.RETR_CCOMP,
#         cv2.CHAIN_APPROX_SIMPLE
#     )
#     single_cell_contours = []
#     sub_multi_contours_idx = []
#     for c_idx, sub_c in enumerate(contour_tree):
#         h = hierarchy[0][c_idx]
#         if h[3] != -1:
#             # it's a child contour, ignore it
#             continue
#         sub_c_area = cv2.contourArea(sub_c)
#         if sub_c_area < max_cell_area * .33:
#             # too small, some kind of fragment
#             continue
#         elif sub_c_area > max_cell_area * 1.1:
#             # too big for a single cell, try to split it
#             sub_multi_contours_idx.append(c_idx)
#         else:
#             # just right, probably a cell so save it
#             single_cell_contours.append(sub_c)
#             if plot:
#                 sc_mask = np.zeros(signal_img.shape, dtype=np.uint8)
#                 cv2.drawContours(
#                     sc_mask,
#                     [sub_c],
#                     -1,
#                     255,
#                     cv2.FILLED
#                 )
#                 plt.figure(figsize=(16, 16))
#                 plt.imshow(sc_mask)
#                 plt.axis('off')
#                 plt.show()
#     for c_idx in sub_multi_contours_idx:
#         mc_mask = np.zeros(signal_img.shape, dtype=np.uint8)
#         cv2.drawContours(
#             mc_mask,
#             contour_tree,
#             c_idx,
#             255,
#             cv2.FILLED,
#             hierarchy=hierarchy
#         )
#         # Convert the image into a graph with the value of the gradient on the
#         # edges.
#         region_graph = sk_image.img_to_graph(
#             signal_img,
#             mask=mc_mask.astype(np.bool)
#         )
#         # Take a decreasing function of the gradient: we take it weakly
#         # dependent from the gradient the segmentation is close to a voronoi
#         region_graph.data = np.exp(-region_graph.data / region_graph.data.std())
#         n_clusters = 2
#         labels = spectral_clustering(
#             region_graph,
#             n_clusters=n_clusters,
#             eigen_solver='arpack',
#             n_init=10
#         )
#         label_im = np.full(mc_mask.shape, -1.)
#         label_im[mc_mask.astype(np.bool)] = labels
#         if plot:
#             plt.figure(figsize=(16, 16))
#             plt.imshow(label_im)
#             plt.axis('off')
#             plt.show()
#         for label in range(n_clusters):
#             new_mask = label_im == label
#             single_cell_contours.extend(
#                 split_multi_cell(signal_img, new_mask, max_cell_area, plot=plot)
#             )
#     return single_cell_contours

# SPECTRAL CLUSTER MULTI CELL SPLITTING (SCOTT'S LUNGMAP METHOD) ###
# split_cell_contours = utils.split_multi_cell(input_img, gate_img, max_cell_area, plot=False)
# print('# contours from spectral clustering: ', len(split_cell_contours))
#
# cell_split_points = []
#
# for i, apt in enumerate(regionprops(apt_blobs)):
#     apt_coords = [tuple(j) for j in apt.coords]
#     for c in split_cell_contours:
#         M = cv2.moments(c)
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
#         test_point = tuple([cy, cx])
#         if test_point in apt_coords:
#             chamber_cell_count_array_contours[i] += 1
#             cell_split_points.append(test_point)  # use for scatter plot of counted cells
#
# fig, ax = plt.subplots(figsize=(10, 6))
# # subbed new_img for input_img to show detected apartment and text regions
# ax.imshow(new_img, cmap='gray')
#
# for point in cell_split_points:
#     ax.scatter(point[1], point[0], s=4, c='lightcoral', marker='x')
#
# ax.set_axis_off()
# plt.title('#_of_detected_cell_contours_in_image = ' + str(len(cell_split_points)))
# plt.tight_layout()
# plt.savefig(img_name + '_detected_split_cells' + '.png')
# plt.close()
# END SPECTRAL CLUSTER SPLITTING ###




# mask = np.load('test_apt_mask.npy')
# img = np.load('test_scaled_img.npy')
# gate = np.load('test_thresh_gated_cells.npy')
# gate = gate.astype(np.uint8)
#
# contour_tree, hierarchy = cv2.findContours(
#     gate.astype(np.uint8),
#     cv2.RETR_CCOMP,
#     cv2.CHAIN_APPROX_SIMPLE
# )
#
# filtered_contours = []
# for contour in contour_tree:
#     area = cv2.contourArea(contour)
#     if 15 < area < 300:
#         filtered_contours.append(contour)
#
# print(len(contour_tree))
# print(len(filtered_contours))
#
# image = cv2.imread('BF_ST_080_APT_004_20190315115128.tif')
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.namedWindow("cell contours", cv2.WINDOW_NORMAL)
# cv2.drawContours(image, filtered_contours, -1, (0,255,0), 1)
# cv2.imshow("cell contours", image)
# cv2.waitKey(0)
