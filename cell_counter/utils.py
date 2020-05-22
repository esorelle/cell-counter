import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from warnings import warn

dig_refs = []
for _ref_digit in range(10):
    dig_ref = Image.open('resources/dig_ref_%d.tif' % _ref_digit)
    dig_ref = np.asarray(dig_ref)
    dig_refs.append(dig_ref)


# defined image flip
def flip_horizontal(input_img):
    flipped_img = input_img[:, ::-1]
    return flipped_img


# defined rotation function
def rotate(point, origin=(0, 0), degrees=0):
    angle = np.deg2rad(-degrees)
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


# identify digits from templates
def identify_digit(dig_region):
    # padding is crucial & needs to be about 1/2  of the template width/height
    # changes from median padding to avoid error...see if this works
    dig_region_pad = np.pad(dig_region, 10, mode='constant', constant_values=(128, 128))
    scores = []
    for i, ref_digit in enumerate(dig_refs):
        res = cv2.matchTemplate(dig_region_pad, ref_digit, cv2.TM_CCOEFF_NORMED)
        scores.append(res.max())

    if np.max(scores) > 0.6:
        return np.argmax(scores), np.max(scores)
    else:
        return '_', np.max(scores)


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
