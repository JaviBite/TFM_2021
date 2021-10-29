# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:18:44 2021

https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hoghistogram.pyx

@author: cvlab
"""

from skimage.draw import line
import numpy as np

def cell_hog(magnitude, orientation,
              orientation_start, orientation_end,
              cell_columns, cell_rows, 
              column_index, row_index,
              size_columns, size_rows,
              range_rows_start, range_rows_stop,
              range_columns_start, range_columns_stop):
             
             
    

    total = 0.

    ini = np.int(range_rows_start)
    fin = np.int(range_rows_stop)
    for cell_row in range(ini, fin):
        cell_row_index = np.int(row_index + cell_row)
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue
        ini2= np.int(range_columns_start)
        fin2= np.int(range_columns_stop)
        for cell_column in range (ini2, fin2):
            cell_column_index = np.int(column_index + cell_column)
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)


def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')

    return out

def normalize(orientation_histogram, cells_per_block=(3, 3), block_norm='L2-Hys', number_of_orientations=9):

    n_cells_row, n_cells_col = orientation_histogram.shape[:2]
    b_row, b_col = cells_per_block

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1

    """
    The next stage computes normalization, which takes local groups of
    cells and contrast normalizes their overall responses before passing
    to next stage. Normalization introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalize each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalizations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalizations. This may seem redundant but it improves the performance.
    We refer to the normalized block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, number_of_orientations),
        dtype=float
    )

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                _hog_normalize_block(block, method=block_norm)



    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """
      

    normalized_blocks = normalized_blocks.ravel()

    return normalized_blocks





def hog(magnitude, orientation, number_of_orientations=9, pixels_per_cell=(16, 16), 
        cells_per_block=(3, 3), block_norm='L2-Hys', visualize='true'):
    

    """
    The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """    
    
    s_row, s_col = magnitude.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, number_of_orientations), dtype=float)
    
    size_columns = s_col
    size_rows = s_row
    cell_columns = c_col
    cell_rows = c_row    
    
    
    r_0 = cell_rows / 2
    c_0 = cell_columns / 2
    cc = cell_rows * n_cells_row
    cr = cell_columns * n_cells_col
    range_rows_stop = (cell_rows + 1) / 2
    range_rows_start = -(cell_rows / 2)
    range_columns_stop = (cell_columns + 1) / 2
    range_columns_start = -(cell_columns / 2)
    number_of_orientations_per_180 = 180. / number_of_orientations


    # compute orientations integral images
    for i in range(number_of_orientations):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        c = c_0
        r = r_0
        r_i = 0
        c_i = 0
        
        #print(cc)
        #print(cr)
        
        
        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                orientation_histogram[r_i, c_i, i] = \
                    cell_hog(magnitude, orientation,
                             orientation_start, orientation_end,
                             cell_columns, cell_rows, c, r,
                             size_columns, size_rows,
                             range_rows_start, range_rows_stop,
                             range_columns_start, range_columns_stop)
                c_i += 1
                c += cell_columns
                #print(c)

            r_i += 1
            r += cell_rows
            # print(c)

    # now compute the histogram for each cell
    hog_image = np.zeros((n_cells_row, n_cells_col), dtype=float)

    if visualize:

        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(number_of_orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / number_of_orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = line(int(centre[0] - dc),
                                  int(centre[1] + dr),
                                  int(centre[0] + dc),
                                  int(centre[1] - dr))
                    hog_image[rr, cc] += orientation_histogram[r, c, o]

    if visualize:
        return orientation_histogram, hog_image
    else:
        return orientation_histogram 
                    
                    

    """
    The next stage computes normalization, which takes local groups of
    cells and contrast normalizes their overall responses before passing
    to next stage. Normalization introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalize each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalizations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalizations. This may seem redundant but it improves the performance.
    We refer to the normalized block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, number_of_orientations),
        dtype=float
    )

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                _hog_normalize_block(block, method=block_norm)



    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """
      

    normalized_blocks = normalized_blocks.ravel()
    # normalized_blocks_out = np.zeros((normalized_blocks.shape[0], normalized_blocks.shape[1], number_of_orientations), dtype=float)

    # for r in range(normalized_blocks.shape[0]):
    #         for c in range(normalized_blocks.shape[1]):
    #             for o in orientations_arr:
    #                 normalized_blocks_out[r, c, o] = np.mean(normalized_blocks[r,c,:,:,o])

    

    # if visualize:

    #     radius = min(c_row, c_col) // 2 - 1
    #     orientations_arr = np.arange(number_of_orientations)
    #     # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    #     orientation_bin_midpoints = (
    #         np.pi * (orientations_arr + .5) / number_of_orientations)
    #     dr_arr = radius * np.sin(orientation_bin_midpoints)
    #     dc_arr = radius * np.cos(orientation_bin_midpoints)
    #     hog_image = np.zeros((s_row, s_col), dtype=float)
    #     for r in range(normalized_blocks.shape[0]):
    #         for c in range(normalized_blocks.shape[1]):
    #             for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
    #                 centre = tuple([r * c_row + c_row // 2,
    #                                 c * c_col + c_col // 2])
    #                 rr, cc = line(int(centre[0] - dc),
    #                               int(centre[1] + dr),
    #                               int(centre[0] + dc),
    #                               int(centre[1] - dr))
    #                 hog_image[rr, cc] += normalized_blocks_out[r, c, o]

    if visualize:
        return orientation_histogram, hog_image
    else:
        return orientation_histogram 

    if visualize:
        return normalized_blocks_out, hog_image * 255
    else:
        return normalized_blocks    
    
    










