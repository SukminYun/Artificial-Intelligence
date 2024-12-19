import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    # differential filter
    filter_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]) / 3
        
    filter_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ]) / 3
        
    return filter_x, filter_y


def filter_image(im, filter):
    m, n = im.shape
    k = filter.shape[0]

    # define padding size
    pad = k // 2
    im_filtered= np.zeros_like(im)

    for i in range(m):
        for j in range(n):
            sum = 0
            for fi in range(k):
                for fj in range(k):
                    ii = i - pad + fi
                    jj = j - pad + fj
                    
                    # image boundary check
                    if ii < 0 or ii >= m or jj < 0 or jj >= n:
                        # if out of boundary, use 0
                        pixel_value = 0
                    else:
                        pixel_value = im[ii, jj]
                    
                    sum += pixel_value * filter[fi, fj]
            
            im_filtered[i, j] = sum

    return im_filtered


def get_gradient(im_dx, im_dy):
    # Compute gradient magnitude and
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    
    # Convert angle to range [0, 180)
    grad_angle = np.arctan2(im_dy, im_dx) * (180) / np.pi 
    grad_angle = grad_angle % 180
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # Compute cell size
    M, N = grad_mag.shape
    M = M // cell_size
    N = N // cell_size
    ori_histo = np.zeros((M, N, 6))
    
    # Define angle ranges
    angle_ranges = [
        (165, 180), 
        (0, 15),
        (15, 45),
        (45, 75),
        (75, 105),
        (105, 135),
        (135, 165)
    ]
    
    # Compute histogram
    for i in range(M):
        for j in range(N):
            for u in range(cell_size):
                for v in range(cell_size):
                    x = i * cell_size + u
                    y = j * cell_size + v
                    angle = grad_angle[x, y]
                    magnitude = grad_mag[x, y]
                    
                    # Find the bin to which the angle belongs
                    for k, (start, end) in enumerate(angle_ranges):
                        if k == 0:  # first bin includes both (165, 180) and (0, 15)
                            if start <= angle < end:
                                ori_histo[i, j, 0] += magnitude
                                break

                        elif start <= angle < end:
                            ori_histo[i, j, k-1] += magnitude
                            break

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # Compute block descriptor
    M, N, _ = ori_histo.shape
    new_M = M - block_size + 1
    new_N = N - block_size + 1
    ori_histo_normalized = np.zeros((new_M, new_N, 6 * block_size**2))

    for i in range(new_M):
        for j in range(new_N):
            # flatten elements in 2 x 2 block
            block = ori_histo[i:i+block_size, j:j+block_size, :].flatten()
            
            # L2 normalization
            norm = np.sqrt(np.sum(block**2) + 1e-6)  # e = 0.001
            block_normalized = block / norm
            
            # assign to ori_histo_normalized
            ori_histo_normalized[i, j, :] = block_normalized

    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.savefig('hog.png')
    plt.show()


def extract_hog(im, visualize=False, cell_size=8, block_size=2):
    # conver to float and normalize to range [0, 1]
    im = im.astype(float)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))

    # get differential filter
    filter_x, filter_y = get_differential_filter()

    # compute the gradient of the image
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)

    # compute gradient magnitude and angle
    im_mag, im_angle = get_gradient(im_dx, im_dy)

    # build histogram of oriented gradients
    ori_histo = build_histogram(im_mag, im_angle, cell_size)

    # build the desciptor of all blocks with normalization
    hog = get_block_descriptor(ori_histo, block_size)

    # Visualize the filtered images
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title('Filtered Image - Dx')
    # plt.imshow(im_dx, cmap='jet', vmin=0, vmax=np.max(im_dx))
    # plt.subplot(1, 2, 2)
    # plt.title('Filtered Image - Dy')
    # plt.imshow(im_dy, cmap='jet', vmin=0, vmax=np.max(im_dy))
    # plt.show()

    # # Visualize the gradient magnitude and angle
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title('Magnitude')
    # plt.imshow(im_mag, cmap='jet', vmin=0, vmax=1)
    # plt.subplot(1, 2, 2)
    # plt.title('Angle')
    # plt.imshow(im_angle, cmap='jet', vmin=0, vmax=180)
    # plt.show()

    if (visualize):
        visualize_hog(im, hog, cell_size, block_size)

    # return a long vector of hog by concatenating all block descriptors
    return hog


def face_recognition(I_target, I_template):
    h, w = I_target.shape
    M, N = I_template.shape

    hog_template = extract_hog(I_template, visualize=False)

    bounding = []

    for y in range(0, h - M + 1):
        for x in range(0, w - N + 1):
            # Extract HOG feature of the window
            hog_window = extract_hog(I_target[y:y+M, x:x+N], visualize=False)
            
            # normalize HOG feature
            hog_window_norm = (hog_window - np.mean(hog_window)) 
            hog_template_norm = (hog_template - np.mean(hog_template))
            norm_hog_window_norm = np.sqrt(np.sum(hog_window_norm**2))
            norm_hog_template_norm = np.sqrt(np.sum(hog_template_norm**2))

            # Compute normalized cross-correlation (NCC) score
            ncc_score = (np.dot(hog_window_norm.flatten(), hog_template_norm.flatten())) / (norm_hog_window_norm * norm_hog_template_norm)

            # Thresholding
            if ncc_score > 0.48: 
                bounding.append([x, y, ncc_score])
    
    bounding = np.array(bounding)

    if len(bounding) == 0:
        return []
    
    # sort bounding boxes by score
    bounding = bounding[bounding[:, 2].argsort()[::-1]]
    x1 = bounding[ : ,0]
    y1 = bounding[ : ,1]
    scores = bounding[ : ,2]

    area = M * N
    indices = np.argsort(scores)[::-1]

    keep = []
    # Non-maximum suppression
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x1[i] + M, x1[indices[1:]] + N)
        yy2 = np.minimum(y1[i] + M, y1[indices[1:]] + N)

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (area + area - inter)

        # with IoU 50% 
        indices = indices[np.where(iou <= 0.5)[0] + 1]

    bounding_boxes = []

    for k in range(len(keep)):
        bounding_boxes.append(bounding[keep[k]])

    bounding_boxes = np.array(bounding_boxes)

    return bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.imsave('result_face_detection.png', fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':
    im = cv2.imread('einstein.png', 0)
    hog = extract_hog(im, visualize=True)

    I_target= cv2.imread('target.png', 0) # MxN image

    I_template = cv2.imread('template.png', 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png') # MxN image (just for visualization)
    
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0]) # visualization code



