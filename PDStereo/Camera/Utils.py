import numpy as np
import cv2
import os

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def changeAllToNumpy(lst, dtype=np.float32):
    if not isinstance(lst, list):
        return lst
    return np.array([changeAllToNumpy(item) for item in lst], dtype=dtype)

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def get_disparity_map(leftFrame, rightFrame,
    window_size=5, lmbda=80000, sigma=1.2, visual_multiplier=1.0,
    min_disp=16, num_disp=96): #num_disp = 112-min_disp
    """left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )"""
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=10,
        uniquenessRatio=1,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=4,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    """
    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    disp = stereo.compute(leftFrame, rightFrame).astype(np.float32) / 16.0
    """
    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(leftFrame, rightFrame)
    dispr = right_matcher.compute(rightFrame, leftFrame)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, leftFrame, None, dispr)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    #visible_filteredImg = filteredImg
    #visible_filteredImg = cv2.normalize(src=filteredImg, dst=visible_filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    #visible_filteredImg = np.uint8(visible_filteredImg)
    #cv2.imshow('Disparity Map', visible_filteredImg)
    return filteredImg #, visible_filteredImg
    #cv2.imshow('Disparity Map', np.uint8(cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)))
    
def savePointClound(filename, leftFrame, disparity, Q, imgSize):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points_3D[mask]
    out_colors = colors[mask]

    out_fn = filename + '.ply'
    write_ply(out_fn, out_points, out_colors)

def saveRGBD(dir, filename, leftFrame, disparity):
    if not os.path.exists(dir):
        os.mkdir(dir)
    rgbd = np.dstack((leftFrame, disparity))
    cv2.imwrite('{}/color-{}.png'.format(dir, filename), leftFrame)
    cv2.imwrite('{}/depth-{}.png'.format(dir, filename), disparity)
    cv2.imwrite('{}/merged-{}.png'.format(dir, filename), rgbd)
    return rgbd