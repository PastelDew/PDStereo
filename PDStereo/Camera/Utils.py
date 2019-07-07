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
    #disparity map이란 정합을 위한 두 이미지에서의 객체의 위치 상의 다른 차이를 말한다.
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
    left_matcher = cv2.StereoSGBM_create(       #Semi - global block 매칭 알고리즘을 사용하여 스테레오 일치를 계산하는 클래스
        minDisparity=min_disp,                 #가능한 최소 disparity 값, 일반적으로 0이지만 정류 알고리즘이 이미지를 이동 할 수 있으므로 적절하게 조절
        numDisparities=num_disp,             # max_disp has to be dividable by 16 f. E. HH 192, 256 max-min > 0 and num%16=0
        blockSize=5,                    # sadwindowszie 절대 차이의 합은 블록간의 유사성을 측정 한 값.
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,   # P1,P2 disparity의 부드러움을 제어하는 매개변수.p1은 인접픽셀들 사이에 disparity에 대한 패널티이며, +1 or -1한다.  p2 >1 p1,p2에 대한 식은 합리적으로 좋은 값이다.
        #p2값이 클 수 록 disparity가 부드러워진다.
        disp12MaxDiff=10,               #좌우 disparity검사에서 최대 허용차이 사용하지 않을때는 음수 입력
        uniquenessRatio=1,              #발견된 일치 항목을 고려하기 위해 최상의 계산된 비용 함수 값이 차선책 보다 나은 백분율로 표현된 마진 5~15값이 적당하다.
        speckleWindowSize=150,          #노이즈 얼룩 및 무효화를 고려한 부드러운 Disparity 영역의 최대 크기. 스펙클 필터링을 사용하지 않으려면 0, 아니면 150~200의 범위중 설정.
        speckleRange=2,                 #연결된 각 구성 요소 내 최대 불일치 변화. 스펙클 필터링을 수행하는 경우 매개 변수를 양수 값으로 설정하면 암시적으로 16을 곱한다. 일반적으로 1 or 2이면 충분
        preFilterCap=4,                 #사전 필터링 된 이미지 픽셀의 truncation value이다. 알고리즘은 각 픽셀에서 x-미분값을 계산하고 -prefiltercap,prefiltercap간격으로 값을 클리핑 한다. 결과값은 Birchfield-Tomasi픽셀 비용 함수에 전달.
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