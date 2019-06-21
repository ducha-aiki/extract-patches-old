import numpy as np
import math

def extract_patches(kpts, img, PS=32, mag_factor = 1.0):
    import cv2
    """
    Builds image pyramid and rectifies patches around openCV keypoints
    from the appropriate level of image pyramid, 
    removing high freq artifacts. Border mode is set to "replicate", 
    so the boundary patches don`t have crazy black borders
    Returns list of patches.
    Upgraded version of
    https://github.com/vbalnt/tfeat/blob/master/tfeat_utils.py
    """
    if len(img.shape) == 2:
        ch = 1
    else:
        h,w,ch = img.shape
    patches = []
    img_pyr = [img]
    cur_img = img

    while np.min(cur_img.shape[:2]) > PS:
        cur_img = cv2.pyrDown(cur_img)
        img_pyr.append(cur_img)
        
    for i, kp in enumerate(kpts):
        x,y = kp.pt
        s = kp.size
        a = kp.angle
        s = mag_factor * s / PS
        pyr_idx = int(math.log(s,2)) 
        d_factor = float(math.pow(2.,pyr_idx))
        s_pyr = s / d_factor
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix([
            [+s_pyr * cos, -s_pyr * sin, (-s_pyr * cos + s_pyr * sin) * PS / 2.0 + x/d_factor],
            [+s_pyr * sin, +s_pyr * cos, (-s_pyr * sin - s_pyr * cos) * PS / 2.0 + y/d_factor]])
        patch = cv2.warpAffine(img_pyr[pyr_idx], M, (PS, PS),
                             flags=cv2.WARP_INVERSE_MAP + \
                             cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_REPLICATE)
        patches.append(patch)
    return patches

def extract_patches_pil(kpts, img, PS=32, mag_factor = 1.0):
    from PIL import Image
    """
    Builds image pyramid and rectifies patches around openCV keypoints
    Returns list of patches.
    Upgraded version of
    https://github.com/vbalnt/tfeat/blob/master/tfeat_utils.py
    Uses PIL instead of OpenCV and has BLACK BORDERS for boundary patches
    """
    if len(img.shape) == 2:
        ch = 1
    else:
        h,w,ch = img.shape
    patches = []
    img_pyr = [Image.fromarray(img)]
    w1,h1 = img_pyr[-1].size    
    while min(w1,h1) > PS:
        cur_img = img_pyr[-1].copy()
        cur_img.thumbnail((w1//2,h1//2),Image.LANCZOS)
        img_pyr.append(cur_img)
        w1,h1 = img_pyr[-1].size
    for i, kp in enumerate(kpts):
        x,y = kp.pt
        s = kp.size
        a = kp.angle
        s = mag_factor * s / PS
        pyr_idx = int(math.log(s,2)) 
        d_factor = float(math.pow(2.,pyr_idx))
        s_pyr = s / d_factor
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.array([
            [+s_pyr * cos, -s_pyr * sin, (-s_pyr * cos + s_pyr * sin) * PS / 2.0 + x/d_factor],
            [+s_pyr * sin, +s_pyr * cos, (-s_pyr * sin - s_pyr * cos) * PS / 2.0 + y/d_factor],])
        patch = img_pyr[pyr_idx].transform(
            (PS,PS), Image.AFFINE,
            M.ravel(),
            Image.BILINEAR
        )
        patches.append(np.array(patch))
    return patches

def patch_extract_vbalnt(kpts, img,  PS = 32, mag_factor = 1.0):
    import cv2
    """
    Rectifies patches around openCV keypoints, and returns patches list
    https://github.com/vbalnt/tfeat/blob/master/tfeat_utils.py
    """
    patches = []
    for i, kp in enumerate(kpts):
        x,y = kp.pt
        s = kp.size
        a = kp.angle
        s = mag_factor * s / PS
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)
        M = np.matrix([
            [+s * cos, -s * sin, (-s * cos + s * sin) * PS / 2.0 + x],
            [+s * sin, +s * cos, (-s * sin - s * cos) * PS / 2.0 + y]])

        patch = cv2.warpAffine(img, M, (PS, PS),
                             flags=cv2.WARP_INVERSE_MAP + \
                             cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
        patches.append(patch)
    return patches