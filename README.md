# extract-patches

Simple function for local patch extraction from OpenCV keypoints.

Usage:
    
```python
from extract_patches import extract_patches
PATCH_SIZE = 32
mrSize = 3.0
patches = extract_patches(kps1, img1, PATCH_SIZE, mrSize)
```

See another example in this [notebook](patch-extraction-demo.ipynb)

Thanks to Vassileios(https://github.com/vbalnt) for the baseline implementation.
