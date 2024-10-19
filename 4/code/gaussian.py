def conv(img, filter):
    if len(img.shape) == 3:
        _,_,c = img.shape
        img_list = [img[:,:,i] for i in range(c)]
    elif len(img.shape) == 2:
        img_list = [img]
    results = []
    for img in img_list:
        assert len(img.shape) == len(filter.shape)
        results.append(convolve2d(img,filter,'same'))
    if len(results) == 1:
        return results[0]
    else:
        return np.dstack(results)
    
def GaussianStack(image,ksize=40,layers=2):
    stack = [image]
    for i in range(layers):
        sigma = 2 ** i
        D = cv2.getGaussianKernel(ksize,sigma)
        gaussian_filter = D @ D.T
        blur_img = conv(image,gaussian_filter)
        stack.append(blur_img)
    return stack

def LaplacianStack(image,ksize=20,layers=5):
    gaussian_stack = GaussianStack(image,ksize,layers)
    stack = []
    for i in range(len(gaussian_stack)-1):
        pre_blur_img = gaussian_stack[i]
        cur_blur_img = gaussian_stack[i+1]
        stack.append(pre_blur_img-cur_blur_img)
    stack.append(gaussian_stack[-1])
    return stack

def blend(img1,img2,mask):
    L_img1_stack = LaplacianStack(img1)
    L_img2_stack = LaplacianStack(img2)
    g_mask_stack = GaussianStack(mask)
    assert len(L_img1_stack) == len(L_img2_stack) == len(g_mask_stack)
    collapse_imgs = []
    collapse = np.zeros_like(g_mask_stack[0]).astype(np.float64)
    for i in range(len(L_img1_stack)):
        collapse = collapse + g_mask_stack[i] * L_img1_stack[i] + (1-g_mask_stack[i])*L_img2_stack[i]
    return collapse