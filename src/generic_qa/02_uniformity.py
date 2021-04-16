"""
Collection of code chunks from
https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality

*Average Pixel Width* is a measure which indicates the amount of edges present in the image.
If this number comes out to be very low, then the image is most likely a uniform image
and may not represent right content.
"""

def average_pixel_width(img):
    path = images_path + img 
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100