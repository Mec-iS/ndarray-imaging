"""
Collection of code chunks from
https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality

"Diatom Autofocusing in Brightfield Microscopy: A Comparative Study".

In this paper the author Pech-Pacheco et al. has provided variance of
 the Laplacian Filter which can be used to measure if the image blurryness score.
"""

def get_blurrness_score(image):
    path =  images_path + image 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm
features['blurrness'] = features['image'].apply(get_blurrness_score)
features[['image','blurrness']].head(5)