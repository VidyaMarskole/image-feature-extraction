import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure, filters, segmentation, color
from scipy import ndimage
import tkinter as tk

def visualize_image_features(image_path):
    # Get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Calculate optimal figure size (80% of screen size)
    fig_width = screen_width * 0.8 / 100
    fig_height = screen_height * 0.8 / 100
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create figure with screen-adjusted size
    plt.figure(figsize=(fig_width, fig_height))
    
    # Dynamically adjust spacing based on screen size
    title_fontsize = max(8, min(12, screen_height // 80))
    pad_size = max(8, min(15, screen_height // 80))
    wspace = 0.3 + (0.5 * (screen_width / 1920))  # Scale with screen width
    hspace = 0.4 + (0.5 * (screen_height / 1080)) # Scale with screen height
    
    plt.subplots_adjust(wspace=wspace, hspace=hspace, 
                      left=0.05, right=0.95, 
                      top=0.95, bottom=0.05)
    
    # Feature visualization functions
    def plot_feature(pos, image, title, cmap=None):
        plt.subplot(4, 5, pos)
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(f'{pos}. {title}', pad=pad_size, fontsize=title_fontsize)
        plt.axis('off')
    
    # Row 1
    plot_feature(1, img_rgb, 'Original Image')
    
    plt.subplot(4, 5, 2)
    for i, color in enumerate(('r', 'g', 'b')):
        plt.plot(cv2.calcHist([img], [i], None, [256], [0, 256]), color=color)
        plt.xlim([0, 256])
    plt.title('2. Color Histogram', pad=pad_size, fontsize=title_fontsize)
    
    plot_feature(3, cv2.cvtColor(img, cv2.COLOR_BGR2HSV), 'HSV Color Space')
    plot_feature(4, cv2.cvtColor(img, cv2.COLOR_BGR2LAB), 'LAB Color Space')
    plot_feature(5, cv2.Canny(img_gray, 100, 200), 'Edges (Canny)', 'gray')
    
    # Row 2
    corner_img = img_rgb.copy()
    corners = cv2.cornerHarris(np.float32(img_gray), 2, 3, 0.04)
    corner_img[cv2.dilate(corners, None) > 0.01 * corners.max()] = [255, 0, 0]
    plot_feature(6, corner_img, 'Corners (Harris)')
    
    plot_feature(7, local_binary_pattern(img_gray, 24, 3, method='uniform'), 
               'Texture (LBP)', 'gray')
    
    glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256, 
                       symmetric=True, normed=True)
    plot_feature(8, graycoprops(glcm, 'contrast'), 'GLCM Contrast', 'gray')
    
    _, hog_image = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=True)
    plot_feature(9, exposure.rescale_intensity(hog_image, in_range=(0, 10)), 
                'Shape (HOG)', 'gray')
    
    gabor = cv2.filter2D(img_gray, cv2.CV_8UC3, 
                        cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 0.5, 0, cv2.CV_32F))
    plot_feature(10, gabor, 'Gabor Filter', 'gray')
    
    # Row 3
    sobel = np.sqrt(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)**2 + 
                   cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)**2)
    plot_feature(11, sobel, 'Sobel Edge', 'gray')
    
    plot_feature(12, cv2.Laplacian(img_gray, cv2.CV_64F), 'Laplacian', 'gray')
    
    # Watershed
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    sure_bg = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    markers = cv2.connectedComponents(sure_fg.astype(np.uint8))[1] + 1
    markers[cv2.subtract(sure_bg, sure_fg.astype(np.uint8)) == 255] = 0
    markers = cv2.watershed(img_rgb, markers)
    img_rgb[markers == -1] = [255, 0, 0]
    plot_feature(13, img_rgb, 'Watershed Seg')
    
    # Face detection
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
              .detectMultiScale(img_gray, 1.1, 4)
    face_img = img_rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    plot_feature(14, face_img, 'Face Detection')
    
    # Blob detection
    keypoints = cv2.SimpleBlobDetector_create().detect(img_gray)
    blob_img = cv2.drawKeypoints(img_rgb, keypoints, np.array([]), (0,0,255),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plot_feature(15, blob_img, 'Blob Detection')
    
    # Row 4
    plot_feature(16, cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)[1], 
                'Simple Threshold', 'gray')
    
    plot_feature(17, cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8)), 
                'Morph Gradient', 'gray')
    
    plot_feature(18, filters.rank.entropy(img_gray, np.ones((9,9))), 'Entropy Filter', 'gray')
    
    plot_feature(19, segmentation.mark_boundaries(img_rgb, 
                                               segmentation.slic(img_rgb, n_segments=100, compactness=10)), 
                'Superpixels (SLIC)')
    
    # Optical flow
    translated = ndimage.shift(img_gray, (10, 10))
    flow = cv2.calcOpticalFlowFarneback(img_gray, translated, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    plot_feature(20, cv2.cartToPolar(flow[..., 0], flow[..., 1])[0], 'Optical Flow', 'gray')
    
    plt.tight_layout()
    plt.show()

# Example usage
visualize_image_features(r'D:\vidyaPY\Image-feature-detection\image copy.png')