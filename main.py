import cv2
import numpy as np
import os
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt
from PIL import Image
from freeman import main as freeman8
from freeman4 import main as freeman4

class MooreBoundary:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.binary_image = None
        self.boundary_image = None

    def convert_to_grayscale(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        return v

    def preprocess_image(self, grayscale_image):
        filtered_image = cv2.medianBlur(grayscale_image, ksize=5)
        return filtered_image

    def binarize_image(self, grayscale_image):
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        _, binary_image = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.binary_image = binary_image
        return binary_image

    def postprocess_image(self, binary_image):
        kernel = np.ones((5, 5), np.uint8)
        processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        processed_image[processed_image > 0] = 255
        self.binary_image = processed_image
        return processed_image

    def is_boundary_pixel(self, y, x):
        if self.binary_image[y, x] == 0:
            return False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if dy == 0 and dx == 0:
                    continue
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.binary_image[ny, nx] == 0:
                        return True
        return False

    def get_boundary_image(self):
        boundary_image = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if self.is_boundary_pixel(y, x):
                    boundary_image[y, x] = 255
        self.boundary_image = boundary_image
        return boundary_image

def save_image(image, filename):
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, filename), image)
    

def main():
    image_path = "ishihara-18.jpg"
    original_image = cv2.imread(image_path)



    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    i = 0 
    for angle, code in rotations.items():
        print(f"\nProcessing rotation of {angle} degrees:")
        if code is not None:
            rotated_image = cv2.rotate(original_image, code)
            
        else:
            rotated_image = original_image.copy()
            


        moore_boundary = MooreBoundary(rotated_image)
        grayscale_image = moore_boundary.convert_to_grayscale()
        preprocessed_image = moore_boundary.preprocess_image(grayscale_image)
        binary_image = moore_boundary.binarize_image(preprocessed_image)
        postprocessed_image = moore_boundary.postprocess_image(binary_image)
        boundary_image = moore_boundary.get_boundary_image()

        # Save each stage of the processing
        save_image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), f"rotated_{angle}.png")
        save_image(grayscale_image, f"grayscale_{angle}.png")
        save_image(preprocessed_image, f"preprocessed_{angle}.png")
        save_image(binary_image, f"binary_{angle}.png")
        save_image(postprocessed_image, f"postprocessed_{angle}.png")
        save_image(boundary_image, f"boundary_{angle}.png")
        freeman4(boundary_image, i)
        freeman8(boundary_image, i)

        # Apply mask and save segmented image
        mask_bool = (postprocessed_image == 255)
        mask_3_channels = np.dstack([mask_bool, mask_bool, mask_bool])
        background = np.zeros_like(rotated_image)
        segmented_image = np.where(mask_3_channels, rotated_image, background)
        save_image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), f"segmented_{angle}.png")
        i+=1


if __name__ == "__main__":
    main()
