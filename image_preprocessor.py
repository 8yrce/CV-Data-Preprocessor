"""
demo usage:
python3 image_preprocessor.py --image_path ./sample_images --contrast --gamma --display_results

image_preprocessor.py
    Preprocesses images to bring them to the optimal condition for use with computer vision operations
    Can apply:
        - Contrast / Brightness improvements
        - Color space matching ( for multi-domain dataset normalization )
        - Gamma correction

Bryce Harrington
09/02/22
"""
import os
import cv2
import traceback
import numpy as np
from skimage import exposure
from dataclasses import dataclass
from argparse import ArgumentParser

parser = ArgumentParser(description="image-preprocessor, pass args to enable operations")
parser.add_argument("--image_path", metavar='i', help="Path to the images to modify", type=str, required=True)
parser.add_argument("--output_path", metavar='o', help="Output path for modified images (overwrites original if none)")
parser.add_argument("--display_results", action="store_true", help="Display results of each modified image")
parser.add_argument("--contrast", help="Correct the contrast of the image", action="store_true")
parser.add_argument("--color_match_path", metavar='m', help="Image path to match color space of dataset to", type=str)
parser.add_argument("--gamma", help="Correct gamma in image if underexposed", action="store_true")
args = parser.parse_args()


@dataclass
class ImagePreprocessor:
    image: np.array
    match_image: np.array
    original_image: np.array

    def __init__(self):
        # read in match image ( if applicable )
        self.match_image = cv2.imread(args.color_match_path) if args.color_match_path is not None else None

        # grab images
        for image_name in os.listdir(args.image_path):
            self.image = cv2.imread(os.path.join(args.image_path, image_name))
            self.original_image = self.image

            # run through all the functions that were enabled
            if args.contrast:
                self.contrast()

            if args.color_match_path:
                self.color_match(self.match_image)

            if args.gamma:
                self.gamma()

            if args.display_results:
                self.display_results()

    def contrast(self):
        """
        Adjust the contrast of the image by equalizing th histogram
        :param image: input to modify contrast of
        :sets: self.image: the image post modification
        """
        try:
            # convert and split to HSV for intensity mod
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            (H, S, V) = cv2.split(image)

            clahe = cv2.createCLAHE(tileGridSize=(8, 8), clipLimit=3)
            clahe_v = clahe.apply(V)

            image = cv2.merge([H, S, clahe_v])
            self.image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        except Exception as e:
            print("[ERROR] Unable to perform contrast operation: {}:{}".format(e, traceback.format_exc()))

    def color_match(self, match_image):
        """
        Apply color matching through histogram matching to known good image
        :param image: input to remap colors of
        :param match_image: image to remap colors to
        :sets: self.image: the image post modification
        """
        try:
            # apply histogram matching
            self.image = exposure.match_histograms(self.image, match_image, multichannel=(self.image.shape[-1] > 1))

        except Exception as e:
            print("[ERROR] Unable to apply color matching: {}:{}".format(e, traceback.format_exc()))

    def gamma(self):
        """
        Correct gamma in image if underexposed
        :param image: input to correct gamma on ( if necessary )
        :sets: self.image: the image post modification
        """
        try:
            # check if the exposure is low before modifying gamma
            gamma = 1.0
            gamma_mod = 0.2
            while exposure.is_low_contrast(self.image):
                gamma += gamma_mod
                self.image = exposure.adjust_gamma(self.image, gamma, gain=1)

        except Exception as e:
            print("[ERROR] Unable to perform gamma correction: {}:{}".format(e, traceback.format_exc()))

    def display_results(self):
        """
        Display some information regarding the modifications done on the image
        """
        try:
            cv2.imshow("Original", self.original_image)
            cv2.imshow("Modified", self.image)
            cv2.waitKey(0)
        except Exception as e:
            print("[ERROR] Unable to display results: {}:{}".format(e, traceback.format_exc()))


if __name__ == "__main__":
    ImagePreprocessor()