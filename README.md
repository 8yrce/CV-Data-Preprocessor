# CV-Data-Preprocessor
Run your dataset through the 'CV-Data-Preprocessor' to auto-magically fix issues in contrast, brightness, color space, gamma and more! 

`python3 image_preprocessor.py --image_path ./sample_images --contrast --gamme --display_results`

Preprocess images to bring them to the optimal condition for use with computer vision operations.

<h3>This program can apply:</h3>

* Contrast / brightness histogram equalization ( CLAHE )
* Color space histogram matching
* Gamma correction

<h3>Detailed param overview</h3>

* `--image_path` - Path (str) to the input dataset directory
* `--output_path` - Path to save modified images to ( no output path = overwrite original )
* `--contrast` - Correct the contrast of the image
* `--color_match_path` - Path (str) to a reference image to normalize dataset colors to ( multi-domain normalization )
* `--gamma` - Correct gamma ( if deemed applicable during exposure analysis )
* `--display_results` - Display a detailed histogram analysis and image view before and after modification

<h3>Dependencies</h3>

* skimage ( tested at 0.19.2 )
* cv2 ( tested at 4.6.0 )
* numpy ( tested at 1.21.5 )
