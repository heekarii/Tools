## Image Tools
### 1. `PSNR-SSIM.py`
   Calculate *Peak Signal-to-Noise Ratio*(**PSNR**) and *Structural Similarity Index Measure*(**SSIM**) between Grount Truths and target Images with same size<br>
   **dependecy**
    opencv
    PIL
    scikit<br>
   **How to run**
   ``` bash
    python PSNR-SSIM.py # No additional arguments
  ```

### 2. `resizeCalc.py`
  Calculate PSNR and SSIM between Ground Truths and target Images with diffrent size.
  To calculate PSNR and SSIM, this code make target Images\` size to Ground Truths\` size using bicubic. It can makes the value lower.
  #### **dependency**<br>
  same with `PSNR-SSIM.py`<br>
  #### **How to run**
  ```bash
python reszieCalc.py # No additional arguments
```

### 3. `downscale.py`
   Downscale images using bicubic
   #### **dependency**<br>
   opencv<br>
   #### **How to run**
   ```bash
   python resize.py --image [folder path that you want to downscale] --scale [scale that you want ex) 2, 1.5 ...]
   ```

### 4. `niqe.py`
   Calculate Natural Image Quality Evaluator(NIQE)<br>
   Edit folder path in python file. You can calculate NIQE score of multiple images in that path and save results with csv file.
   #### **dependency**<br>
   opencv<br>
   numpy<br>
   pytorch<br>
   basicsr<br>
   pandas
   #### **How to run**
   ```bash
   python niqe.py
   ```

### 5. `lpips.py`
   Calculate Learned Perceptual Image Patch Similarity(LPIPS)<br>
   ####  **dependency**
   pytorch<br>
   Pillow Image
   #### **How to run**
   ```bash
   python lpips.py --target [folder path of images] --gt [folder path of GT images] --output [result path of csv file]
   ```

### 6.`bicubic.py`
   Upscale Images with bicubic interpolation<br>
   Default scale factor is 4. If you want to change scale factor, change 27th line of python file. (`upscaled = cv2.resize(img, ...`)
   #### **dependency**
   opencv
   #### **How to run**
   ```bash
   python bicubic.py --input [folder path of input images] --output [output folder path where you want to save upsampled images]
   ```

### 7. `crop.py`
   Crop rectcangle patch of images<br>
   If you want to get patch of images with same resolution, use this file.
   You can select patch size and position by changing 6th line of python file.<br>
   Input image path and Output path can be editable in code(9th and 10th line)
   #### **dependency**
   opencv
   #### **How to run**
   ```bash
   python crop.py
   ```

### 8. `fid.py & result_judge.py`
   These files are made to calculate FID score, but they aren't perfect.
