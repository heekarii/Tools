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
