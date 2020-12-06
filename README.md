Code for the report:

# Machine Learning in Medical Image Processing:
# Third Lumbar Slice Detection for Rapid Visceral Fat Calculation

Obesity has become a chronic medical condition internationally, deserving of increased medical attention and research. One recent development within obesity research is interest to improve efficiency and accuracy of the calculation of the Visceral Fat Index (VFI) obesity measure. VFI calculates the overall composition of the body by measuring the adipose tissue that lies between organs in the torso and is currently manually calculated by physicians through a visual analysis of CT scans. This manual process involves locating the slice image at the third lumbar vertebra (L3) level in the spine before computing the VFI in that location of the body. This is a tedious and time-consuming process, as the CT scan does not automatically identify the location of the L3 within the CT scan images. To address this issue, we have identified and used a previous implementation of a Fully Convolutional Neural Network model and optimized it to obtain a precision score of 100%. Our algorithm is also able to correctly process full-body CT scans, unlike the previous implementation, which was limited to only abdominal or chest CT scans. This tool significantly reduces the amount of time required by a physician to process a CT scan for VFI calculation purposes or any other study where L3 slice position is relevant.


