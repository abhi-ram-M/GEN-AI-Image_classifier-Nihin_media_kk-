\# Medical vs Non-medical Image Classifier (CLIP + Linear Probe)



\## Overview

Classifies images as \*\*medical\*\* or \*\*non\_medical\*\* using CLIP (ViT-B/32) as a frozen feature extractor with a small linear layer on top.  

Supports images from URLs and embedded images in PDFs, shown in a Streamlit app.



\## Approach

\- Data in `medical/` and `non\_medical/`, resized to 224×224 RGB.

\- Balanced classes, split into train/val/test with fixed seed.

\- Extracted embeddings from frozen CLIP, trained a single linear layer.

\- Loss: Cross-Entropy, Optimizer: Adam/AdamW, early stopping.

\- URL images via BeautifulSoup, PDFs via PyMuPDF.



\## Results

\- \*\*Validation:\*\* 100% accuracy  

\- \*\*Test:\*\* 100% accuracy  

(Small, clean dataset made separation easy.)



\## Performance

\- Only trained the linear head → fast \& low memory.

\- Batched inference (32) + `torch.inference\_mode` for speed.

\- Used CLIP’s preprocessing to avoid mismatches.



