## Abstract

​	Visible watermark is a common means of protecting digital image copyright. Analyzing the removal of visible watermarks can help verify the effectiveness of the watermark on the image, and provide references and inspiration for watermark designers. Currently, most watermark removal methods are based on natural images, while document images are widely used in daily life and have a different style from natural images, but there is a lack of research on watermark removal in document images. In this paper, we found that using existing watermark removal methods can easily leave watermark main body or contour artifacts in the removal results of document image visible watermarks. To address this issue, a two-stage document image visible watermark removal model based on global and local features was proposed. First, a Single Document Image Watermark Removal Dataset (SDIWRD) containing watermarked document images was constructed to study watermark removal for document images. Then, a two-stage visible watermark removal network was designed based on the encoder-decoder structure with half instance normalization. In the coarse stage encoder, a global and local feature extraction module was designed to enhance the ability to capture spatial feature information while retaining the ability to extract local detail information. In the refinement stage, the network structure shared coarse stage weights and a recurrent feature fusion module was constructed to fully explore the effective features of the coarse stage encoder, providing rich contextual information for the refinement stage to help remove watermarks accurately. In addition to the network structure, a structural similarity perception loss was combined to obtain better visual quality. The proposed method achieved a peak signal-to-noise ratio of *--* dB, structural similarity of -- and root-mean-square error of -- after removing watermarks on the SDIWRD dataset, which was superior to existing advanced watermark removal methods and could effectively alleviate watermark pseudo artifacts. Finally, some suggestions were provided to prevent watermark removal. After the paper is accepted, the source code will be made available.

![](readme.assets/%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%842.png)

## 1. Datasets link

you can download at here (After the paper is accepted, the datasets will be made available.)

<img src="readme.assets/SDIWRD%E6%95%B0%E6%8D%AE%E9%9B%86%E5%B1%95%E7%A4%BA_en.png" style="zoom: 25%;" />

## 2. Compare with state of the art methods

![](readme.assets/%E5%90%84%E6%96%B9%E6%B3%95%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94%E5%9B%BE_en.png)

