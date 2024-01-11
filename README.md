# RSICR-Net
Recently, U-shaped neural networks have gained widespread application in remote sensing image dehazing
and achieved promising performance. However, most of the existing U-shaped dehazing networks neglect the global and local information interaction across layers during the encoding phase, which leads
to incomplete utilization of the extracted features for image restoration. Moreover, in the process of
image reconstruction, utilizing only the information from the terminal layers of the decoding phase for
haze-free image restoration leads to a dilution of semantic information, resulting in color and texture
deviations in the dehazed image. To address these issues, We propose a Hierarchical Slice Information
Interaction and Multi-layer Decoding Collaborative Reconstruction dehazing network(HSMR-Net).
Specifically, a Hierarchical Slice Information Interaction Module(HSIIM) is proposed to introduce
Intra-layer feature autocorrelation and Inter-layer feature cross-correlation to facilitate global and
local information interaction across layers, thereby enhancing the encoding features representation
capability and improving the network’s dehazing performance. Furthermore, a Multi-layer Decoding
Collaborative Reconstruction module(MDCR) is proposed to fully utilize feature information in each
decoding layer, mitigate semantic information dilution, and improve the network’s capability to restore
image colors and textures. The experimental results demonstrate that our HSMR-Net outperforms
several state-of-the-art methods in dehazing on two publicly available datasets.
