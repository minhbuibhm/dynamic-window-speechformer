# Dynamic Window Speechformer (DW-Speechformer)
Based on Speechformer++ and DWFormer
<figure>
  <img
  src="./figures/1-overview.svg"
  alt="DW-Speechformer architecture.">
  <figcaption>Fig 1. DW-Speechformer architecture.</figcaption>
</figure>

## Based on Speechformer++
[\[IEEE/ACM TASLP\]](https://ieeexplore.ieee.org/abstract/document/10011559) SpeechFormer++: A Hierarchical Efficient Framework for Paralinguistic Speech Processing

## Proposed M-DWFormer block based on DWFormer
[\[ICASSP 2023\]](https://ieeexplore.ieee.org/abstract/document/10094651) DWFormer: Dynamic Window Transformer for Speech Emotion Recognition

## Compare our DW-Speechformer with Speechformer++
<figure>
  <img
  src="./figures/15_compare.png"
  alt="t-SNE visualization of DW-Speechformer and Speechformer++ features.">
  <figcaption>Fig 2. t-SNE visualization of DW-Speechformer and Speechformer++ features.</figcaption>
</figure>

## Result of proposed M-DWFormer Block

<figure>
  <img
  src="./figures/10-visualize-angry.drawio.svg"
  alt="Visualize proposed M-DWFormer Block on Sample 65.">
  <figcaption>Fig 3. Visualize proposed M-DWFormer Block on Sample 65.</figcaption>
</figure>
<figure>
  <img
  src="./figures/11-visualize-ang.drawio.svg"
  alt="Visualize proposed M-DWFormer Block on Sample 60.">
  <figcaption>Fig 4. Visualize proposed M-DWFormer Block on Sample 60.</figcaption>
</figure>

## Usage
Please follow the guidelines of Speechformer [here](https://github.com/HappyColor/SpeechFormer).

## Citation
```
@ARTICLE{chen2023,
  author={Chen, Weidong and Xing, Xiaofen and Xu, Xiangmin and Pang, Jianxin and Du, Lan},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={SpeechFormer++: A Hierarchical Efficient Framework for Paralinguistic Speech Processing}, 
  year={2023},
  volume={31},
  number={},
  pages={775-788},
  doi={10.1109/TASLP.2023.3235194}}
```

```
@ARTICLE{chen2023,
  author={S. Chen, X. Xing, W. Zhang, W. Chen and X. Xu},
  journal={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece}, 
  title={DWFormer: Dynamic Window Transformer for Speech Emotion Recognition}, 
  year={2023},
  volume={31},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10094651}}
```
