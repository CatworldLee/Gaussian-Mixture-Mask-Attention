![image](images/main_picture_v4.png)

# RetNet Viewed through Convolution

**Paper:** [Read the Full Paper (PDF)](https://arxiv.org/pdf/2309.05375.pdf)

**Authors:** Chenghao Li, Chaoning Zhang



## Environment

The environment configuration file has already been provided, and you can use the following script to create the required environment.

```bash
pip install -r requirements.txt
```


## Intro

The success of Vision Transformer (ViT) in image recognition tasks has been well-documented. While ViT can capture global dependencies better than Convolutional Neural Networks (CNN), CNN's local characteristics are still valuable due to their resource efficiency. Recently, [RetNet](https://arxiv.org/abs/2307.08621) has demonstrated remarkable performance in language modeling, outperforming Transformers with explicit local modeling. This has led researchers to explore Transformers in the Computer Vision (CV) field.

[This paper](https://arxiv.org/pdf/2309.05375.pdf) investigates the applicability of RetNet from a CNN perspective and introduces a customized RetNet variant for visual tasks. Similar to RetNet, we enhance ViT's local modeling by introducing a weight mask to the self-attention matrix. Our initial results with a learnable element-wise weight mask (ELM) show promise. However, ELM introduces additional parameters and optimization complexity.

To address this, our work proposes a novel Gaussian mixture mask (GMM) with only two learnable parameters, making it suitable for various ViT variants with adaptable attention mechanisms. Experiments on multiple small datasets illustrate the effectiveness of our Gaussian mask in enhancing ViTs at minimal additional cost in terms of parameters and computation.

## Approach

**Overview of Gaussian Mixture Mask (GMM) Attention Mechanism**

1. **Input Feature Vectors:** Initially, $N \times D$ feature vectors are processed in an attention module, resulting in three matrices, namely $Q$, $K$, and $V$.

2. **Gaussian Masks:** Multiple Gaussian masks with varying parameters, specifically $\sigma$ and $\alpha$, are defined. These masks are then linearly combined to create a Gaussian mixture mask, which is of size $2N - 1$.

3. **Mask Application:** This Gaussian mixture mask is applied to the original self-attention mechanism. It undergoes a shifting-window extension and unfolds to generate corresponding attention scores for each patch.

4. **Attention Map:** The unfolded Gaussian mixture mask contributes to the creation of an attention map for each patch.

5. **Output Computation:** The final output patch feature is computed as the dot product of the matrix $V$ and the attention map.

In summary, the GMM Attention Mechanism enhances the self-attention mechanism by using a mixture of Gaussian masks with different parameters to generate attention maps for patches, thereby improving the modeling of local relationships in the input data.


![image](images/motivation_v1.png)

By observing the experimental results, we found that this simple learnable mask has the following regularities: patches inhibit the flow of information to themselves, and the correlation coefficient between patches varies with distance. Based on these geometric features, we propose a Gaussian Mixture Mask to fit the distribution and show how it fits the simple and learnable mask.

## Outcomes

| Model      | CIFAR-10  | CIFAR-100 | SVHN     | Tiny-ImageNet | Parameters | MACs     | Depth |
|------------|-----------|-----------|----------|---------------|------------|----------|-------|
| ViT        | 93.65%    | 75.36%    | 97.93%   | 59.89%        | 2.7M       | 170.9M   | 9     |
| GMM-ViT    | **95.06%**| **77.81%**| **98.01%** | **62.27%**   | 2.7M       | 170.9M   | 9     |
| Swin       | 95.26%    | 77.88%    | 97.89%   | 60.45%        | 7.1M       | 236.9M   | 12    |
| GMM-Swin   | **95.39%**| **78.26%**| **97.90%** | **61.03%**   | 7.1M       | 236.9M   | 12    |
| CaiT       | 94.79%    | 78.42%    | 98.13%   | 62.46%        | 5.1M       | 305.9M   | 26    |
| GMM-CaiT   | **95.15%**| **78.97%**| 98.09%   | **63.64%**    | 5.1M       | 305.9M   | 26    |
| PiT        | 93.68%    | 72.82%    | 97.78%   | 57.63%        | 7.0M       | 239.1M   | 12    |
| GMM-PiT    | **94.41%**| **74.16%**| **97.82%** | **58.37%**   | 7.0M       | 239.1M   | 12    |
| T2T        | 95.32%    | 78.10%    | 97.99%   | 61.50%        | 6.5M       | 417.4M   | 13    |
| GMM-T2T    | **96.16%**| **79.91%**| 97.98%   | **63.33%**    | 6.5M       | 417.4M   | 13    |

## Run scripts

### Original model

```bash
python main.py --model [model_name]
```

### GMM variants

```bash
python main.py --model [model_name] --is_GMM
```

## Acknowledgements

We would like to express our great appreciation to the authors of the SPT_LSA_ViT and pytorch-image-models repositories, aanna0701 and rwightman, for their great help to the machine learning community. Their provided models and related technical supports have helped us to explore this field in a more comprehensive and in-depth way, thus enhancing our learning efficiency.

