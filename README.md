# Diffusion-Transformers
![alt text](images/dit.png)

## Latent-Diffusion Models
* In this paper, we apply DiTs to latent space, although they could be applied to pixel space without modification as well

## Classifier-free Guidance
![alt text](images/dit-1.png)

## DiTs
* Following patchify, we apply standard ViT frequency-based positional embeddings (the sine-cosine version) to all input tokens
![alt text](images/dit-2.png)

### DiT block design
We explore four variants of transformer blocks that process conditional inputs differently
* ***In-context conditioning:*** We simply append the vector
embeddings of t and c as two additional tokens in
the input sequence, treating them no differently from
the image tokens. This is similar to cls tokens in
ViTs, and it allows us to use standard ViT blocks without
modification.
* ***Cross-attention block:*** We concatenate the embeddings
of t and c into a length-two sequence, separate from
the image token sequence. The transformer block is
modified to include an additional multi-head crossattention
layer following the multi-head self-attention
block, similar to the original design from Vaswani et
al., and also similar to the one used by LDM for
conditioning on class labels. Cross-attention adds the
most Gflops to the model, roughly a 15% overhead.
* ***Adaptive layer norm (adaLN) block:*** We explore replacing standard layer norm layers in transformer blocks with adaptive layer norm (adaLN). Rather than directly learn dimensionwise scale and shift parameters $\gamma$ and $\beta$, we regress
them from the sum of the embedding vectors of $t$ and
$c$. Of the three block designs we explore, adaLN adds
the least Gflops and is thus the most compute-efficient
* ***adaLN-Zero block*** : Zero-initializing the final batch norm scale factor in each block accelerates large-scale training in the supervised learning setting. Diffusion U-Net models use a similar initialization strategy, zero-initializing the final convolutional layer in each block prior to any residual connections. We explore a modification of the adaLN DiT block which does the same. In addition to regressing $\gamma$ and $\beta$, we also regress dimensionwise scaling parameters $\alpha$ that are applied immediately prior to any residual connections within the DiT block

## Training Setup
* We initialize the final linear layer with zeros and otherwise use standard weight initialization techniques from ViT
* AdamW
* We use a constant learning rate of $1\times 10^{-4}$, no weight decay and a batch size of $256$
* Exponential moving average (EMA) of DiT weights over training with a decay of $0.9999$
* The VAE encoder has a downsample factor of 8, given an
RGB image $x$ with shape $256 \times 256 \times 3$, $z = E(x)$ has shape $32 \times 32 \times 4$
* $t_{\max} = 1000$ linear variance schedule ranging from $1 \times 10^{-4}$ to $2 \times 10^{-2}$