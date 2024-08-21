# Diffusion-Transformers
## Contents
* [**CelebA**](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#celeba) 
   * **[Generated-images](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#generated-images)** <====== See the Model Generated Images here
   * **[Training-insights](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#training-insights)**
* **[MNIST-experiment](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#mnist-experiment)**
   * [**Training on MNIST**](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#training-on-mnist)
* **[Diffusion-Transformers Paper Summary](https://github.com/VachanVY/diffusion-transformer?tab=readme-ov-file#latent-diffusion-models)**

---
* To clone the repo and install libraries from requirements.txt run the below commands
   ```python
   git clone https://github.com/VachanVY/diffusion-transformer.git
   pip install -r requirements.txt
   ```
---
## CelebA
* <img src="images/_____image.png" alt="CelebA Sample" width="400"/>\
   The CelebA dataset consists of celebrity images, such as the example shown above.
* *The model has been trained for only 100K steps so far. Ideally, it should be trained for 400K steps to improve the quality of generated images. While the current model is undertrained, the generated images are still decent*. Check them out below.
* ***If you have GPUs and can train this model for 400k steps, please edit the generate.py file to include a download link to your weights, and send a pull request. I’d be happy to incorporate it! You can start training from the checkpoint, download it using the below command***
  ```python
   python checkpoints/celeba/download_celeba_ckpt.py
  ```

### Generated Images
* Run the following command to generate images:
   ```python
   python checkpoints/celeba/download_celeba_ckpt.py # download weights
   python generate.py --labels female male female female --ema True # generate images
   ```
* Some generated images:\
    <img src="https://github.com/user-attachments/assets/6cbe6bc7-1e7a-44ed-83ae-df64c40bf8d4" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/181f31fe-4b4c-4719-93df-f3d767853608" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/eb463c13-45b2-474e-a431-b3b296fe00c4" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/2048e0ba-4ebd-48f1-a379-df39680495f9" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/e7bea3b5-ce52-4de6-a03e-a7694e66b320" alt="Alt text" width="300">
    <img src="https://github.com/user-attachments/assets/fe7d1f7d-f7a8-4059-a1f2-172a0e2345d5" alt="Alt text" width="300">


### Training Insights
* ![www](images/loss_vs_steps.png)
* Run the following file to get the graph of `loss` vs `learning_rate` to select the best `max_lr`:
  ```python
  python torch_src/config.py
  ```
  ![d34r3r](images/loss_vs_lr.png)
  You can see that "log10 learning rates" after `-4.0 (lr=1e-4)` are unstable. When using a max_lr=3e-4, the training loss spiked, and the model forgot everything midway through training
  ```python
  python train.py
  ```
* A GIF showing images as the training progress is displayed below
 ![Animated GIF](images/training_progress.gif)


## MNIST Experiment
* The MNIST dataset is used as a test case for the diffusion model.


### Training on MNIST
* To train the model on MNIST, run:
   ```python
   python torch_src/diffusion_transformer.py
   ```
* Additionally, check out the jax_mnist_diffusion_transformer.ipynb notebook for a JAX version of the model:
![alt text](images/______image-1.png)


## Latent-Diffusion Models
* In this paper, we apply DiTs to latent space, although they could be applied to pixel space without modification as well

## Classifier-free Guidance
![alt text](images/dit-1.png)

## DiTs
* ![alt text](images/dit.png)
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
