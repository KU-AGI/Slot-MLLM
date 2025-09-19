"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import numpy as np
from functools import partial
from einops import rearrange
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
from typing import Tuple

from .blip2 import Blip2Base, disabled_train
from .vit import Block
from .utils import download_cached_file, is_url

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True,
                 discarding_threshold=0.01):
                #  discarding_threshold=0.01):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        # self.codebooks_used = torch.zeros(self.n_e, dtype=torch.int32).to(device)
        self.codebooks_used = torch.zeros(self.n_e, dtype=torch.int32)

        # self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12


    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    # def l2norm(self, t):
    #     return F.normalize(t, p = 2, dim = -1)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        bz = z.shape[0]
        z_flattened = z.view(-1, self.e_dim)
        #print('z_flattened', z_flattened.shape)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z)**2) + torch.mean((z_q - z.detach())**2)
        else:
            loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = z_q.reshape(bz, -1, z_q.shape[-1])

        # claculating the perplexity (average usage of codebook entries)
        encodings = torch.zeros(z_flattened.shape[0], self.n_e, device=z_flattened.device)
        encodings.scatter_(1, min_encoding_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))

        with torch.no_grad():
            self.codebooks_used = self.codebooks_used.to(z_flattened.device)
            self.codebooks_used[min_encoding_indices] += 1

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, min_encoding_indices, perplexity

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def replace_unused_codebooks(self, num_batches):

        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        For more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
         replaced codebooks might increase. However, the main trend must be decreasing after some training time.
         If it is not the case for you, increase the "num_batches" or decrease the "discarding_threshold" to make
         the trend for number of replacements decreasing. Stop calling the function at the latest stages of training
         in order not to introduce new codebook entries which would not have the right time to be tuned and optimized
         until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """

        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            device = self.embedding.weight.device
            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.embedding.weight += self.eps * torch.randn(self.embedding.weight.size(), device=device).clone()
            else:
                used = self.embedding.weight[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.embedding.weight[unused_indices] *= 0
                self.embedding.weight[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.e_dim), device=device).clone()

            print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0

        return unused_count

class NSVQ(VectorQuantizer2):
    # def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True, device=torch.device('cpu'), discarding_threshold=0.01, initialization='normal'):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True, discarding_threshold=0.01, initialization='normal'):
        super().__init__(
            n_e=n_e, e_dim=e_dim, beta=beta, remap=remap, unknown_index=unknown_index, sane_index_shape=sane_index_shape, legacy=legacy,
        )
        """
                Inputs:

                1. num_embeddings = Number of codebook entries

                2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)

                3. device = The device which executes the code (CPU or GPU)

                ########## change the following inputs based on your application ##########

                4. discarding_threshold = Percentage threshold for discarding unused codebooks

                5. initialization = Initial distribution for codebooks

                """
        # self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12

        if initialization == 'normal':
            # codebooks = torch.randn(self.n_e, self.e_dim, device=device)
            print("NSVQ normal initialization")
            codebooks = torch.randn(self.n_e, self.e_dim)
        elif initialization == 'uniform':
            print("NSVQ uniform initialization")
            codebooks = uniform_dist.Uniform(-1 / self.n_e, 1 / self.n_e).sample([self.n_e, self.e_dim])
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        # self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True).to(device)
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

        # Counter variable which contains the number of times each codebook is used
        # self.codebooks_used = torch.zeros(self.n_e, dtype=torch.int32, device=device)
        self.codebooks_used = torch.zeros(self.n_e, dtype=torch.int32)

    def forward(self, input_data):

        """
        This function performs the main proposed vector quantization function using NSVQ trick to pass the gradients.
        Use this forward function for training phase.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for training | shape: (NxD) )
                perplexity (average usage of codebook entries)
        """
        bz = input_data.shape[0]
        input_data = input_data.view(-1, self.e_dim)
        device = input_data.device
        # compute the distances between input and codebooks vectors
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, self.codebooks.t()))
                     + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)).to(device)

        # min_indices = torch.argmin(distances, dim=1)
        min_indices = torch.argmin(distances, dim=1).to(device)

        hard_quantized_input = self.codebooks[min_indices]
        random_vector = normal_dist.Normal(0, 1).sample(input_data.shape).to(device)

        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_quantization_residual / norm_random_vector + self.eps) * random_vector

        quantized_input = input_data + vq_error

        # reshape back to match original input shape
        quantized_input = quantized_input.reshape(bz, -1, quantized_input.shape[-1])

        # claculating the perplexity (average usage of codebook entries)
        encodings = torch.zeros(input_data.shape[0], self.n_e, device=input_data.device)
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            self.codebooks_used = self.codebooks_used.to(device)
            self.codebooks_used[min_indices] += 1

        # use the first returned tensor "quantized_input" for training phase (Notice that you do not have to use the
        # tensor "quantized_input" for inference (evaluation) phase)
        # Also notice you do not need to add a new loss term (for VQ) to your global loss function to optimize codebooks.
        # Just return the tensor of "quantized_input" as vector quantized version of the input data.

        # codebook loss: just for logging
        input_data = input_data.reshape(bz, -1, input_data.shape[-1])
        codebook_loss = self.beta * torch.mean((quantized_input.detach() - input_data) ** 2) + torch.mean((quantized_input - input_data.detach()) ** 2)

        return quantized_input, codebook_loss, min_indices, perplexity,

    def replace_unused_codebooks(self, num_batches):

        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        For more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
         replaced codebooks might increase. However, the main trend must be decreasing after some training time.
         If it is not the case for you, increase the "num_batches" or decrease the "discarding_threshold" to make
         the trend for number of replacements decreasing. Stop calling the function at the latest stages of training
         in order not to introduce new codebook entries which would not have the right time to be tuned and optimized
         until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """

        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            device = self.codebooks.device
            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.e_dim), device=device).clone()

            print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0

        return unused_count

    def get_codebook_entry(self, indices, shape=None):
        codebooks = self.codebooks.detach().clone()
        ###########################################
        quantized_input = codebooks[indices]

        #use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input

class EMAVectorQuantizer(nn.Module):
    """
    EMAVectorQuantizer
    """
    def __init__(self,
                 dim: int,
                 n_embed: int,
                 beta: float,
                 decay: float = 0.99,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(n_embed, dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embedding_avg", embedding.clone())

    def forward(self,
                z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()  # [B,C,H,W] -> [B,H,W,C]
        z_flattened = z.view(-1, self.dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.embedding).view(z.shape)
        embed_onehot = F.one_hot(min_encoding_indices, self.n_embed).type(z_flattened.dtype)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = embed_onehot.transpose(0, 1) @ z_flattened

            dist.all_reduce(embed_onehot_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)

            self.embedding.data.copy_(embed_normalized)

        diff = self.beta * torch.mean((z_q.detach() - z) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        perplexity = torch.exp(-torch.sum(embed_onehot * torch.log(embed_onehot + 1e-12), dim=1)).mean()

        return z_q, diff, min_encoding_indices, perplexity

    def get_codebook_entry(self, indices):
        z_q = F.embedding(indices, self.embedding)
        return z_q

class Blip2QformerQuantizer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(self,
                 vit_model="eva_clip_g",
                 img_size=224,
                 drop_path_rate=0,
                 use_grad_checkpoint=False,
                 vit_precision="fp16",
                 freeze_vit=True,
                 num_query_token=32,
                 cross_attention_freq=2,
                 embed_dim=256,
                 max_txt_len=32,
                 codebook_embed_dim=32,
                 n_embed=8192,
                 num_quantizers=4,
                 vq_type="vq2",
                 discarding_thre=0.01,
                 recon_s=True,
                 blocks_for_image=True,
                 decode_depth=4,
                 use_recon_s_for_image=False,
                 use_qformer_image=False,
                 image_features_dim=1024,
                 device="cuda",
                 is_train=True,
                 legacy=False,
                 ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model, img_size, drop_path_rate, use_grad_checkpoint,
                                                                       vit_precision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            # self.ln_vision.weight.requires_grad = False
            # self.ln_vision.bias.requires_grad = False

        # 32
        self.codebook_embed_dim = codebook_embed_dim
        # 8192
        self.n_embed = n_embed
        # True
        self.recon_s = recon_s
        # True
        self.blocks_for_image = blocks_for_image
        # False
        self.use_recon_s_for_image = use_recon_s_for_image
        # 4
        self.depth = decode_depth
        # 1024
        self.image_features_dim = image_features_dim
        # False
        self.use_qformer_image = use_qformer_image

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, self.visual_encoder.num_features)

        # self.Qformer.cls = None

        # self.Qformer.bert.config.vocab_size = 30523
        # self.Qformer.set_output_embeddings(nn.Linear(self.Qformer.config.hidden_size, 30523))
        # # word_embedding [30522, 768] to [30523, 768]
        # self.Qformer.bert.embeddings.word_embeddings = nn.Embedding(30523, self.Qformer.config.hidden_size)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        # For ITC
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        for name, param in self.Qformer.named_parameters():
            param.requires_grad = is_train
        self.query_tokens.requires_grad = is_train

        if vq_type == "vq2":
            self.quantize = VectorQuantizer2(n_embed, codebook_embed_dim, beta=0.25, remap=None, sane_index_shape=False, legacy=legacy,
                                            #  device=device,
                                             discarding_threshold=discarding_thre,
                                            )
        elif vq_type == "ema_vq":
            self.quantize = EMAVectorQuantizer(codebook_embed_dim, n_embed, beta=0.25)
        elif vq_type == "nsvq":
            self.quantize = NSVQ(n_embed, codebook_embed_dim, beta=0.25, remap=None, sane_index_shape=False,
                                #  legacy=False, device=device,
                                 legacy=legacy,
            )
        elif vq_type == "residual_vq":
            from vector_quantize_pytorch import ResidualVQ
            self.quantize = ResidualVQ(
                dim=self.Qformer.config.hidden_size,
                num_quantizers=num_quantizers,
                codebook_size=n_embed,
                codebook_dim=codebook_embed_dim,
                shared_codebook=True
            )

        else:
            self.quantize = None
        # 768 => 32
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.Qformer.config.hidden_size, codebook_embed_dim)  # for quantize
        )

        # 32 => 768
        self.decode_task_layer = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, self.Qformer.config.hidden_size)  # for quantize
        )

        # if not is_train:
        #     self.quantize = self.quantize.eval()
        #
        # self.quantize.training = is_train
        # for name, param in self.named_parameters():
        #     if 'quantize' in name or 'encode_task_layer' in name or 'decode_task_layer' in name:
        #         #print('freeze params', name)
        #         param.requires_grad = is_train

        if self.recon_s:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
            self.blocks = nn.ModuleList([
                Block(dim=self.Qformer.config.hidden_size,
                      num_heads=12,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
            ])

        if self.blocks_for_image:
            self.pos_embed_image = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
            self.blocks_image = nn.ModuleList([
                Block(dim=self.Qformer.config.hidden_size,
                      num_heads=12,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
            ])

        if self.use_qformer_image:
            num_reverse_token = 1
            self.Reverse_Qformer, self.reverse_tokens = self.init_Qformer(num_reverse_token, self.Qformer.config.hidden_size)

            self.Reverse_Qformer.cls = None
            self.Reverse_Qformer.bert.embeddings.word_embeddings = None
            self.Reverse_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Reverse_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.distill_image_proj = nn.Linear(self.Qformer.config.hidden_size, image_features_dim)

        else:
            self.image_down = nn.Sequential(
                nn.Linear(self.Qformer.config.hidden_size, 256, bias=False),
                nn.ReLU(),
                nn.Linear(256, 128, bias=False),
                nn.ReLU(),
                nn.Linear(128, 32, bias=False),
            )
            self.distill_image_proj = nn.Linear(num_query_token * 32, image_features_dim)
 
    def get_causal_embeddings(self, image, use_slot=False, slot_config=None, use_causal=False):
        # Yes grad for training
        # with torch.no_grad():
        with self.maybe_autocast():
        #with torch.no_grad():
            # [b, 257, 1408]
            image_embeds = self.ln_vision(self.visual_encoder(image))
            # [b, 256]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # [b, 32, 768]
        # Original query_tokens shape is [1, 32, 768]
        # Match to batch size
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_output : [b, 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_slot=use_slot,
            slot_config=slot_config,
            return_dict=True,
            use_causal=use_causal,
        )
        return query_output.last_hidden_state

    def get_codebook_indices(self, image):
        causal_embeddings = self.get_causal_embeddings(image)
        # Notice: query_output_down is match to clip embedding?
        # CLIP ViT-H/14 text embeeding is [b, 1024]. Then it matches to [b, 32, 32]?
        # [b, 32, 768] => [b, 32, 32]
        query_output_down = self.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind, _ = self.quantize(query_output_down)
        # 
        embed_ind = embed_ind.reshape(quant.shape[0], -1)
        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.decode_task_layer(quant)
        # TODO: query_output_up should be trained to be similar with original causal embeddings (query_output)
        # [b, 32], 
        return embed_ind, query_output_up

    def get_codebook_entry(self, indices):
        # pdb.set_trace()
        # indicie => [b, 32], quant_embedding => [b, 32, 32]
        quant_embedding = self.quantize.get_codebook_entry(indices)
        # print('quant_embedding_shape: ', quant_embedding.shape)
        # print(self.decode_task_layer)
        # exit()
        # [b, 32, 768]
        query_output_up = self.decode_task_layer(quant_embedding)

        query_output_up = self.get_transformer_decoded_embedding(query_output_up)

        # pdb.set_trace()

        if self.use_qformer_image:
            query_atts = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(query_output_up.device)
            reverse_tokens = self.reverse_tokens.expand(query_output_up.shape[0], -1, -1)
            reverse_output = self.Reverse_Qformer.bert(
                query_embeds=reverse_tokens,
                encoder_hidden_states=query_output_up,
                encoder_attention_mask=query_atts,
                return_dict=True,
            )
            reverse_output = reverse_output.last_hidden_state
            reverse_output_proj = self.distill_image_proj(reverse_output).squeeze(1)
        # Default set to false
        else:
            reverse_output_proj = self.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj
    
    def get_mlp_decoded_embedding(self, query_output_up):
        # 2 layer mlp to 768 -> 32, [b, 32, 32]
        reverse_output = self.image_down(query_output_up)
        # [b, 32, 32] => [b, 32 * 32]
        reverse_output = reverse_output.reshape(reverse_output.shape[0], -1)
        # [b, 1024] => [b, 1024]
        reverse_output_proj = self.distill_image_proj(reverse_output)
        return reverse_output_proj

    def get_transformer_decoded_embedding(self, query_output_up):
        query_output_up_pos_image = self.add_positional_embedding(query_output_up)
        # Transformers block
        for blk in self.blocks_image:
            query_output_up_pos_image = blk(query_output_up_pos_image)
        # Still [b, 32, 768]
        return query_output_up_pos_image

    def add_positional_embedding(self, query_output_up):
        pos_embed_image = self.pos_embed_image.repeat(query_output_up.shape[0], 1, 1)
        query_output_up_pos_image = query_output_up + pos_embed_image
        return query_output_up_pos_image

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        vit_model = kwargs.get("vit_model", "eva_clip_g")
        img_size = kwargs.get("image_size", 224)
        num_query_token = kwargs.get("num_query_token", 32)
        cross_attention_freq = kwargs.get("cross_attention_freq", 2)

        drop_path_rate = kwargs.get("drop_path_rate", 0)
        use_grad_checkpoint = kwargs.get("use_grad_checkpoint", False)
        vit_precision = kwargs.get("vit_precision", "fp16")
        freeze_vit = kwargs.get("freeze_vit", True)

        max_txt_len = kwargs.get("max_txt_len", 32)
        vq_type = kwargs.get("vq_type", "vq2")
        discarding_thre = kwargs.get("discarding_threshold", 0.01)

        num_quantizers = kwargs.get("num_quantizers", 4)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            vq_type=vq_type,
            discarding_thre=discarding_thre,
            num_quantizers=num_quantizers,
        )

        if pretrained_model_path.startswith('http'):
            print('start download seed model...')
            cached_file = download_cached_file(pretrained_model_path, check_hash=False, progress=True)
            print(cached_file)
            ckpt = torch.load(cached_file, map_location="cpu")
        else:
            ckpt = torch.load(pretrained_model_path, map_location="cpu")

        # if ckpt has 'model', it means it is a checkpoint from blip2
        if 'model' in ckpt:
            ckpt = ckpt['model']
        missing, unexcepted = model.load_state_dict(ckpt, strict=False)
        print('missing keys: ', len(missing), 'unexpected keys:', len(unexcepted))
        print('missing keys: ', missing)
        print('unexpected keys:', unexcepted)
        return model

    @classmethod
    def from_pretrained_debug(cls, pretrained_model_path, **kwargs):
        vit_model = kwargs.get("vit_model", "eva_clip_g")
        img_size = kwargs.get("image_size", 224)
        num_query_token = kwargs.get("num_query_token", 32)
        cross_attention_freq = kwargs.get("cross_attention_freq", 2)

        drop_path_rate = kwargs.get("drop_path_rate", 0)
        use_grad_checkpoint = kwargs.get("use_grad_checkpoint", False)
        vit_precision = kwargs.get("vit_precision", "fp16")
        freeze_vit = kwargs.get("freeze_vit", True)

        max_txt_len = kwargs.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )

        return model
    