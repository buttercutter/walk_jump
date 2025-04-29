# Credit : gpt-4o , Claude-3.5-Sonnet-200k , Gemini-Pro-1.5

# Reference :
# [Protein Discovery with Discrete Walk-Jump Sampling](http://arxiv.org/abs/2306.12360)
# [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](http://arxiv.org/abs/2407.01392)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy
import os
import random
import string

from collections import Counter
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.parametrizations import weight_norm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from adam_mini import Adam_mini


# the models have been trained / finetuned, run inference code only
INFERENCE_ONLY = 0

# Just for code development / debugging purpose
TEST_OVERFIT = 0

# for the denoiser module, choose only ONE of the following options :
USE_PRETRAINED_BERT = 0
USE_PRETRAINED_BERT_MLM = 0
USE_PRETRAINED_T5 = 0
USE_CUSTOM_TRANSFORMER_ENCODER = 0  # the most RAM memory efficient option
USE_CUSTOM_TRANSFORMER_ENCODER_DECODER = 1

# Early-stopping for the models training
USE_EARLY_STOP = 0
EARLY_STOP_THRESHOLD = 2.175  #1.91

# for sentence completion downstream task
ENABLE_MASK_LEARNING = 1

# google colab T4 GPU does not have a lot of RAM for computation
# custom transformer module can now handle multiple masked tokens
if torch.cuda.is_available():  #or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
    MASK_RATIO = 0.15   # use 0.15 for 15% masking probability, use the value of -1 to indicate only a single masked token
else:
    MASK_RATIO = 0.15   # use 0.15 for 15% masking probability, use the value of -1 to indicate only a single masked token

# allows the denoiser model to train on [batch_size, sequence_length, vocab_size]
USE_LOGITS_FOR_THE_ENTIRE_SENTENCE = 1
USE_LOGITS_FOR_THE_ENTIRE_SENTENCE = USE_LOGITS_FOR_THE_ENTIRE_SENTENCE or (MASK_RATIO != -1)  # if masking more than 1 token, then it makes sense to train on [batch_size, sequence_length, vocab_size]
# custom transformer module can now handle multiple masked tokens
#USE_LOGITS_FOR_THE_ENTIRE_SENTENCE = USE_LOGITS_FOR_THE_ENTIRE_SENTENCE and not (USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER)

# Analyze walk-jump's output samples for debugging purpose
ENABLE_SAMPLE_ANALYSIS = 0  # turns off for reducing memory consumption

if torch.cuda.is_available():
    device_str = "cuda"
else:
    device_str = "mps"

device = torch.device(device_str)

# Automatic Mixed Precision for training
from torch import autocast
if torch.cuda.is_available():
    from torch.amp import GradScaler
    USE_MIXED_PRECISION_TRAINING = 0  # optional, turns off for this code since it hurts model performance
else:
    USE_MIXED_PRECISION_TRAINING = 0  # not implemented


# for saving RAM memory during training : https://github.com/zyushun/Adam-mini
USE_ADAM_MINI = 0

# 0: Sinusoidal Positional Embedding , 1: Rotary Positional Embedding
USE_ROPE = 0

# Just for code development / debugging purpose
USE_DUMMY_TRAINING_DATA = 0

# for adjusting the generation process due to fixed output length
GENERATES_OUTPUT_OF_VARYING_LENGTH = 0

# for more difficult denoising task
ADD_EXTRA_GAUSSIAN_NOISE = 0  # turns off for now

# Select between diffusion forcing and walk-jump
# if the following two variables are turned off, it would be walk-jump (single constant noise level)
USE_DIFFUSION_FORCING = 1 & ADD_EXTRA_GAUSSIAN_NOISE
USE_PRECOMPUTE_NOISE_SCHEDULE = 0  # testing only, do not recommend to use due to expensive storage

# Regarding two different approaches for Langevin MCMC sampling
USE_MCMC = 1
USE_ALGORITHM_1_OR_4 = 0  # value of 1 means Algorithm 1, value of 0 means Algorithm 4, see walk-jump paper
USE_OBABO = 0  # Using KIPLMC2 is slow because of the need to compute gradients of U with respect to both theta and X

# sequential monte-carlo (SMC)
USE_SMC = 0  # if use SMC, then ignore USE_ALGORITHM_1_OR_4 which is related to Langevin MCMC

# Markov-approximate fractional Brownian motion (MA-fBM)
USE_MAFBM = 0  # if use MAFBM, then ignore USE_ALGORITHM_1_OR_4 which is related to Langevin MCMC

# Once turned on, it will be different from the walk-jump denoise update equation
USE_LOGITS_FOR_DENOISING = 0  # consumes much more RAM memory
USE_LOGITS_FOR_DENOISING = USE_LOGITS_FOR_DENOISING and (USE_SMC or USE_MAFBM or USE_MCMC)

# kl_div method (requires extra run of denoiser model) to improve sampling based on prior distribution
# Only turn on USE_GRAD_KL if USE_PRETRAINED_T5 is disabled, because USE_GRAD_KL uses
# "tokenizer.vocab_size"-rounds of denoiser module execution, hence extremely long execution time.
# Using large pretrained T5 model as denoiser module will only worsen the runtime issue.
USE_GRAD_KL = 0

# Choose only one of the following training receipes for walk-jump sampling
USE_dWJS_ENERGY = 1
USE_dWJS_SCORE = ~USE_dWJS_ENERGY

# Define parameters
input_dim = 128
model_dim = input_dim
model_dim_ebm = model_dim >> 2  # specific only to EBM model
hidden_dim = 256
num_layers = 4
num_layers_ebm = num_layers >> 1  # specific only to EBM model
num_heads = 8
num_heads_ebm = num_heads >> 2  # specific only to EBM model
num_smc_steps = 5  # sequential monte-carlo (SMC)
N_particles = 10  # sequential monte-carlo (SMC)
hurst = 0.7  # Markov-approximate fractional Brownian motion (MA-fBM)
T_fbm = 1.0  # Markov-approximate fractional Brownian motion (MA-fBM)
n_steps = 1000  # Markov-approximate fractional Brownian motion (MA-fBM)
K_fbm = 3  # Markov-approximate fractional Brownian motion (MA-fBM)
num_walk_steps = 5  # for langevin dynamics MCMC sampling process
num_jump_steps = 20  #num_walk_steps
walk_step_size = 0.6  # for langevin dynamics MCMC sampling process
sigma_max = 1.1
sigma_min = 0.1
num_epochs = 500
batch_size = 512


if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
    # BERT model is larger than TransformerDenoiser() module
    batch_size = batch_size >> 6

elif USE_PRETRAINED_T5:
    # T5 models are way larger than both BERT model and TransformerDenoiser() module
    batch_size = 1

elif USE_CUSTOM_TRANSFORMER_ENCODER_DECODER:
    # we have extra decoder layers inside the TransformerDenoiser() module
    batch_size = batch_size >> 4

else:  # USE_CUSTOM_TRANSFORMER_ENCODER
    # we do not have extra decoder layers inside the TransformerDenoiser() module
    batch_size = batch_size >> 3


#if torch.cuda.is_available():  # so far colab run session has some extra unused GPU RAM on T4 GPU
#    batch_size = batch_size << 2  # increasing batch_size worsens the validation loss convergence rate


# Monitors the quality of the generated samples throughout the training and validation
# processes to assess the model's performance and identify potential issues
def analyze_samples(generated_samples, tokenizer, skip_special_tokens=False, num_samples=1):
    decoded_samples = []
    if num_samples != 1:
        num_samples = generated_samples.size(0)

    for i in range(num_samples):
        sample = generated_samples[i]
        sample = sample.long()  # Convert the sample to integer tensor
        decoded_sample = tokenizer.decode(sample, skip_special_tokens=skip_special_tokens)
        print(f"Sample {i+1}: {decoded_sample}")
        decoded_samples.append(decoded_sample)

    return decoded_samples

def assert_sample_range_compliance(sample, tokenizer):
    # Assert that all token IDs are within the valid range
    assert sample.min() >= 0, f"Token ID is less than 0!  sample = {sample}, sample.min() = {sample.min()}"
    assert sample.max() < tokenizer.vocab_size, f"Token ID exceeds valid range! Max ID: {sample.max()}, Vocab Size: {tokenizer.vocab_size}"

    # Assert that the tokens input to the model are not all zeros
    assert not torch.all(sample == 0), "Error: sample contains all zeros!"
    return True

def check_for_vanishing_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            if grad_norm < 1e-5:  # Threshold for detecting vanishing gradients
                print(f"Warning: Vanishing gradient detected in {name} with norm {grad_norm.item():.6f}")

if USE_PRETRAINED_T5: #or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
    tokenizer = AutoTokenizer.from_pretrained("pnawrot/nanoT5-base")
    #tokenizer = T5Tokenizer.from_pretrained('google/t5-efficient-tiny')
else:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenizer_function(raw_sequence_input, tokenizer, max_length=input_dim):
    tokenized_sequence = tokenizer(
                            raw_sequence_input,
                            padding='max_length',
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                         )

    return tokenized_sequence.to(device)


#print(f"tokenizer.pad_token_id = {tokenizer.pad_token_id}")
# for initializing target_label for denoiser module
CONSTANTS_VALUE_IGNORE = tokenizer.pad_token_id  # -100

# for creating data loader for span-masking task
class DataCollatorForSpanCorruption:
    def __init__(self, tokenizer, mlm_probability=0.15, mean_noise_span_length=3, input_length=input_dim):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length

    def __call__(self, examples):
        # If examples are tensors, convert them to lists
        if isinstance(examples[0], torch.Tensor):
            input_ids = [example.tolist() for example in examples]
            attention_mask = None  # No attention mask for tensor inputs
        else:
            # Assuming examples are dicts with 'input_ids' keys
            input_ids = [example['input_ids'] for example in examples]
            attention_mask = [example['attention_mask'] for example in examples] if 'attention_mask' in examples[0] else None

        batch = self._collate_batch(input_ids)

        # Add attention mask if it exists
        if attention_mask is not None:
            batch['attention_mask'] = pad_sequence(
                [mask.clone().detach() for mask in attention_mask],
                batch_first=True,
                padding_value=0
            )

        return batch

    def _collate_batch(self, input_ids_list):
        # Pad input_ids to the same length
        batch_input_ids = pad_sequence(
            [ids.clone().detach() for ids in input_ids_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        # Create masked inputs and labels
        if USE_PRETRAINED_T5:
            masked_input_ids, labels, mlm_mask = self._mask_tokens_span(batch_input_ids)
            return {'input_ids': masked_input_ids, 'labels': labels, 'mask_indices': mlm_mask}

        elif USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
            #labels, mlm_mask = self._mask_tokens_span(batch_input_ids)
            labels, mlm_mask = self._mask_tokens_standard(batch_input_ids)
            return {'input_ids': batch_input_ids, 'labels': labels, 'mask_indices': mlm_mask}

        else:  # USE_CUSTOM_TRANSFORMER_ENCODER or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER
            #labels, mlm_mask = self._mask_tokens_span(batch_input_ids)
            labels, mlm_mask = self._mask_tokens_standard(batch_input_ids)
            return {'input_ids': batch_input_ids, 'labels': labels, 'mask_indices': mlm_mask}

    # span-masking strategy
    def _mask_tokens_span(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked span language modeling according to T5's objective.
        """
        inputs = inputs.clone()
        labels = torch.full(inputs.shape, self.tokenizer.pad_token_id)
        special_tokens = {self.tokenizer.pad_token_id}

        batch_size, seq_len = inputs.shape
        mask_indices = []

        # Track masking locations
        mask_indices_tensor = torch.zeros_like(inputs, dtype=torch.bool)

        for i in range(batch_size):
            input_ids = inputs[i].tolist()
            num_to_mask = max(1, int(round(seq_len * self.mlm_probability)))

            # Get candidate indices to mask
            candidate_indices = [
                idx for idx in range(len(input_ids)) if input_ids[idx] not in special_tokens
            ]

            # Shuffle candidate indices
            random.shuffle(candidate_indices)

            masked_indices = set()
            current_idx = 0
            spans = []
            while len(masked_indices) < num_to_mask and current_idx < len(candidate_indices):
                span_length = max(1, int(numpy.random.poisson(lam=self.mean_noise_span_length)))
                start = candidate_indices[current_idx]
                end = min(start + span_length, seq_len)
                span_indices = list(range(start, end))

                # Avoid overlapping spans
                if any(idx in masked_indices for idx in span_indices):
                    current_idx += 1
                    continue

                masked_indices.update(span_indices)
                spans.append((start, end))
                current_idx += 1

            # Sort spans in reverse order to avoid index shifting issues
            spans = sorted(spans, key=lambda x: x[0], reverse=True)

            target_tokens = []
            prev_end = seq_len
            for idx, (start, end) in enumerate(spans):
                # Replace span with sentinel token in inputs
                if USE_PRETRAINED_T5:
                    sentinel_token_id = self.tokenizer.convert_tokens_to_ids(f'<extra_id_{idx}>')
                else:
                    sentinel_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                inputs[i, start:end] = sentinel_token_id
                # Build labels
                target_tokens = [sentinel_token_id] + input_ids[start:end] + target_tokens

            # Record the masked positions
            for start, end in spans:
                mask_indices_tensor[i, start:end] = True

            # Handle unmasked positions in labels
            #labels[~mask_indices_tensor] = CONSTANTS_VALUE_IGNORE
            #labels[i, :len(target_tokens)] = torch.tensor(target_tokens, dtype=torch.long)

            # debug prints
            if len(spans) > 0:
                total_masked = sum(end - start for start, end in spans)
                #print(f"Sequence {i}: Created {len(spans)} spans, masking {total_masked} tokens")
                #print(f"Spans: {spans}")

        if USE_PRETRAINED_T5:
            return inputs, labels, mask_indices_tensor  # T5 masking tokens are not unique, so need to return masked "inputs"
        else:
            return labels, mask_indices_tensor  # Return the mask information

    # standard BERT masking strategy without any span-masking
    def _mask_tokens_standard(self, inputs):
        """
        Prepare masked tokens inputs/labels for standard masked language modeling (e.g., BERT).
        """
        labels = inputs.clone()

        # Create a mask for tokens to mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for masked tokens, set CONSTANTS_VALUE_IGNORE for others
        #labels[~masked_indices] = CONSTANTS_VALUE_IGNORE  # We only compute loss on masked tokens

        # Replace masked input tokens according to BERT's strategy
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, device=device, dtype=torch.long)
        indices_random = indices_random.to(device)
        inputs[indices_random] = random_words[indices_random]

        # The rest 10% of the time, keep the original token (do nothing)

        return labels, masked_indices


sigma = 0.5  # single noise level
mask_token_penalty_weight = 1.0  # Increase this value to penalize more heavily
sep_token_penalty_weight = 1.0  # Increase this value to penalize more heavily
unused_token_penalty_weight = 0.005  # Increase this value to penalize more heavily
ebm_energy_regularization_scale = 16  # for L2 regularization on EBM loss
if USE_CUSTOM_TRANSFORMER_ENCODER_DECODER:
    ebm_energy_regularization_scale = ebm_energy_regularization_scale << 1  # for L2 regularization on EBM loss


'''
log(Σ exp(x_i)) = log(Σ exp(x_i - C + C))
               = log(Σ exp(x_i - C) * exp(C))
               = log(exp(C) * Σ exp(x_i - C))
               = log(exp(C)) + log(Σ exp(x_i - C))
               = C + log(Σ exp(x_i - C))
where C is any constant.

The log_sum_exp() implementation chooses C to be max_val (the maximum value among the x_i values). Here's why this is brilliant:

1. Shifting by max_val: By subtracting max_val from each x_i before exponentiating, we ensure that:
   - The largest value among x_i - max_val will be 0 (because max_val - max_val = 0).
   - All other values of x_i - max_val will be negative or 0.
2. Avoiding Overflow: Since exp(0) = 1, and exp(x) for negative x is always between 0 and 1, we avoid computing exp() of large positive numbers, thus preventing overflow.
3. Reducing Underflow: While underflow might still occur for extremely small values of exp(x_i - max_val), it's less severe because we are summing these values. The sum is less likely to underflow to zero compared to individual terms.
4. Adding Back max_val: Finally, we add max_val back to the result to compensate for the subtraction we did earlier. This ensures that we get the correct value of log_sum_exp(x_i).
'''
def log_sum_exp(x):
    max_val = x.max()
    return max_val + torch.log(torch.sum(torch.exp(x - max_val)))

# USE_ROPE = 0
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(f"pe.shape = {self.pe.shape}")
        return x + self.pe[:x.size(0), :]


# USE_ROPE = 1
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        sinusoid = torch.einsum('i,j->ij', positions, self.inv_freq)
        sin = sinusoid.sin()
        cos = sinusoid.cos()
        return cos, sin


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, is_causal=False, batch_first=False):
        super().__init__()
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        self.nhead = nhead
        self.d_model = d_model

        self.is_causal = is_causal
        self.batch_first = batch_first

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def apply_rotary_emb(self, x, cos, sin):
        """
        Apply rotary embeddings to the input tensor using the provided cosine and sine values.

        Args:
            x (torch.Tensor): Input tensor.
            cos (torch.Tensor): Precomputed cosine values.
            sin (torch.Tensor): Precomputed sine values.

        Returns:
            torch.Tensor: Tensor with rotary embeddings applied.
        """
        assert x.ndim == 4  # Ensure input is for multi-head attention
        #print(f"x.ndim = {x.ndim}")
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if self.batch_first:
            batch_size, tgt_len, embed_dim = query.shape
        else:
            tgt_len, batch_size, embed_dim = query.shape

        src_len = key.shape[1]
        scaling = float(self.head_dim) ** -0.5

        q = self.q_proj(query).view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(max(src_len, tgt_len))
        q = self.apply_rotary_emb(q, cos, sin)
        k = self.apply_rotary_emb(k, cos, sin)

        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * scaling

        # Apply causal mask for decoder self-attention
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=q.device), diagonal=1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            attn += attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        output = torch.matmul(attn, v)
        #print(f"After attention, output.shape = {output.shape}, tgt_len = {tgt_len}")

        # This is for USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_DECODER
        # Before the final reshape, handle the case where tgt_len < embed_dim
        if tgt_len < embed_dim:
            # Method 1: Pad with zeros to reach embed_dim
            padding = torch.zeros(batch_size, self.nhead, embed_dim - tgt_len, self.head_dim, device=output.device)
            #print(f"padding.shape = {padding.shape}")
            output = torch.cat([output, padding], dim=2)
            tgt_len = embed_dim

            # OR Method 2: Repeat the output to reach embed_dim
            # output = output.repeat_interleave(math.ceil(embed_dim / tgt_len), dim=1)[:, :embed_dim, :]

        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        #print(f"After reshape view, output.shape = {output.shape}")
        output = self.out_proj(output)

        return output

# Encoder Layer: Uses single self-attention (bidirectional)
class RoPETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        # Single self-attention layer (non-causal/bidirectional)
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout, is_causal=False, batch_first=batch_first)

        # One set of normalization and feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Two layer norms (pre-attention and pre-FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Two dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src

        # Single self-attention block
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout1(attn_output)

        # Single feedforward block
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        x = x + self.dropout2(ff_output)

        return x

# Decoder Layer: Uses both self-attention (causal) and cross-attention
class RoPETransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        # Causal self-attention for decoder
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout, is_causal=True, batch_first=batch_first)

        # Cross-attention to connect with encoder outputs
        self.multihead_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout, is_causal=False, batch_first=batch_first)

        # Same feedforward as encoder
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Three layer norms (pre-self-attn, pre-cross-attn, pre-FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Three dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                memory_is_causal=True, tgt_is_causal=True):
        x = tgt

        # Self-attention block (causal)
        attn_output = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )

        #print(f"In RoPETransformerDecoderLayer(), x.shape = {x.shape}, attn_output = {attn_output.shape}")
        if USE_CUSTOM_TRANSFORMER_ENCODER_DECODER:
            x = x.mean(dim=1).unsqueeze(1)
        x = x + self.dropout1(attn_output)

        # Cross-attention block
        cross_attn_output = self.multihead_attn(
            self.norm2(x), self.norm2(memory), self.norm2(memory),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        x = x + self.dropout2(cross_attn_output)

        # Feedforward block
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(self.norm3(x)))))
        x = x + self.dropout3(ff_output)

        return x


# USE_PRETRAINED_BERT = 1
class BertDenoiser(nn.Module):
    def __init__(self, model_dim, use_bert_mlm=USE_PRETRAINED_BERT_MLM):  # model_dim == sequence_length
        super(BertDenoiser, self).__init__()
        self.use_bert_mlm = use_bert_mlm
        self.final_layer = nn.Linear(tokenizer.vocab_size, 1)

        # SiLU layer
        self.SiLU = nn.SiLU()

        # ReLU layer
        self.ReLU = nn.ReLU()

        if self.use_bert_mlm:
            self.model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny").to(device)
            #self.model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)

            self.config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
            #self.config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
        else:
            self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(device)
            #self.model = AutoModel.from_pretrained("bert-base-uncased").to(device)
            self.middle_layer = nn.Linear(self.model.config.hidden_size, tokenizer.vocab_size)

        self.dropout = nn.Dropout(0.2)  # Add dropout

        # Apply Xavier/Glorot or He initialization
        #self._initialize_weights()

        # Initialize the final layer
        #nn.init.xavier_uniform_(self.final_layer_A.weight)
        #nn.init.xavier_uniform_(self.final_layer_B.weight)
        #nn.init.zeros_(self.final_layer_A.bias)
        #nn.init.zeros_(self.final_layer_B.bias)

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(param, torch.nn.Parameter):
                    if param.dim() > 1:  # Only apply to matrices, not biases
                        if 'self_attn' in name or 'multihead_attn' in name:
                            torch.nn.init.xavier_uniform_(param)  # Xavier for attention layers
                        else:
                            torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')  # He for ReLU-based layers
            elif 'bias' in name:
                torch.nn.init.zeros_(param)  # Biases are usually initialized to zero

    def forward(self, inputs, mlm_mask=None):
        if isinstance(inputs, dict):
            # Convert input_ids to long tensor
            input_ids = inputs['input_ids'].long()
            labels = inputs['labels'].to(device)
            attention_mask = inputs['attention_mask']
        else:
            input_ids = inputs.long()
            labels = input_ids.clone().detach()
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

        #print(f"input_ids.shape = {input_ids.shape}")

        if self.use_bert_mlm:
            # Process text
            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                              ).logits
        else:
            # Process text
            outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                              ).last_hidden_state  # [batch_size, sequence_length, hidden_size=128]

            outputs = self.middle_layer(outputs)

        # Shape: [batch_size, seq_len, vocab_size]
        #print(f"outputs.shape = {outputs.shape}")

        outputs = self.dropout(outputs)  # Apply dropout

        denoised_sentence = self.final_layer(outputs).squeeze(-1)  # shape : [batch_size, sequence_length]

        # Apply activation function
        denoised_sentence = self.SiLU(denoised_sentence)

        if mlm_mask is not None:
            masked_positions = mlm_mask.bool()
            denoised_masked_token_logits = outputs[masked_positions]  # shape : [batch_size, vocab_size]  if MASK_RATIO = -1
            denoised_token_logits = outputs

            #print(f"denoised_sentence.shape = {denoised_sentence.shape}, denoised_masked_token_logits.shape = {denoised_masked_token_logits.shape}, denoised_token_logits.shape = {denoised_token_logits.shape}")

            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                return denoised_sentence, denoised_masked_token_logits, denoised_token_logits
            else:
                return denoised_sentence, denoised_masked_token_logits
        else:
            return denoised_sentence


'''
Original Sentence: "The quick brown fox jumps over the lazy dog."

Masked Encoder Input (input_ids): ['The', 'quick', '‹extra_id_0>', 'jumps', 'over', 'the', '<extra_id_1>', 'dog', '.']
Decoder's Target Output (labels): ['<extra_id_0>', 'brown', 'fox', '<extra_id_1>', 'lazy']

Decoder's Generation Process:
-----------------------------------------------------------------------------------
Time Step   | Decoder Input Token   | Target Label Token    | Prediction Objective
-----------------------------------------------------------------------------------
t=0         | ‹pad>                 | ‹extra_id_0>          | Predict < extra_id_0>
t=1         | ‹extra_id_0>          | brown                 | Predict brown
t=2         | brown                 | fox                   | Predict fox
t=3         | fox                   | ‹extra_id_ 1>         | Predict < extra_id_1›
t=4         | ‹extra_id_ 1>         | lazy                  | Predict lazy
-----------------------------------------------------------------------------------

Note: There are no timesteps corresponding to 'jumps', 'over', 'the' in the decoder's output because these tokens are unmasked and present in the encoder input.
'''

# USE_PRETRAINED_T5 = 1
class T5Denoiser(nn.Module):
    def __init__(self, model_dim):
        super(T5Denoiser, self).__init__()
        #self.model = T5ForConditionalGeneration.from_pretrained('google/t5-efficient-tiny')
        self.model = AutoModelForSeq2SeqLM.from_pretrained("pnawrot/nanoT5-base")

        # Projection layer to map logits space back to sequence_length (which is same as model_sim)
        self.projection_A = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, model_dim),
            #nn.ReLU()  # no need of activation function before being fed into cross-entropy loss function
        )

        # Projection layer to map logits space back to a single token embedding
        self.projection_B = nn.Sequential(
            nn.Linear(self.model.config.vocab_size, 1),
            #nn.ReLU()  # no need of activation function before being fed into cross-entropy loss function
        )

    def forward(self, input_ids, target_label=None, decoder_input_ids=None, mlm_mask=None):
        if decoder_input_ids is not None:
            # Shift tgt to the right to create decoder input ids
            decoder_input_ids = self.model._shift_right(decoder_input_ids)
        else:
            batch_size = input_ids.size(0)

            # Use the decoder start token and expand it to match the batch size
            # If tgt is not provided, use the BOS token as the initial input for decoder
            decoder_start_token = torch.tensor([[self.model.config.decoder_start_token_id]], device=device)
            decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=device)
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_start_token.expand(batch_size, -1)), dim=1)

        #print(f"decoder_input_ids.shape = {decoder_input_ids.shape}")

        # Generate output logits
        # We do not need to manually feed in decoder_input_ids, we let the model handles them internally during training
        output = self.model(input_ids=input_ids.long(), labels=target_label)
        #output = self.model(input_ids=input_ids.long(), decoder_input_ids=decoder_input_ids.long())
        output = output.logits  # shape : [batch_size, tgt_sequence_length, vocab_size]
        #print(f"output.shape = {output.shape}")

        if ENABLE_MASK_LEARNING:  # there is a new token concatenated to tgt tensor
            # Get the most recent timestep prediction
            # We want to update denoised_sentence based on the prediction for the last token in the sequence
            denoised_sentence = output[:, -1, :]  # Select the last timestep
        else:
            # Remove unnecessary dimension
            denoised_sentence = output.squeeze(1)

        #print(f"denoised_sentence.shape = {denoised_sentence.shape}")
        # denoised_sentence has a shape of [batch_size, vocab_size]

        # projection_A layer uses almost same amount of RAM as projection_B layer (which relies on broadcast operation)
        # We should not use torch.max() because introduces non-differentiable points, hindering gradient-based optimization.
        # Besides, only the maximum value receives a gradient; all other inputs get zero gradients, which is inefficient for learning.
        if mlm_mask is not None:
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                denoised_token_logits = output
                denoised_masked_token_logits = denoised_sentence

                denoised_sentence = self.projection_A(denoised_sentence)  # shape : [batch_size, sequence_length]
                #denoised_sentence = self.projection_B(denoised_sentence)  # shape : [batch_size, 1]
                #denoised_sentence, _ = torch.max(denoised_sentence, dim=-1, keepdim=True)  # shape : [batch_size, 1]

                return denoised_sentence, denoised_masked_token_logits, denoised_token_logits
            else:
                denoised_masked_token_logits = denoised_sentence

                denoised_sentence = self.projection_A(denoised_sentence)  # shape : [batch_size, sequence_length]
                #denoised_sentence = self.projection_B(denoised_sentence)  # shape : [batch_size, 1]
                #denoised_sentence, _ = torch.max(denoised_sentence, dim=-1, keepdim=True)  # shape : [batch_size, 1]

                return denoised_sentence, denoised_masked_token_logits
        else:
            denoised_sentence = self.projection_A(denoised_sentence)  # shape : [batch_size, sequence_length]
            #denoised_sentence = self.projection_B(denoised_sentence)  # shape : [batch_size, 1]
            #denoised_sentence, _ = torch.max(denoised_sentence, dim=-1, keepdim=True)  # shape : [batch_size, 1]

            return denoised_sentence


# USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER
class TransformerDenoiser(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, max_noise_level):
        super(TransformerDenoiser, self).__init__()

        self.embedding = nn.Embedding(tokenizer.vocab_size, model_dim)
        #self.noise_level_embeddings = nn.Embedding(max_noise_level, model_dim)

        if not USE_ROPE:
            self.pos_encoder = PositionalEncoding(model_dim)
            self.pos_decoder = PositionalEncoding(model_dim)

        if USE_ROPE:
            # Use RoPE Transformer Encoder layers
            encoder_layers = RoPETransformerEncoderLayer(
                model_dim,
                num_heads,
                model_dim,
                batch_first=True
            )

            # Use RoPE Transformer Decoder layers
            decoder_layers = RoPETransformerDecoderLayer(
                model_dim,
                num_heads,
                model_dim,
                batch_first=True
            )
        else:
            # Use Transformer Encoder Layers from Pytorch library
            encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, model_dim, batch_first=True)
            # Use Transformer Decoder Layers from Pytorch library
            decoder_layers = nn.TransformerDecoderLayer(model_dim, num_heads, model_dim, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        # Layer Normalization to prevent vanishing gradients
        self.norm = nn.LayerNorm(model_dim)

        # SiLU layer
        self.SiLU = nn.SiLU()

        #Sigmoid layer
        #self.Sigmoid = nn.Sigmoid()

        """
        # Projection layer to map single token embedding back to logits space
        self.projection = nn.Sequential(
            nn.Linear(1, tokenizer.vocab_size),
            #nn.ReLU()  # no need of activation function before being fed into cross-entropy loss function
        )
        """

        # Projection layer (tie weights with embedding)
        self.projection = nn.Linear(model_dim, tokenizer.vocab_size)
        self.projection.weight = self.embedding.weight  # Weight tying

        """
        self.denoise_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU()
        )
        """
        # Convolutional denoise head does not depend on input_dim or
        # input sequence length.  This is helpful in NLP domain, because the
        # NLP model will see varying input sequence length
        # Weight normalization is one technique to address vanishing gradients
        self.denoise_head = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=3, padding=1)),
            #nn.SiLU(),
            #weight_norm(nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=3, padding=1)),
            #nn.SiLU(),
            #weight_norm(nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=3, padding=1)),
            #nn.ReLU()
        )

        # Apply Xavier/Glorot or He initialization
        self._initialize_weights()

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(param, torch.nn.Parameter):
                    if param.dim() > 1:  # Only apply to matrices, not biases
                        if 'self_attn' in name or 'multihead_attn' in name:
                            torch.nn.init.xavier_uniform_(param)  # Xavier for attention layers
                        else:
                            torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')  # He for ReLU-based layers
            elif 'bias' in name:
                torch.nn.init.zeros_(param)  # Biases are usually initialized to zero

        #nn.init.xavier_uniform_(self.projection[0].weight)
        #nn.init.zeros_(self.projection[0].bias)

    # for decoder only
    def _shift_right(self, input_ids, start_token_id):
        """
        Shift input_ids to the right by one position and prepend the start_token_id.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.size())
        shifted_input_ids[:, 0] = start_token_id
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        return shifted_input_ids

    def forward(self, inputs, tgt=None, input_pad_mask=None, mlm_mask=None):
        if isinstance(inputs, dict):
            src = inputs['input_ids']
        else:
            src = inputs

        # src: [batch_size, sequence_length]
        # Embed input tokens
        src = self.embedding(src.long())  # [batch_size, sequence_length, model_dim]
        #print(f"After nn.embedding(), src.shape = {src.shape}")

        # Saves memory
        del inputs

        # Add sequence length dimension
        #src = src.unsqueeze(1)
        if tgt is not None:
            #tgt = tgt.unsqueeze(2)
            # Determine the start token ID based on tokenizer and model
            if USE_PRETRAINED_T5:
                start_token_id = tokenizer.pad_token_id  # T5 uses pad_token_id as start token
            else:
                start_token_id = tokenizer.cls_token_id  # BERT uses cls_token_id as start token

            # Shift tgt to the right
            tgt = self._shift_right(tgt, start_token_id)
            tgt = self.embedding(tgt.long())  # [batch_size, sequence_length, model_dim]
            #print(f"After nn.embedding(), src.shape = {src.shape}")

        # Apply input masking for padding if provided
        if input_pad_mask is not None:
            src = src.masked_fill(input_pad_mask.unsqueeze(1), 0.0)
            #print(f"After masked_fill, src = {src}")

        if not USE_ROPE:
            # Add positional encodings
            src = self.pos_encoder(src)
            #print(f"src.shape = {src.shape}")

            if tgt is not None:
                #print(f"Before pos_decoder(), tgt.shape = {tgt.shape}")
                tgt = self.pos_decoder(tgt)

        # Pre-Norm: Apply LayerNorm before the encoder layer
        src = self.norm(src)

        # Pass through the transformer encoder
        memory = self.transformer_encoder(src)#, src_key_padding_mask=input_pad_mask)
        #print(f"memory.shape = {memory.shape}")

        # Add residual connection
        memory = memory + src

        # Pre-Norm: Apply LayerNorm before the decoder layer
        memory = self.norm(memory)
        if tgt is not None:
            tgt = self.norm(tgt)
            #print(f"tgt.shape = {tgt.shape}")

        # Decoder
        if tgt is not None:
            #print(f"Before transformer_decoder(), tgt.shape = {tgt.shape} , memory.shape = {memory.shape}")
            output = self.transformer_decoder(tgt, memory)#, tgt_key_padding_mask=input_pad_mask, memory_key_padding_mask=input_pad_mask)
            #print(f"After transformer_decoder(), output.shape = {output.shape}")
        else:
            output = memory  # bypass the decoder for EBM module under dWJS_SCORE mode

        if tgt is not None:
            #print(f"In TransformerDenoiser(), output.shape = {output.shape}, tgt = {tgt.shape}")
            # Add residual connection
            if USE_ROPE:
                output = output + tgt.mean(dim=1).unsqueeze(1)  # Residual connection
            else:
                output = output + tgt  # Residual connection
                output = output.mean(dim=1).unsqueeze(1)

        #print(f"In TransformerDenoiser(), output.shape = {output.shape}, src = {src.shape}")
        # Add residual connection
        output = output + src  # Residual connection

        # Apply normalization
        output = self.norm(output)

        # Transpose for Conv1d: (batch_size, model_dim, 1)
        output = output.transpose(1, 2)

        # Pass through the denoising head
        #print(f"Before denoise_head(), output.shape = {output.shape}, output = {output}")
        output = self.denoise_head(output)
        #print(f"After denoise_head(), output.shape = {output.shape}, output = {output}")

        # Transpose back to original shape: (batch_size, 1, model_dim)
        output = output.transpose(1, 2)

        # Add residual connection
        output = output + src

        # Add residual connection
        #output = output + memory

        # Add residual connection
        if tgt is not None:
            output = output + tgt.mean(dim=1).unsqueeze(1)  # Residual connection

        # Apply normalization
        output = self.norm(output)

        # Apply activation function
        output = self.SiLU(output)
        #output = self.Sigmoid(output)

        #print(f"output.shape = {output.shape}")  # shape: [batch_size, tgt_sequence_length, src_sequence_length]

        if ENABLE_MASK_LEARNING:  # there is a new token concatenated to tgt tensor
            # Get the most recent timestep prediction
            # We want to update denoised_sentence based on the prediction for the last token in the sequence
            #denoised_sentence = output[:, -1, :]  # Select the last timestep
            denoised_sentence = output.mean(dim=1)  # shape: [batch_size, model_dim]
        else:
            # Remove unnecessary dimension
            denoised_sentence = output.squeeze(1)  # shape: [batch_size, model_dim]

        #print(f"denoised_sentence.shape = {denoised_sentence.shape}")
        # denoised_sentence has a shape of [batch_size, sequence_length]

        # Projects a single token back to logits space, so this is the opposite of softmax operation
        if mlm_mask is not None:
            #print(f"mlm_mask.shape = {mlm_mask.shape}")
            denoised_masked_token_logits = self.projection(denoised_sentence)  # shape: [batch_size, vocab_size]
            #denoised_masked_token_logits = self.projection(denoised_sentence[mlm_mask.bool()].unsqueeze(-1))
            #print(f"denoised_masked_token_logits.shape = {denoised_masked_token_logits.shape}")

            #print(f"denoised_sentence.shape = {denoised_sentence.shape}, denoised_masked_token_logits.shape = {denoised_masked_token_logits.shape}, denoised_token_logits.shape = {denoised_token_logits.shape}")

            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                denoised_token_logits = self.projection(output)  # shape: [batch_size, sequence_length, vocab_size]
                #print(f"denoised_token_logits.shape = {denoised_token_logits.shape}")
                return denoised_sentence, denoised_masked_token_logits, denoised_token_logits
            else:
                return denoised_sentence, denoised_masked_token_logits
        else:
            return denoised_sentence


# will switch to transformer model due to varying input sequence length as well as
# the higher-order gradient issue as described in http://arxiv.org/abs/1907.05600
class EnergyBasedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnergyBasedModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def denoiser_model(noisy_y, mlm_mask=None, target_label=None, tgt=None, input_pad_mask=None):
    denoised_sentence = None
    denoised_masked_token_logits = None
    denoised_token_logits = None

    if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
        if mlm_mask is not None:
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                denoised_sentence, denoised_masked_token_logits, denoised_token_logits = denoiser(inputs=noisy_y, mlm_mask=mlm_mask)
            else:
                denoised_sentence, denoised_masked_token_logits = denoiser(inputs=noisy_y, mlm_mask=mlm_mask)
        else:
            denoised_sentence = denoiser(inputs=noisy_y, mlm_mask=mlm_mask)

    elif USE_PRETRAINED_T5:
        if mlm_mask is not None:
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                denoised_sentence, denoised_masked_token_logits, denoised_token_logits = denoiser(input_ids=noisy_y, target_label=target_label, mlm_mask=mlm_mask)
            else:
                denoised_sentence, denoised_masked_token_logits = denoiser(input_ids=noisy_y, target_label=target_label, mlm_mask=mlm_mask)
        else:
            denoised_sentence = denoiser(input_ids=noisy_y, target_label=target_label, mlm_mask=mlm_mask)

    else:  # USE_CUSTOM_TRANSFORMER
        # Use denoiser with the current noisy sequence (src) and current target sequence (tgt)
        if mlm_mask is not None:
            if USE_CUSTOM_TRANSFORMER_ENCODER_DECODER:
                if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                    denoised_sentence, denoised_masked_token_logits, denoised_token_logits = denoiser(noisy_y, tgt, input_pad_mask, mlm_mask)
                else:
                    denoised_sentence, denoised_masked_token_logits = denoiser(noisy_y, tgt, input_pad_mask, mlm_mask)
            else:  # USE_CUSTOM_TRANSFORMER_ENCODER
                if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                    denoised_sentence, denoised_masked_token_logits, denoised_token_logits = denoiser(noisy_y, None, None, mlm_mask)  # for isolating decoder related code
                else:
                    denoised_sentence, denoised_masked_token_logits = denoiser(noisy_y, None, None, mlm_mask)  # for isolating decoder related code
        else:
            if USE_CUSTOM_TRANSFORMER_ENCODER_DECODER:
                denoised_sentence = denoiser(noisy_y, tgt, input_pad_mask, mlm_mask)
            else:  # USE_CUSTOM_TRANSFORMER_ENCODER
                denoised_sentence = denoiser(noisy_y, None, None, mlm_mask)  # for isolating decoder related code

    return denoised_sentence, denoised_masked_token_logits, denoised_token_logits


# sequential monte-carlo (SMC)
def proposal(particle):
    # Clone the particle to avoid in-place modifications
    new_particle = particle.clone()
    batch_size, seq_length = new_particle.size()

    # Number of modifications per sample
    num_modifications = max(1, int(0.05 * seq_length))  # Modify 5% of tokens

    # Generate random indices and tokens for all samples
    indices = torch.randint(0, seq_length, (batch_size, num_modifications), device=particle.device)
    random_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, num_modifications), device=particle.device)

    # Create batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=particle.device).unsqueeze(1).expand(-1, num_modifications)

    # Modify the new_particle tensor
    new_particle[batch_indices, indices] = random_tokens.float().to(device)

    return new_particle

# sequential monte-carlo (SMC)
def compute_weights(particles, ebm):
    # particles: list of length N_particles, each tensor of shape [batch_size, seq_length]
    #print(f"particles[0].shape = {particles[0].shape}")
    batch_size = particles[0].size(0)
    N_particles = len(particles)

    with torch.no_grad():
        # Compute scalar energies for all particles
        energies = torch.stack([ebm(particle).sum(dim=1) for particle in particles], dim=1)  # Shape: [batch_size, N_particles]

    #print(f"energies.shape = {energies.shape}")

    # Convert energies to weights
    weights = torch.exp(-energies)  # Lower energy = higher probability
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize over particles

    # Verify the shape of weights
    #print(f"weights shape after computation: {weights.shape}")  # Should be [batch_size, N_particles]

    return weights  # Shape: [batch_size, N_particles]

# sequential monte-carlo (SMC)
def resample(particles, weights):
    # particles: list of length N_particles, each tensor of shape [batch_size, seq_length]
    # weights: tensor of shape [batch_size, N_particles]
    batch_size = particles[0].size(0)
    N_particles = len(particles)
    seq_length = particles[0].size(1)

    # Stack particles to create a tensor of shape [N_particles, batch_size, seq_length]
    particles_tensor = torch.stack(particles, dim=0)  # Shape: [N_particles, batch_size, seq_length]

    # Transpose to shape [batch_size, N_particles, seq_length]
    particles_tensor = particles_tensor.permute(1, 0, 2)  # Shape: [batch_size, N_particles, seq_length]

    # Perform batch-wise multinomial sampling
    # particle_indices: tensor of shape [batch_size, N_particles]
    particle_indices = torch.multinomial(weights, N_particles, replacement=True)

    # Create batch indices
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, N_particles)  # Shape: [batch_size, N_particles]

    # Gather the resampled particles
    resampled_particles_tensor = particles_tensor[batch_indices, particle_indices]  # Shape: [batch_size, N_particles, seq_length]

    # Transpose back to [N_particles, batch_size, seq_length]
    resampled_particles_tensor = resampled_particles_tensor.permute(1, 0, 2)  # Shape: [N_particles, batch_size, seq_length]

    # Split into a list of particles
    resampled_particles = [resampled_particles_tensor[i] for i in range(N_particles)]

    return resampled_particles


# USE_MAFBM
class MA_fBM:
    def __init__(self, hurst, T, n_steps, K):
        """
        Initialize MA-fBM with optimal coefficients from the GFDM paper.

        Args:
            hurst (float): Hurst parameter H in (0,1)
            T (float): Terminal time
            n_steps (int): Number of time steps
            K (int): Number of Ornstein-Uhlenbeck processes
            device (str): 'cuda' for GPU, 'cpu' for CPU
        """
        self.H = hurst
        self.T = T
        self.n_steps = n_steps
        self.K = K
        self.dt = T / n_steps

        # Calculate gamma grid according to paper
        n = (K + 1) / 2
        r = 1.5  # r > 1 , Geometric spacing parameter from paper
        self.gammas = torch.tensor([r**(k-n) for k in range(1, K+1)],
                                 device=device, dtype=torch.float32)

        # Compute optimal coefficients
        self.weights = self._compute_optimal_coefficients()

    def _torch_gamma_inc(self, a, x):
        """
        PyTorch implementation of regularized lower incomplete gamma function P(a,x)
        P(a,x) = 1/Γ(a) ∫₀ˣ t^(a-1) e^(-t) dt

        Args:
            a: Shape parameter (Hurst index + 0.5)
            x: Upper limit of integration (gamma_k * T)

        Uses series expansion approximation:
        e^(-t) = ∑_{n=0}^∞ (-t)^n/n!

        γ(a,x) = ∫₀ˣ t^(a-1) e^(-t) dt
               = ∫₀ˣ t^(a-1) ∑_{n=0}^∞ (-t)^n/n! dt
               = ∑_{n=0}^∞ (-1)^n/n! ∫₀ˣ t^(a-1+n) dt
               = ∑_{n=0}^∞ (-1)^n/n! [t^(a+n)/(a+n)]₀ˣ
               = ∑_{n=0}^∞ (-1)^n x^(a+n)/(n!(a+n))

        P(a,x) = γ(a,x)/Γ(a)
               = (1/Γ(a)) ∑_{n=0}^∞ (-1)^n x^(a+n)/(n!(a+n))
        """
        eps = 1e-8  # Convergence threshold
        iterations = 100  # Maximum iterations

        # Convert inputs to tensors
        a = torch.tensor(a, device=device, dtype=torch.float32)
        x = torch.as_tensor(x, device=device, dtype=torch.float32)

        # Initialize sum
        result = torch.zeros_like(x, device=device)
        temp = torch.ones_like(x, device=device)
        temp = x**a  # Initialize and start with x^a
        factorial = 1

        # Compute series expansion
        for n in range(iterations):
            # Update term: term[n] = term[n-1] * (-1)^n * x / (n! * (a+n))
            # Use -x instead of x to get (-1)^n term
            factorial = factorial * (n + 1) if n > 0 else 1
            temp = temp * (-x) / (factorial * (a + n))  # self-multiply gives the x^n term
            result += temp

            # Check convergence when terms become very small (< eps)
            # Adding more terms won't significantly change the result
            if torch.all(torch.abs(temp) < eps):
                break

        # P(a,x) = result / Γ(a)
        return result / torch.exp(torch.lgamma(a))

    def _compute_optimal_coefficients(self):
        """
        Compute optimal approximation coefficients following the GFDM paper's equation (9) : Aω = b
        """
        # Create matrix A and vector b
        A = torch.zeros((self.K, self.K), device=device)
        b = torch.zeros(self.K, device=device)

        # Compute matrix A
        for i in range(self.K):
            for j in range(self.K):
                gamma_i = self.gammas[i]
                gamma_j = self.gammas[j]

                A[i,j] = (2*self.T + (torch.exp(-(gamma_i + gamma_j)*self.T) - 1) /
                         (gamma_i + gamma_j)) / (gamma_i + gamma_j)

        # Compute vector b
        z = self.H + 0.5
        for k in range(self.K):
            gamma_k = self.gammas[k]
            x = gamma_k * self.T

            # Compute regularized incomplete gamma functions
            P_1 = self._torch_gamma_inc(z, x)       # P(H+1/2, γₖT)
            P_2 = self._torch_gamma_inc(z + 1, x)   # P(H+3/2, γₖT)

            # Implementation of b formula,
            # see Appendix D.2 (Type II) of [Variational Inference for SDEs Driven by Fractional Noise](http://arxiv.org/abs/2310.12975)
            b[k] = (self.T * P_1 / (gamma_k**z) -
                   (self.H + 0.5) * P_2 / (gamma_k**(z+1)))

        # Solve linear system Aω = b for optimal weights
        if device_str == 'mps':
            weights = self.conjugate_gradient_solver(A, b)
        else:
            weights = torch.linalg.solve(A, b)  # linalg.solve has no MPS backend support yet
        return weights

    def conjugate_gradient_solver(self, A, b, tol=1e-6, max_iter=1000):
        """
        Solve Ax = b using the Conjugate Gradient method.

        Args:
            A (torch.Tensor): Symmetric positive definite matrix of shape [N, N].
            b (torch.Tensor): Right-hand side vector of shape [N].
            tol (float): Tolerance for convergence.
            max_iter (int): Maximum number of iterations.

        Returns:
            x (torch.Tensor): Solution vector of shape [N].
        """
        x = torch.zeros_like(b)  # Initial guess
        r = b - torch.matmul(A, x)
        p = r.clone()
        rs_old = torch.dot(r, r)

        for i in range(max_iter):
            Ap = torch.matmul(A, p)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)

            if torch.sqrt(rs_new) < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    @torch.no_grad()
    def simulate(self, return_processes=False):
        """
        Generate a sample path of MA-fBM.

        Args:
            return_processes (bool): If True, also return individual OU processes

        Returns:
            tuple: Time points, MA-fBM path, and optionally OU processes
        """
        times = torch.linspace(0, self.T, self.n_steps, device=device)
        ou_processes = torch.zeros((self.K, self.n_steps),
                                 device=device)

        # Generate Brownian increments
        dW = torch.randn(self.K, self.n_steps-1,
                        device=device) * torch.sqrt(torch.tensor(self.dt, device=device))

        # Simulate OU processes
        for i in range(1, self.n_steps):
            ou_processes[:, i] = (ou_processes[:, i-1] * torch.exp(-self.gammas * self.dt) +
                                dW[:, i-1])

        # Combine OU processes to get MA-fBM
        mafbm_path = torch.sum(self.weights.reshape(-1,1) * ou_processes, dim=0)

        if return_processes:
            return times, mafbm_path, ou_processes
        return times, mafbm_path


# Define the Langevin MCMC step function based on Algorithm 4 of the walk-jump paper
def langevin_mcmc_step_advanced(y, v, mlm_mask, input_pad_mask, ebm, denoiser, step_size, u=1.0, gamma=0.1, K=1):
    # See Algorithm 1 of [Langevin Dynamics with Variable Coefficients and Nonconservative Forces: From Stationary States to Numerical Methods](https://www.mdpi.com/1099-4300/19/12/647)

    step_size = torch.tensor(step_size, device=device)  # Convert step_size to tensor

    # Gather the correct tokens from the train data based on the mask positions
    #train_data_correct = y[mask.bool()].long()

    def energy_func(input_tensor):
        return ebm(input_tensor).sum()  # This is where the EBM is used

    """
    Formula Breakdown:
       - (w_t - w_t_minus_1)**2 measures the squared distance between the two states
       - exp(-((w_t - w_t_minus_1)**2) / (2 * self.sigma**2)) gives the unnormalized probability
       - The denominator (self.sigma * math.sqrt(2 * math.pi)) normalizes the probability

    Walk-Jump Perspective:
       - In the "walk" phase, we're exploring the smoothed data space using this forward process
       - This Gaussian model allows for continuous transitions between states, which is key to the walk-jump approach for discrete data
       - The "jump" phase then projects these smoothed states back to the discrete space
    """
    def forward_process(w_t, w_t_minus_1):
        # Normalize inputs such that `diff` does not result in very large values
        w_t_norm = (w_t - w_t.mean()) / w_t.std()
        w_t_minus_1_norm = (w_t_minus_1 - w_t_minus_1.mean()) / w_t_minus_1.std()

        # Apply forward process on normalized inputs
        diff = w_t_norm - w_t_minus_1_norm
        return torch.exp(-(diff**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))

    # p(ŵ|w_t, w_{t-1})
    def auxiliary_prob(w_hat, w_t, w_t_minus_1, denoiser):
        # Construct the noisy input
        noisy_input = w_t  # Shape: [batch_size, seq_len]

        # Provide w_t_minus_1 as context
        context = w_t_minus_1  # Shape: [batch_size, seq_len]

        #print(f"noisy_input.shape = {noisy_input.shape}, context.shape = {context.shape}")

        # Apply the denoiser to predict possible clean versions of w_t, given w_t and w_{t-1}
        denoised_sentence, denoised_masked_token_logits = denoiser(src=noisy_input, tgt=context, input_pad_mask=input_pad_mask, mlm_mask=mlm_mask)  # Shape: [batch_size, seq_len] , [batch_size, vocab_size]

        # Get probabilities
        probs = F.softmax(denoised_masked_token_logits.squeeze(1), dim=-1)  # Shape: [batch_size, vocab_size]

        # The probability the denoiser assigns to ŵ can be interpreted as how likely ŵ is to be the "clean" version of w_t, given w_t and w_{t-1}
        # Return the probability of w_hat
        return probs[0, w_hat]

    def prior():
        # Could be uniform or based on token frequencies in the dataset
        return 1.0 / tokenizer.vocab_size

    # to compute p(w_{t-1}|w_t, ŵ) which is the probability of the previous state given the current state and the auxiliary variable
    def compute_transition_probability(w_t, w_hat, w_t_minus_1, denoiser, position=None, prev_word=None):
        print(f"w_t.shape = {w_t.shape}, w_hat = {w_hat}, w_t_minus_1.shape = {w_t_minus_1.shape}")

        # p(w_t|w_{t-1})
        p_w_t_given_w_t_minus_1 = forward_process(w_t, w_t_minus_1)
        print(f"forward_process() gives {p_w_t_given_w_t_minus_1} which has a shape of {p_w_t_given_w_t_minus_1.shape}")

        # p(ŵ|w_t, w_{t-1})
        p_w_hat_given_w_t_w_t_minus_1 = auxiliary_prob(w_hat, w_t, w_t_minus_1, denoiser=denoiser)
        print(f"auxiliary_prob() gives {p_w_hat_given_w_t_w_t_minus_1} which has a shape of {p_w_hat_given_w_t_w_t_minus_1.shape}")

        # p(w_{t-1})
        p_w_t_minus_1 = prior() # prior(w_t_minus_1, position, prev_word)

        # Compute p(w_{t-1}, w_t, ŵ)
        p_w_t_minus_1_w_t_w_hat = p_w_t_given_w_t_minus_1 * p_w_hat_given_w_t_w_t_minus_1 * p_w_t_minus_1
        print(f"p_w_t_minus_1_w_t_w_hat = {p_w_t_minus_1_w_t_w_hat}")

        # Approximate p(w_t, ŵ) using the current w_{t-1}
        # Avoids explicit marginalization over all possible w_{t-1} using `vocab_size` loop iterations
        p_w_t_w_hat = p_w_t_given_w_t_minus_1 * p_w_hat_given_w_t_w_t_minus_1
        print(f"p_w_t_w_hat = {p_w_t_w_hat}")

        print(f"p_w_t_minus_1_w_t_w_hat.shape = {p_w_t_minus_1_w_t_w_hat.shape}, p_w_t_w_hat.shape = {p_w_t_w_hat.shape}")

        # Compute p(w_{t-1}|w_t, ŵ) using Bayes' rule
        p_w_t_minus_1_given_w_t_w_hat = p_w_t_minus_1_w_t_w_hat / p_w_t_w_hat

        # transition probability
        return p_w_t_minus_1_given_w_t_w_hat


    # KL divergence function using one-hot encoded target
    # See Algorithm 2 of [Protein Design with Guided Discrete Diffusion](http://arxiv.org/abs/2305.20009)
    def kl_div_func(model_output):
        #target_distribution = F.one_hot(train_data_correct, num_classes=tokenizer.vocab_size).float()
        #target_distribution = torch.rand_like(model_output) / model_output.size(-1)

        p_h = torch.zeros_like(y)

        with torch.no_grad():  # for saving RAM memory consumption
            for w in range(tokenizer.vocab_size):
                # here, model is the denoiser
                p_transition = compute_transition_probability(y, w, model_output, denoiser)
                print(f"p_transition = {p_transition}")
                # Calculate p_h
                p_h += p_transition * model_output

            # Add a small epsilon to avoid log(0) in KL divergence calculation
            epsilon = 1e-8
            p_h = p_h + epsilon

            # Normalize to ensure sum equals 1 such that p_h is a valid probability distribution
            p_h = p_h / p_h.sum(dim=-1, keepdim=True)

            # Ensure model_output is also a valid probability distribution
            model_output = F.softmax(model_output, dim=-1)
            #print(f"model_output.shape = {model_output.shape}, p_h.shape = {p_h.shape}")

        # Re-enable gradients for the KL divergence computation
        with torch.enable_grad():
            # the following is semantically similar to kl_div(current_prob.log(), previous_prob)
            return F.kl_div(model_output.log(), p_h, reduction='batchmean')

    # gradients for a Gaussian likelihood with prior N(0, I)
    def grad_U_theta(ebm, theta, X, mlm_mask=None):
        """
        Gradient of U with respect to theta.
        Args:
            ebm: BERT-based EBM (computes -log p(y|x)).
            theta: Tokenized sequences (Tensor, shape [batch_size, seq_length]).
            X: Latent particles (Tensor, shape [N_particles, batch_size, seq_length]).
            mlm_mask: Optional mask for MLM tasks (Tensor, shape [batch_size, seq_length]).
        Returns:
            Gradient w.r.t theta (Tensor, shape [batch_size, seq_length]).
        """
        # Ensure theta requires gradients
        theta = theta.requires_grad_(True)

        # Initialize gradient accumulator
        grad_sum = 0

        # Iterate over particles
        for x in X:
            # Compute energy for this particle
            #energy = ebm(theta, mlm_mask=mlm_mask).sum()

            # Compute gradient w.r.t theta for this particle
            grad = torch.autograd.functional.jacobian(energy_func, theta)
            '''
            grad = torch.autograd.grad(
                outputs=energy,
                inputs=theta,
                create_graph=True
            )[0]
            '''

            # Accumulate gradient
            grad_sum += grad

        # Average the gradients over all particles
        grad_theta = grad_sum / X.size(0)

        return grad_theta

    def prior_grad(X):
        """
        Gradient of the log prior w.r.t X.
        Args:
            X: Latent particles (Tensor, shape [N_particles, batch_size, seq_length]).
        Returns:
            Gradient w.r.t X (Tensor, same shape as X).
        """
        return -X

    def grad_U_X(ebm, X, mlm_mask=None, prior_grad=None):
        """
        Gradient of U with respect to X.
        Args:
            ebm: BERT-based EBM (computes -log p(y|x)).
            X: Latent particles (Tensor, shape [N_particles, batch_size, seq_length]).
            mlm_mask: Optional mask for MLM tasks (Tensor, shape [batch_size, seq_length]).
            prior_grad: Optional function to compute gradient of log prior w.r.t X.
        Returns:
            Gradient w.r.t X (Tensor, same shape as X).
        """
        # Ensure X requires gradients
        X = X.requires_grad_(True)

        # Initialize gradient accumulator
        grad_sum = 0

        # Iterate over particles
        for x in X:
            # Compute energy for this particle
            #energy = ebm(x, mlm_mask=mlm_mask).sum()

            #print(f"in grad_U_X, x has a shape of {x.shape}")

            # Compute gradient w.r.t X for this particle
            grad = torch.autograd.functional.jacobian(energy_func, x)
            '''
            grad = torch.autograd.grad(
                outputs=energy,
                inputs=x,
                create_graph=True
            )[0]
            '''

            # Accumulate gradient
            grad_sum += grad

        # Average the gradients over all particles
        grad_X = grad_sum / X.size(0)

        # Add prior gradient if specified,
        # In the case of NLP domain, it is hard to define prior grad when it is not gaussian-distributed.
        if prior_grad is not None:
            grad_X += prior_grad(X)

        return grad_X


    y.requires_grad_(True)
    #print(f"y shape: {y.shape}, y grad: {y.requires_grad}")
    batch_size, particle_dim = y.shape  # reusing the shape dimension of tensor y for particles X

    # Initialize variables for OBABO
    #X = torch.stack([y.clone() for _ in range(N_particles)], dim=0)  # this is variable X in KIPLMC2
    theta = torch.rand_like(y, device=device, requires_grad=True) * (tokenizer.vocab_size-1)  # θ (parameter)
    v_theta = torch.randn_like(theta, device=device)                       # V^θ_0 (velocity for θ)
    X = torch.rand(N_particles, batch_size, particle_dim, device=device, requires_grad=True) * (tokenizer.vocab_size-1)  # X^i (particles)
    v_X = torch.randn_like(X, device=device)                               # V^{X,i}_0 (velocities for X)

    for k in range(K):
        # Compute gradients
        if USE_OBABO:
            grad_theta = grad_U_theta(ebm=ebm, theta=y, X=X)  # Gradient w.r.t theta
            grad_X = grad_U_X(ebm=ebm, X=X)                   # Gradient w.r.t X
        else:
            if USE_dWJS_ENERGY:
                # Compute gradients
                #print(f"torch.autograd.grad(energy, y, allow_unused=True) = {torch.autograd.grad(energy, y, allow_unused=True)}")
                #grad_energy = torch.autograd.grad(energy, y)[0]
                #grad_energy = y.grad
                grad_energy = torch.autograd.functional.jacobian(energy_func, y)
                total_grad = grad_energy
            else:  # USE_dWJS_SCORE
                with torch.no_grad():  # for saving RAM memory consumption
                # energy and score are related by a derivative
                # score = (denoised - inputs) / (sigma ** 2) = ∇log p(y) = -∇f(y) = -∇energy = -∇ebm
                    x_hat, _, _ = denoiser_model(y)
                score = (x_hat - y) / (sigma**2)  # This is ∇log p(y)
                total_grad = -1 * score

            if USE_GRAD_KL:
                with torch.no_grad():  # for saving RAM memory consumption
                    model_output, _, _ = denoiser_model(y)
                grad_kl = torch.autograd.functional.jacobian(kl_div_func, model_output)

                # Combine gradients, we subtract grad_kl because we want to always minimize KL divergence
                #print(f"grad_energy.shape = {grad_energy.shape}, grad_kl.shape = {grad_kl.shape}")
                total_grad = total_grad - grad_kl

            grad_theta = total_grad
            grad_X = total_grad

        # See equation 6 of [Rational Construction of Stochastic Numerical Methods for Molecular Sampling](https://arxiv.org/abs/1203.5428)
        # "To slightly simplify the presentation that follows, we make the change of variables q -> M^(−1/2) * q, p -> M^(+1/2) * p,
        # with a corresponding adjustment of the potential; this is equivalent to assuming M = I"
        # q is equivalent to y, p is equivalent to v, M is equivalent to u
        # Besides, the following equation v is similar in form to langevin dynamic recursive equation (in discrete domain) : x[t+1] = x[t] + τsθ(x[t]) + sqrt(2τ)z ,
        # where τ is the step_size, and z is the eps (gaussian white-noise)
        # there is exp() inside the following equation v is because it is the result of solving the recursive equation using https://en.wikipedia.org/wiki/Magnus_expansion
        # after it is being rearranged as differential equation (in continous domain)
        eps = torch.randn_like(y)

        # Sample alpha from [0, 1]
        alpha = torch.rand(1).to(device)  # This corresponds to the random midpoint step

        # KIPLMC2 OBABO step (O+B): First velocity update with noise and friction
        v_theta = torch.exp(-gamma * step_size) * v_theta - u * step_size * torch.exp(-2 * gamma * (step_size - alpha * step_size)) * grad_theta + torch.sqrt(u * (1 - torch.exp(-2 * gamma * step_size))) * eps
        v_X = torch.exp(-gamma * step_size) * v_X - u * step_size * torch.exp(-2 * gamma * (step_size - alpha * step_size)) * grad_X + torch.sqrt(u * (1 - torch.exp(-2 * gamma * step_size))) * eps

        # equation (4) of [The Randomized Midpoint Method for Log-Concave Sampling](https://arxiv.org/abs/1909.05503) uses x
        # while KIPLMC2 from [Kinetic Interacting Particle Langevin Monte Carlo](http://arxiv.org/abs/2407.05790) uses θ, but here we use symbol y instead
        # we are also not using the midpoint n+1/2 method due to extra compute logic for gradients
        # KIPLMC2 OBABO step (A): Update positions θ and X^i
        y = y + (step_size / 2) * v_theta
        X = X + (step_size / 2) * v_X
        #print(f"y.max() = {y.max()}, y.min() = {y.min()}")

        #print(f"u = {u} , step_size = {step_size}")
        #v = v + u * (step_size / 2) * g  # only needed in walk-jump paper Algorithm 4, but not in KIPLMC2 OBABO

        # We are using underdamped langevin dynamics, see equations (4) and (5) of [The Randomized Midpoint Method for Log-Concave Sampling](https://arxiv.org/abs/1909.05503)
        # If we remove the first term below, then it will become overdamped langevin dynamics.
        # This is also similar to the KIPLMC1 (which uses Exponential Integrators) approach in [Kinetic Interacting Particle Langevin Monte Carlo](http://arxiv.org/abs/2407.05790)
        # KIPLMC2 OBABO step (B+O): Second velocity update with noise and gradients
        v_theta = torch.exp(-gamma * step_size) * v_theta - u * step_size * torch.exp(-2 * gamma * (step_size - alpha * step_size)) * grad_theta + torch.sqrt(u * (1 - torch.exp(-2 * gamma * step_size))) * eps
        v_X = torch.exp(-gamma * step_size) * v_X - u * step_size * torch.exp(-2 * gamma * (step_size - alpha * step_size)) * grad_X + torch.sqrt(u * (1 - torch.exp(-2 * gamma * step_size))) * eps

    # Detaching during the walk phase ensures that the NN model is used as a fixed, pretrained guide
    # for sample generation, preventing any unintended parameter updates.
    return y.detach()

def langevin_mcmc_step(y, model, step_size):
    y.requires_grad_(True)
    energy = model(y).sum()
    energy.backward(retain_graph=True)
    #print(f"torch.autograd.grad(energy, y, allow_unused=True) = {torch.autograd.grad(energy, y, allow_unused=True)}")
    grad = torch.autograd.grad(energy, y)[0]
    #grad = y.grad
    y_next = y - step_size * grad + torch.sqrt(torch.tensor(2 * step_size, device=y.device)) * torch.randn_like(y)
    return y_next.detach()


"""
This implementation below follows the stabilization mechanism described in diffusion forcing paper:

1. The input "sequence" are tokens fully diffused to the maximum noise level (sigma_max).
2. It then denoises tokens one by one, starting from the first token.
3. For each token, it gradually reduces the noise level from sigma_max to sigma_min.
4. When moving to the next token, it treats the previously denoised tokens as slightly noisy ground truth by adding a small amount of noise (sigma_max / M).
5. For subsequent tokens, it ensures that the noise level is at least as high as the noise level of the previously denoised tokens.

This approach should help prevent the accumulation of single-step errors in autoregressive sampling by treating predicted tokens as noisy ground truth, rather than perfect observations.
"""
def apply_noise(sequence, M, sigma_min, sigma_max):
    # M is the number of denoising (or jump) steps
    seq_len = len(sequence)
    noisy_seq = sequence.clone()

    # Gradually denoise tokens while implementing the stabilization mechanism
    for t in range(seq_len):
        #print(f"t = {t}")

        for m in range(M-1, -1, -1):  # Start from highest noise and decrease
            #print(f"m = {m}")

            # Use a slightly higher noise level for previously denoised tokens
            # Decreases noise level gradually as denoising takes place
            noise_level = sigma_max - (sigma_max - sigma_min) * (m / (M-1)) * ((seq_len - t) / seq_len)
            #print(f"noise_level = {noise_level}")

            # Apply denoising step
            noisy_seq[t] = sequence[t] + torch.randn_like(sequence[t]) * noise_level

        # After fully denoising a token, slightly increase its noise level for stability
        if t < seq_len - 1 and m == 0:
            noisy_seq[t] += torch.randn_like(sequence[t]) * (sigma_min + (sigma_max - sigma_min) / M)

    return noisy_seq


def walk_jump_sampling(init_y, mlm_mask, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask, target_label=None):
    if isinstance(init_y, dict):
        y = init_y['input_ids']
    else:
        y = init_y

    batch_size, seq_length = y.size()

    """
    1.	Initialization of tgt:
	    tgt starts as a sequence of [MASK] tokens, which will be updated as the model generates new tokens.
	2.	Using the Transformer Decoder:
	    In each step of walk_jump_sampling, the denoiser now takes both noisy_y (as src) and tgt. The decoder will use the context from noisy_y to generate predictions for tgt.
	3.	Updating tgt:
	    After each step, a new token is generated (using denoised_y), and this token is added to tgt. This way, tgt grows in length during each iteration, effectively generating the sequence step by step.
	4.	Sampling Process:
	    The loop continues, gradually building the output sequence one token at a time. The use of tgt allows the model to generate sequences dynamically, making decisions based on the tokens generated so far.
    """
    if ENABLE_MASK_LEARNING:
        # Initialize tgt with just the CLS (to indicate beginning of sequence) token for custom transformer using BERT tokenizer
        # CLS [sequence 1] SEP [sequence 2] SEP
        #tgt = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long, device=device)  # for inference, but not for training

        # Determine the start token ID based on tokenizer and model
        if USE_PRETRAINED_T5:
            start_token_id = tokenizer.pad_token_id  # T5 uses pad_token_id as start token
        else:
            start_token_id = tokenizer.cls_token_id  # BERT uses cls_token_id as start token

        if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:  # BERT models do not have decoders, so no need 'tgt'
            tgt = None
        else:
            # Shift tgt to the right
            tgt = denoiser._shift_right(y, start_token_id)
    else:
        tgt = None

    # Define the step size schedule function
    def get_step_size(t, initial_step_size=1e-3, gamma=0.55):
        return initial_step_size / (t + 1) ** gamma


    # Preparation steps for walk stage:
    # NLP token sampling search space is huge with tokenizer.vocab_size = 30522
    # Protein token sampling search space is only 20
    # So, we might have to temporarily disable the following rand_like() operation for now
    #y = torch.rand_like(y) * tokenizer.vocab_size + torch.randn_like(y) * sigma

    if mlm_mask is not None and (USE_SMC or USE_MAFBM or USE_MCMC):  # non-masked language modeling task
        # Add more noise to masked positions
        #y[mlm_mask] = torch.rand_like(y[mlm_mask]) * tokenizer.vocab_size + torch.randn_like(y[mlm_mask]) * sigma

        # Add less noise to unmasked positions
        #y[~mlm_mask] = y[~mlm_mask] + torch.randn_like(y[~mlm_mask]) * (sigma * 0.1)

        #y = y + torch.randn_like(y) * sigma
        pass

    v = torch.randn_like(y)  # Initialize velocity from a standard normal distribution
    #v = torch.zeros_like(y)  # Initialize velocity

    # walk then jump, so num_walk_steps == num_jump_steps
    for t in range(num_walk_steps):
        # walk stage (sampling process is guided by using EBM)
        if USE_SMC:  # see [Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo](https://arxiv.org/abs/2404.17546)
            # Initialize particles
            particles = [y for _ in range(N_particles)]

            for t in range(num_smc_steps):
                # Prediction Step: Propose new particles
                particles = [proposal(particle) for particle in particles]

                # Weighting Step: Compute weights based on EBM energy
                weights = compute_weights(particles, ebm)
                #print(f"weights shape: {weights.shape}")  # Shape: [batch_size, N_particles]

                # Resampling Step: Resample particles based on weights
                particles = resample(particles, weights)

            # After SMC steps, select the particle with the highest weight for each batch item
            # Vectorized implementation
            batch_size = particles[0].size(0)

            # Stack particles into a tensor: [N_particles, batch_size, seq_length]
            particles_tensor = torch.stack(particles, dim=0)  # [N_particles, batch_size, seq_length]

            # Permute to [batch_size, N_particles, seq_length]
            particles_tensor = particles_tensor.permute(1, 0, 2)  # [batch_size, N_particles, seq_length]

            # Compute the indices of the best particles for each batch item
            best_particle_indices = torch.argmax(weights, dim=1)  # [batch_size]

            # Prepare batch indices for advanced indexing
            batch_indices = torch.arange(batch_size, device=weights.device)  # [batch_size]

            # Select the final particles using advanced indexing
            final_particles = particles_tensor[batch_indices, best_particle_indices, :]  # [batch_size, seq_length]

            # Update y with the selected final particles
            y = final_particles  # [batch_size, seq_length]

        elif USE_MAFBM:  # see [Generative Fractional Diffusion Models](http://arxiv.org/abs/2310.17638)
            # Create and simulate MA-fBM
            if t == 0:
                ma_fbm = MA_fBM(hurst, T=T_fbm, n_steps=input_dim, K=K_fbm)
            times, path, ou_processes = ma_fbm.simulate(return_processes=True)

            # Print shapes to understand the dimensions
            #print(f"Initial path shape: {path.shape}")  # Should be [n_steps]
            #print(f"Initial ou_processes shape: {ou_processes.shape}")  # Should be [K, n_steps]

            # Reshape path: [n_steps] -> [batch_size, n_steps]
            path = path.unsqueeze(0).expand(batch_size, -1)

            # Add MA-fBM contribution to y: [batch_size, seq_length]
            y = y + walk_step_size * sigma * path

            # saves memory
            del times
            del path
            del ou_processes

        elif USE_MCMC:  # see [Provable Benefit of Annealed Langevin Monte Carlo for Non-log-concave Sampling](http://arxiv.org/abs/2407.16936)
            # Annealing process such that step size decays across time, ensuring convergence
            current_walk_step_size = get_step_size(t, initial_step_size=walk_step_size)

            if USE_ALGORITHM_1_OR_4:
                y = langevin_mcmc_step(y, ebm, current_walk_step_size)  # Langevin MCMC sampling
            else:
                y = langevin_mcmc_step_advanced(y, v, mlm_mask, input_pad_mask, ebm, denoiser, current_walk_step_size)  # Update using advanced Langevin dynamics

            #print(f"USE_MCMC, y = {y}")
            assert not y.isnan().any(), "mcmc is giving NaN output !!!"

        else:
            # Walk sampling stage (which serves the purpose of forward noising) is only needed in
            # Image Denoising: Removing noise from images corrupted by Gaussian noise.
	        # Text Denoising: Correcting sentences with randomly inserted, deleted, or swapped words.
            pass  # no need of walk sampling stage for masked language/image model downstream task

        # jump stage
        if USE_PRECOMPUTE_NOISE_SCHEDULE:
            # ONLY works for static pre-compute noise schedule with fixed input sequence length
            sigma_t = noise_schedule[t]
        else:
            sigma_t = sigma  # there is only a single denoising level for walk-jump equation

        if ADD_EXTRA_GAUSSIAN_NOISE:
            # Add noise for reverse denoising process (this step might be optional since we have a denoiser() NN module further down)
            if USE_DIFFUSION_FORCING:
                # Add noise level according to diffusion forcing scheme
                noisy_y = apply_noise(y, num_jump_steps, sigma_min, sigma_max)
            else:
                # Add noise level according to full sequence diffusion scheme
                noise = torch.randn_like(y) * sigma_t
                noisy_y = y + noise
        else:
            noisy_y = y  # for isolating the extra optional step just above

        assert not noisy_y.isnan().any(), "noisy_y is giving NaN output !!!"

        if (USE_SMC or USE_MAFBM or USE_MCMC):  # masked language modeling task
            # Scales to range of [0, tokenizer.vocab_size-1]
            noisy_y = noisy_y - noisy_y.min()
            noisy_y = noisy_y / noisy_y.max()
            noisy_y = noisy_y * (tokenizer.vocab_size - 1)

        assert not noisy_y.isnan().any(), "noisy_y is giving NaN output !!!"

        # checks for potential issues of NLP model's input range
        #print(f"noisy_y.max() = {noisy_y.max()}, noisy_y.min() = {noisy_y.min()}")
        assert_sample_range_compliance(noisy_y, tokenizer)

        if isinstance(init_y, dict):
            # put the noised 'y' back into the dictionary
            init_y['input_ids'] = noisy_y
            noisy_y = init_y

        # for the purpose of learning to denoised masked token, this is often used in masked language model (MLM)
        denoised_sentence, denoised_masked_token_logits, denoised_token_logits = denoiser_model(noisy_y, mlm_mask, target_label, tgt, input_pad_mask)

        # denoised_sentence has a shape of [batch_size, src_sequence_length]
        #print(f"denoised_sentence.shape = {denoised_sentence.shape}")
        #print(f"denoised_masked_token_logits.shape = {denoised_masked_token_logits.shape}")

        if USE_LOGITS_FOR_DENOISING:
            denoised_y = denoised_token_logits
        else:
            denoised_y = denoised_sentence

        #print(f"y.shape = {y.shape} , noisy_y.shape = {noisy_y.shape} , denoised_y.shape = {denoised_y.shape}")

        if not(USE_SMC or USE_MAFBM or USE_MCMC):  # masked language modeling task
            y = denoised_y  # DIRECT denoiser denoising, no need any specific denoising equation
        else:
            # See section 2.4 or Algorithm 1 of [Discrete Flow Matching](http://arxiv.org/abs/2407.15595)
            if USE_LOGITS_FOR_DENOISING:
                if isinstance(noisy_y, dict):
                    noisy_y = noisy_y['input_ids']
                else:
                    noisy_y = noisy_y

                # Get the shapes
                batch_size, sequence_length, vocab_size = denoised_y.shape

                # Create embedding projection that matches vocabulary size
                embedding_proj = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=vocab_size,
                    padding_idx=tokenizer.pad_token_id,
                    device=device
                )

                # Both delta_X_t and u_t are of the shape of [batch_size, sequence_length, vocab_size]
                delta_X_t = embedding_proj(noisy_y.long())
                u_t = denoised_y
                h = walk_step_size

                # Update rule for equation (12): delta + h * velocity
                prob_distribution = delta_X_t + h * u_t
                prob_distribution = F.softmax(prob_distribution, dim=-1)  # Normalize to a valid probability distribution

                # Sampling the next state
                X_t_h = torch.multinomial(prob_distribution.view(-1, vocab_size), 1).view(batch_size, sequence_length)
                y = X_t_h
                #print(y)
                # Output: Tensor of shape [batch_size, sequence_length] representing sampled token indices
            else:
                y = y + sigma_t ** 2 * denoised_y  # Update based on denoising equation in walk-jump

        # Scales to range of [0, tokenizer.vocab_size-1]
        y = y - y.min()
        y = y / y.max()
        y = y * (tokenizer.vocab_size - 1)

        # checks for potential issues
        #print(f"denoised_y.max() = {denoised_y.max()}, denoised_y.min() = {denoised_y.min()}")
        assert_sample_range_compliance(y, tokenizer)

        # we only runs this during inference, we will instead run "_shift_right()" during training
        if mlm_mask is None and tgt is not None:
            # Get the last token from the denoised sequence
            new_token = denoised_y[:, -1].unsqueeze(-1)  # shape [batch_size, 1]
            #print(f"tgt.shape = {tgt.shape} , tgt = {tgt} , new_token.shape = {new_token.shape} , new_token = {new_token}")

            # Step 1: Normalize the model output between (0, 1)
            new_token_min = new_token.min()  # Get the minimum value
            new_token_max = new_token.max()  # Get the maximum value
            normalized_output = (new_token - new_token_min) / (new_token_max - new_token_min + 1e-8)

            # Step 2: Rescale to the desired range (0, vocab_size)
            rescaled_output = normalized_output * tokenizer.vocab_size

            # Ensure values stay within bounds of (0, vocab_size-1)
            new_token = rescaled_output.clamp(0, tokenizer.vocab_size-1)

            # For each subsequent iteration, concatenate the new token to tgt
            tgt = torch.cat((tgt, new_token), dim=1)

            # Check for end token
            if USE_PRETRAINED_T5: #or USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
                # T5 uses a different end-of-sequence token and does not use a separate SEP token
                if (new_token == tokenizer.eos_token_id).all():
                    break

            else:
                # For other models, check for both EOS and SEP tokens
                if hasattr(tokenizer, 'sep_token_id'):
                    if (new_token == tokenizer.sep_token_id).all():
                        break
                elif hasattr(tokenizer, 'eos_token_id'):
                    if (new_token == tokenizer.eos_token_id).all():
                        break
                else:
                    pass  # nothing happens

    denoised_sentence = y

    if isinstance(noisy_y, dict):
        noisy_ids = noisy_y['input_ids']
    else:
        noisy_ids = noisy_y

    if mlm_mask is not None:
        if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
            return noisy_ids, denoised_sentence, denoised_masked_token_logits, denoised_token_logits
        else:
            return noisy_ids, denoised_sentence, denoised_masked_token_logits
    else:
        return noisy_ids, denoised_sentence


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Use this loss function instead of nn.CrossEntropyLoss
smooth_CE_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)


def train_walk_jump(ebm, denoiser, train_loader, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, optimizer_ebm, optimizer_denoiser, scheduler_ebm, scheduler_denoiser, scaler):
    ebm.train()
    denoiser.train()

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    for train_data in train_loader:
        # Clear memory before processing each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):  # Check if MPS backend exists
            torch.mps.empty_cache()

        #if isinstance(train_data, dict):
        if MASK_RATIO != -1:  # will be using data_collator which returns dict format
            input_ids = train_data['input_ids'].to(device)
            target_label = train_data['labels'].to(device)  # this is the target for the masked tokens in which the model should predict to unmask
        else:
            input_ids = train_data['input_ids'].to(device)
            target_label = input_ids.clone()

        if ENABLE_MASK_LEARNING:
            if MASK_RATIO != -1:
                # We are now using data collator class to deal with masking strategy
                mask = train_data['mask_indices'].to(device)  # Get mask from data collator

                # Randomly mask some tokens in the clean train_data
                #mask = torch.rand(train_data.shape).to(device) < MASK_RATIO  # 15% masking probability
                mask = mask * 1  # converts True/False into 1/0
            else:
                # Randomly mask only 1 single token in the clean train_data
                batch_size, seq_len = train_data['attention_mask'].shape
                mask = torch.zeros_like(train_data['attention_mask'], dtype=torch.bool).to(device)

                # Create a boolean mask for non-pad and non-special tokens
                # (True where tokens are real, and False for [PAD], [CLS], [SEP])
                non_special_tokens = (train_data['input_ids'] != pad_token_id) & (train_data['input_ids'] != cls_token_id) & (train_data['input_ids'] != sep_token_id)
                #print(f"non_special_tokens has a shape of {non_special_tokens.shape}")

                # Sum along the sequence dimension to get the actual length of each sequence (excluding special tokens)
                actual_seq_len = non_special_tokens.sum(dim=1)
                #print(f"actual_seq_len has a shape of {actual_seq_len.shape}")

                # For each sequence in the batch, randomly mask one token
                # random indexing starts from 1 since we do not want to index the first [CLS] token in each training sequence
                # Random index per sequence in the batch
                random_indices = torch.stack([torch.randint(1, length.item(), (1,)) for length in actual_seq_len]).squeeze()
                #print(f"random_indices = {random_indices}, random_indices.shape = {random_indices.shape}")
                # Mask the selected tokens at the random indices
                mask[torch.arange(batch_size), random_indices] = 1

                mask = mask * 1  # converts True/False into 1/0
                assert(mask.sum() == batch_size)  # shape : [batch_size, seq_len] , so only 1 masked token for each sequence

            #print(f"mask = {mask}")

            #if not(MASK_RATIO != -1):  # will not be using data_collator which returns dict format
                # Set non-masked positions in target_label to -100 or tokenizer.pad_token_id
                # See the use of ignore_index in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                #target_label[~mask.bool()] = CONSTANTS_VALUE_IGNORE

            input_ids = input_ids.float().to(device)

            if ADD_EXTRA_GAUSSIAN_NOISE:
                noisy_train_data = add_token_noise(input_ids, tokenizer, noise_level=sigma_min).to(device)  # Or your chosen noising function
            else:
                noisy_train_data = input_ids  # for isolating the extra noise added

            if MASK_RATIO != -1:
                if USE_PRETRAINED_T5:
                    # We should not put '<extra_id_0>' for all masked tokens, they should be numbered accordingly as in '<extra_id_*>' to indicate ordering
                    # We had used the data_collator to prepare train_loader, so no need to manually modify
                    masked_train_ids = noisy_train_data.clone()  # we want to denoise noisy data to its clean version
                else:
                    # we are now using data_collator for masking purpose, see _mask_tokens_standard()
                    # We had used the data_collator to prepare train_loader, so no need to manually modify
                    masked_train_ids = noisy_train_data.clone()  # we want to denoise noisy data to its clean version
                    masked_train_ids[mask.bool()] = tokenizer.mask_token_id
            else:
                if USE_PRETRAINED_T5:
                    masked_train_ids = noisy_train_data.clone()  # we want to denoise noisy data to its clean version
                    masked_train_ids[mask.bool()] = tokenizer.convert_tokens_to_ids('<extra_id_0>')
                else:
                    masked_train_ids = noisy_train_data.clone()  # we want to denoise noisy data to its clean version
                    masked_train_ids[mask.bool()] = tokenizer.mask_token_id

            #print(f"Masked train ids = {masked_train_ids}")

        #if MASK_RATIO == 0.00:
            # for testing purpose only
            #assert(torch.equal(train_data, masked_train_ids))

        # ebm model and denoiser model are trained independently, no gradient connections between them
        masked_train_data = {
            'input_ids': masked_train_ids,
            'labels': target_label,
            'attention_mask': train_data['attention_mask'].clone().detach() if 'attention_mask' in train_data else None
        }

        # Train EBM
        optimizer_ebm.zero_grad()

        # Train denoiser
        optimizer_denoiser.zero_grad()

        # Add noise to input data
        noisy_train_ids = train_data['input_ids'].float() + torch.randn_like(train_data['input_ids'].float()) * sigma

        # Scales to range of [0, tokenizer.vocab_size-1]
        noisy_train_ids = noisy_train_ids - noisy_train_ids.min()
        noisy_train_ids = noisy_train_ids / noisy_train_ids.max()
        noisy_train_ids = noisy_train_ids * (tokenizer.vocab_size - 1)

        # checks for potential issues
        assert_sample_range_compliance(noisy_train_ids, tokenizer)

        if USE_MIXED_PRECISION_TRAINING:
            with autocast(device_type=device_str, dtype=torch.float16):
                # Get energy of noisy input
                energy_real = ebm({
                    'input_ids': noisy_train_ids,
                    'labels': target_label,
                    'attention_mask': train_data['attention_mask']
                })
        else:
            # Get energy of noisy input
            energy_real = ebm({
                'input_ids': noisy_train_ids,
                'labels': target_label,
                'attention_mask': train_data['attention_mask']
            })

        #print(f"energy_real has a shape of {energy_real.shape}")
        assert not torch.all(energy_real == 0), "Error: energy_real contains all zeros!"

        # for the purpose of more efficient run for the denoiser model
        input_pad_mask = (input_ids == tokenizer.pad_token_id).to(device)

        if ENABLE_MASK_LEARNING:
            # Generate samples using walk-jump
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                noisy_ids, generated_samples, denoised_masked_token_logits, denoised_token_logits = walk_jump_sampling(masked_train_data, mask, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask, target_label=target_label)
            else:
                noisy_ids, generated_samples, denoised_masked_token_logits = walk_jump_sampling(masked_train_data, mask, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask, target_label=target_label)
        else:
            # Generate samples using walk-jump
            noisy_ids, generated_samples = walk_jump_sampling(train_data, None, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask, target_label=target_label)

        # Analyzing the generated samples during training
        if ENABLE_SAMPLE_ANALYSIS:
            analyze_samples(generated_samples, tokenizer)

        # checks for potential issues
        assert_sample_range_compliance(noisy_ids, tokenizer)

        # EBM module favours lower energy for the real data from the distribution, and higher energy for fake data from the sampling process
        if USE_MIXED_PRECISION_TRAINING:
            with autocast(device_type=device_str, dtype=torch.float16):
                #energy_fake = ebm(generated_samples)
                energy_fake = ebm({
                    'input_ids': noisy_ids, #train_data['input_ids'],
                    'labels': target_label,
                    'attention_mask': train_data['attention_mask']
                })

        else:
            #energy_fake = ebm(generated_samples)
            energy_fake = ebm({
                'input_ids': noisy_ids, #train_data['input_ids'],
                'labels': target_label,
                'attention_mask': train_data['attention_mask']
            })

        #print(f"energy_fake has a shape of {energy_fake.shape}")
        assert not torch.all(energy_fake == 0), "Error: energy_fake contains all zeros!"


        '''
        log(q(x_real) / q(x_fake)) = log( [exp(-E(x_real)) / Z] / [exp(-E(x_fake)) / Z] )
                           = log( exp(-E(x_real)) / exp(-E(x_fake)) )   (Z cancels out)
                           = -E(x_real) + E(x_fake)

        Z cancels out is a simplification based on the assumption that Z is approximately the same when calculated for both real (ground truth) and fake (generated) data.

        To maximize the log-likelihood ratio, we minimize its negative:  -log(q(x_real) / q(x_fake)) = E(x_real) - E(x_fake) = energy_real.mean() - energy_fake.mean()

        However, if we do not assume that Z cancels out due to imperfect MCMC sampling,

        -log(q(x)) = -log(exp(-E(x)) / Z)
           = -log(exp(-E(x))) + log(Z)
           = E(x) + log(Z)

        We've established that -log(q(x)) ≈ E(x) + log_sum_exp(-energy).
        For real data, we can write: -log(q(x_real)) ≈ E(x_real) + log_sum_exp(-energy_real).
        For fake data, we can write: -log(q(x_fake)) ≈ E(x_fake) + log_sum_exp(-energy_fake).

        Constructing the Loss Function:
        Goal: We want to minimize -log(q(x_real)) (make real data probable) and maximize -log(q(x_fake)) (make fake data improbable).

        Using the Approximation: We can approximate this by minimizing E(x_real) + log_sum_exp(-energy_real) and maximizing E(x_fake) + log_sum_exp(-energy_fake).

        Combining Terms: To achieve this with a single loss function, we can take the negative of the term we want to minimize and add it to the term we want to maximize:
        We had done this on -log(q(x)), so directly we have the following contrastive loss:
        loss = [E(x_real) + log_sum_exp(-energy_real)] - [E(x_fake) + log_sum_exp(-energy_fake)]

        Simplifying: We can rearrange this as:
        loss = E(x_real) + log_sum_exp(-energy_real) - E(x_fake) - log_sum_exp(-energy_fake)

        Averaging: In practice, we work with batches of data, so we take the mean over the data points in the batch:
        loss_ebm = E(x_real).mean() + log_sum_exp(-energy_real).mean() - E(x_fake).mean() - log_sum_exp(-energy_fake).mean()

        Theoretical E(x): In the theoretical derivations of EBMs and contrastive divergence, E(x) represents the energy function. Lower energy corresponds to higher probability.
        Code Implementation - ebm() function: In the code, the ebm() function (or EnergyBasedModel) is implemented to compute the negative log-probability of a given input, up to a constant.
        This is because we want to use gradient-based optimization to minimize this value for real data.

        So, what the ebm() model outputs is not directly E(x), but rather something proportional to -log(q(x)), which in turn is approximately E(x) + log(Z) which we have derived in the text above.

        Given that:
        energy_real = ebm(noisy_train_ids)
        energy_fake = ebm(noisy_ids)

        so we have the following final loss_ebm expression:
        loss_ebm = energy_real.mean() + log_sum_exp(-energy_real).mean() - energy_fake.mean() - log_sum_exp(-energy_fake).mean()
        '''

        # Compute EBM loss with contrastive divergence
        #loss_ebm = (energy_real.mean() - energy_fake.mean()) + ebm_energy_regularization_scale * (energy_real ** 2).mean()  # Added L2 regularization

        # Compute EBM loss with contrastive divergence, log-sum-exp trick and offset
        energies = torch.cat([energy_real, energy_fake])  # Concatenate energy_real and energy_fake
        mean_energy = torch.mean(energies)
        #loss_ebm = log_sum_exp(-(energy_real-mean_energy)) - log_sum_exp(-(energy_fake-mean_energy)) + ebm_energy_regularization_scale * (energy_real ** 2).mean()  # Added L2 regularization
        loss_ebm = energy_real.mean() + log_sum_exp(-energy_real).mean() - energy_fake.mean() - log_sum_exp(-energy_fake).mean() + ebm_energy_regularization_scale * (energy_real ** 2).mean()  # Added L2 regularization

        if USE_MIXED_PRECISION_TRAINING and torch.cuda.is_available():
            scaler.scale(loss_ebm).backward(retain_graph=True)
            #nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=10.0)  # Gradient clipping
            #check_for_vanishing_gradients(ebm)  # Check for vanishing gradients
            scaler.step(optimizer_ebm)
            scaler.update()
        else:
            loss_ebm.backward(retain_graph=True)
            #nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=10.0)  # Gradient clipping
            #check_for_vanishing_gradients(ebm)  # Check for vanishing gradients
            optimizer_ebm.step()

        if ENABLE_MASK_LEARNING:
            #print(f"mask = {mask}")
            unmask = 1 - mask  # Inverse of mask
            #print(f"unmask = {unmask}")

            """
            Cross-Entropy Loss:

            1.	Why it’s preferred for masked language modeling:
                Cross-entropy loss is widely used for classification tasks, including masked language modeling. MLM is essentially a classification problem where the model predicts a token from a discrete set of possibilities (the vocabulary) for each masked position. Cross-entropy loss measures how well the predicted probability distribution over the vocabulary matches the true distribution (typically a one-hot vector where the correct token has a probability of 1).
            2.	How it works:
                Cross-entropy loss penalizes the model when the predicted probability of the correct token is low. It compares the predicted probability distribution with the true distribution and calculates the logarithmic loss, which is then averaged over all predictions. This loss function is effective for tasks where the outputs are discrete and categorical, such as predicting words or tokens in natural language processing tasks.

            MSE Loss:

            1.	Why it’s less suitable:
                MSE loss is more appropriate for regression tasks, where the goal is to predict continuous values. In the context of language modeling, using MSE would treat the token IDs as continuous values and penalize the squared differences between predicted and true token IDs. This approach doesn’t align well with the nature of language, where the relationship between token IDs is not linear or continuous.
            2.	Why it’s inappropriate for MLM:
                Token IDs in a vocabulary do not have a meaningful numerical relationship to each other (e.g., the token ID for “apple” being 103 and “banana” being 104 does not mean they are numerically close in meaning). MSE loss would incorrectly interpret these IDs as continuous variables and could lead to suboptimal training because it doesn’t capture the categorical nature of the task.
            """
            # Compute the loss for unmasked positions
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                # Compute CrossEntropy loss directly between denoised_token_logits and the correct tokens
                #print(f"denoised_token_logits.shape = {denoised_token_logits.shape}, train_data.shape = {train_data.shape}")
                target = train_data['input_ids'].long()
                #print(f"target = {analyze_samples(target, tokenizer, skip_special_tokens=True)}")
                loss_denoiser = nn.CrossEntropyLoss(ignore_index=-100)(denoised_token_logits.view(-1, denoised_token_logits.size(-1)), target.view(-1))
            else:
                # Compute the loss for masked positions
                loss_masked = 0.00
                if MASK_RATIO == -1:  # loss computation in the case of a single masked token
                    # Assuming that 'mask' is a tensor of shape [batch_size, seq_len] where True indicates a masked position
                    train_data_correct = train_data[mask.bool()].long()

                    print(f"generated_samples[mask.bool()] = {analyze_samples(generated_samples[mask.bool()], tokenizer, skip_special_tokens=False)}, shape: {generated_samples[mask.bool()].shape}, dtype: {generated_samples[mask.bool()].dtype}")
                    #print(f"before tokenized, train_data_correct = {train_data_correct}, shape: {train_data_correct.shape}, dtype: {train_data_correct.dtype}")
                    #print(f"after tokenized, train_data_correct = {analyze_samples(train_data_correct, tokenizer, skip_special_tokens=True)}, shape: {train_data_correct.shape}, dtype: {train_data_correct.dtype}")
                    #print(f"for checking, train_data[:, 1] = {train_data[:, 1]}, shape: {train_data[:, 1].shape}, dtype: {train_data[:, 1].dtype}")  # Ensure this is consistent

                    # Compute CrossEntropy loss directly between denoised_masked_token_logits and the correct tokens
                    loss_masked = nn.CrossEntropyLoss(ignore_index=-100)(denoised_masked_token_logits, train_data_correct)
                    #loss_masked = smooth_CE_loss_fn(denoised_masked_token_logits, train_data_correct)

                    # Both generated_samples and train_data are of tokenized embedding nature, hence use MSELoss() here for now
                    #loss_masked = nn.MSELoss()(generated_samples[mask.bool()], train_data[mask.bool()])
                    #loss_masked = nn.CrossEntropyLoss()(generated_samples[mask.bool()], train_data[mask.bool()])
                else:
                    if USE_PRETRAINED_T5:
                        train_data_correct = train_data.long()
                        # we do not run CE loss for computing loss_masked because DataCollatorForSpanCorruption does not yet provide masked positions directly
                    else:
                        train_data_correct = train_data[mask.bool()].long()

                        # BERT_MLM model outputs a shape of [batch_size, sequence_length, vocab_size] which is feasible for computing CE loss
                        if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
                            # Compute CrossEntropy loss directly between denoised_token_logits and the correct tokens
                            loss_masked = nn.CrossEntropyLoss(ignore_index=-100)(denoised_token_logits.view(-1, denoised_token_logits.size(-1)), train_data_correct)

                if not USE_PRETRAINED_T5:
                    #print(f"generated_samples[mask.bool()] = {generated_samples[mask.bool()]}")
                    mask_token_penalty = (generated_samples[mask.bool()].int() == tokenizer.mask_token_id).sum().item()
                    sep_token_penalty = (generated_samples[mask.bool()].int() == tokenizer.sep_token_id).sum().item()
                    #print(f"mask_token_penalty = {mask_token_penalty}, sep_token_penalty = {sep_token_penalty}")
                    #loss_masked = loss_masked + mask_token_penalty * mask_token_penalty_weight + sep_token_penalty * sep_token_penalty_weight  # Adjust the weight as needed

                loss_unmasked = nn.MSELoss()(generated_samples[unmask.bool()], train_data[unmask.bool()])

                if USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
                    alpha = 1.0  # Focus on optimizing loss_masked
                else:
                    alpha = 0.5  # Adjust alpha as needed

                print(f"loss_masked = {loss_masked}")
                print(f"loss_unmasked = {loss_unmasked}")
                loss_denoiser = alpha * loss_masked + (1 - alpha) * loss_unmasked

            #print(f"loss_denoiser = {loss_denoiser}")

            # Identify the range of `[unused]` tokens
            unused_token_min_id = 1  # [unused0]
            unused_token_max_id = 107  # [unused102]

            # Create a mask to check for `[unused]` tokens in the generated samples
            unused_token_mask = (generated_samples >= unused_token_min_id) & (generated_samples <= unused_token_max_id)
            unused_token_penalty = unused_token_mask.sum().item()
            #print(f"unused_token_penalty = {unused_token_penalty}")

            # Penalty for predicting [unused] tokens
            loss_denoiser = loss_denoiser + unused_token_penalty * unused_token_penalty_weight
        else:
            loss_denoiser = nn.MSELoss()(generated_samples, train_data)

        loss_denoiser.backward(retain_graph=True)
        #nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=10.0)  # Gradient clipping
        #check_for_vanishing_gradients(denoiser)  # Check for vanishing gradients
        optimizer_denoiser.step()

        # Update learning rate
        scheduler_ebm.step()
        scheduler_denoiser.step()

        # Explicitly delete tensors to save memory across training epochs
        del train_data
        del noisy_train_data
        del masked_train_data
        del mask
        del unmask
        del target
        del input_ids
        del target_label
        #del non_special_tokens
        #del random_indices
        del input_pad_mask
        del energy_real
        del energy_fake
        del energies
        del mean_energy
        del unused_token_mask

        if ENABLE_MASK_LEARNING:
            # Generate samples using walk-jump
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                del generated_samples
                del denoised_masked_token_logits
                del denoised_token_logits
            else:
                del generated_samples
                del denoised_masked_token_logits
        else:
            # Generate samples using walk-jump
            del generated_samples

    return loss_ebm.item(), loss_denoiser.item()


# Initialize models
#ebm = EnergyBasedModel(input_dim, hidden_dim).to(device)
if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
    ebm = BertDenoiser(model_dim).to(device)
elif USE_PRETRAINED_T5:
    #ebm = T5Denoiser().to(device)  # we do not use T5 due to RAM memory restriction
    ebm = TransformerDenoiser(input_dim, model_dim_ebm, num_layers_ebm, num_heads_ebm, sigma_max).to(device)
else:
    ebm = TransformerDenoiser(input_dim, model_dim_ebm, num_layers_ebm, num_heads_ebm, sigma_max).to(device)

if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
    denoiser = BertDenoiser(model_dim).to(device)
elif USE_PRETRAINED_T5:
    denoiser = T5Denoiser(model_dim).to(device)
else:
    denoiser = TransformerDenoiser(input_dim, model_dim, num_layers, num_heads, sigma_max).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

# the following init_weights are not used because it hurts `loss_unmasked` quite a lot
#ebm.apply(init_weights)
#denoiser.apply(init_weights)

# Load the AG News dataset
dataset = load_dataset('ag_news')

# Load the IMDb dataset
#dataset = load_dataset('imdb')

# Select a small subset of the training set for testing
NUM_OF_SMALL_SUBSET_OF_TRAIN_SET = 6000
train_dataset = dataset['train'].select(range(NUM_OF_SMALL_SUBSET_OF_TRAIN_SET))
if TEST_OVERFIT and not INFERENCE_ONLY:
    # In the following 'for' loop, train_dataset variable seems to be of immutable, hence no data overwriting actually happenedd
    #for i in range(NUM_OF_SMALL_SUBSET_OF_TRAIN_SET):
    #    train_dataset[i]['text'] = train_dataset[0]['text']

    # Set the text of all the entries in the training set to be the same as train_dataset[0]['text']
    new_text = train_dataset[0]['text']
    train_dataset = train_dataset.map(lambda x: {'text': new_text})

dataset['train'] = train_dataset
print(f"an example training dataset at index 0: {train_dataset[0]['text']}")
print(f"an example training dataset at index 300: {train_dataset[300]['text']}")


def add_token_noise(input_ids, tokenizer, noise_level=0.05, noise_fraction=0.1):
    """
    Add noise to tokenized input_ids by randomly replacing tokens based on a noise level,
    but without modifying special tokens and filtering them out during noise application.

    Args:
        input_ids (List[int]): The input token ids.
        tokenizer: Tokenizer with methods to convert ids to tokens and tokens to ids.
        noise_level (float): The noise level that controls the magnitude of the noise.
        noise_fraction (float): The fraction of tokens to apply noise to.

    Returns:
        torch.Tensor: The noisy token ids as a tensor with special tokens re-inserted.
    """
    special_token_ids = set(tokenizer.all_special_ids)  # Get the set of special token IDs
    #print(f"All special token IDs: {tokenizer.all_special_ids}")
    #print(f"Pad token ID: {tokenizer.pad_token_id}")
    #print(f"special_token_ids: {special_token_ids}")

    noisy_input_ids = []

    for ids in input_ids:
        # Ensure the output length matches the input length
        original_length = len(ids)
        ids = ids.tolist()  # Convert tensor to a list of integers
        #print(f"ids: {ids}")
        #print(f"original_length: {original_length}")

        # Store original positions of special tokens and filter them out
        special_tokens_positions = {i: token_id for i, token_id in enumerate(ids) if token_id in special_token_ids}
        filtered_ids = [token_id for token_id in ids if token_id not in special_token_ids]

        #print(f"Input IDs before filtering: {ids}")
        #print(f"Special tokens: {special_tokens_positions}")
        #print(f"Filtered IDs: {filtered_ids}")

        # Generate noise for the filtered non-special tokens
        noise = torch.randn(len(filtered_ids))  # Generate noise for each non-special token
        probs = torch.sigmoid(noise * noise_level)

        # Create a counter of tokens in the tokenizer's vocabulary
        vocab_size = len(tokenizer)
        token_counter = Counter(range(vocab_size))

        # Create a cumulative distribution for token sampling
        cum_dist = []
        total = 0
        for token_id, count in token_counter.items():
            total += count
            cum_dist.append(total)
        cum_dist = [x / total for x in cum_dist]

        # Function to sample a token based on the cumulative distribution and a random value
        def sample_token(rand_val):
            for i, val in enumerate(cum_dist):
                if rand_val <= val:
                    return list(token_counter.keys())[i]
            return list(token_counter.keys())[-1]

        # Apply noise to filtered non-special tokens
        noisy_filtered_ids = [
            sample_token(probs[i].item()) if probs[i].item() > random.random() and random.random() < noise_fraction
            else token_id
            for i, token_id in enumerate(filtered_ids)
        ]

        # Re-insert special tokens into their original positions
        noisy_ids = []
        filtered_idx = 0
        for i in range(len(ids)):
            if i in special_tokens_positions:
                noisy_ids.append(special_tokens_positions[i])  # Add special token
            else:
                noisy_ids.append(noisy_filtered_ids[filtered_idx])  # Add noisy/non-noisy token
                filtered_idx += 1

        # Ensure the length of noisy_ids matches the original length to prevent extra tokens
        noisy_ids = noisy_ids[:original_length]
        noisy_input_ids.append(noisy_ids)

    # Convert noisy_input_ids (a list of lists) back into a tensor
    return torch.tensor(noisy_input_ids)


def add_character_noise(text, noise_level):
    # Sample noise from a normal distribution
    noise = torch.randn(len(text))

    # Convert noise to probabilities between 0 and 1
    probs = torch.sigmoid(noise * noise_level)

    # Create a counter of characters in the text
    char_counter = Counter(string.ascii_letters + string.digits + string.punctuation)

    # Create a cumulative distribution from the counter
    cum_dist = []
    total = 0
    for char, count in char_counter.items():
        total += count
        cum_dist.append(total)
    cum_dist = [x / total for x in cum_dist]

    # Function to sample a character based on the cumulative distribution and a random value
    def sample_char(rand_val):
        for i, val in enumerate(cum_dist):
            if rand_val <= val:
                return list(char_counter.keys())[i]
        return list(char_counter.keys())[-1]  # Just in case of rounding errors

    # Replace characters based on probability
    noisy_text = "".join([
        sample_char(probs[i].item()) if probs[i].item() > random.random() else char
        for i, char in enumerate(text)
    ])
    return noisy_text


def preprocess_function(examples):
    #noisy_text = add_character_noise(examples, noise_level=sigma_max)  # Or your chosen noising function
    tokenized_inputs = tokenizer_function(examples, tokenizer)
    #tokenized_inputs = tokenizer_function(noisy_text, tokenizer)
    return tokenized_inputs


#dataset = dataset.map(preprocess_function, batched=True)
#dataset.set_format(type='torch', columns=['input_ids'])

encoded_inputs_file = 'encoded_inputs_walk_jump.pt'

if os.path.exists(encoded_inputs_file):
    print("Loading pre-tokenized data...")
    encoded_inputs = torch.load(encoded_inputs_file, weights_only=True)
else:
    # Process data
    print("Tokenizing data now ...")
    processed_inputs = [preprocess_function(entry['text'])
                      for entry in dataset['train']]

    # Concatenate tensors for each key
    encoded_inputs = {
        'input_ids': torch.cat([x['input_ids'] for x in processed_inputs], dim=0),
        'attention_mask': torch.cat([x['attention_mask'] for x in processed_inputs], dim=0)
    }
    #encoded_inputs = torch.cat(encoded_inputs, dim=0)

    torch.save(encoded_inputs, encoded_inputs_file)
    print("Finished tokenizing data !!!")

class SomeDataset(Dataset):
    def __init__(self, data):
        #self.data = data
        self.input_ids = data['input_ids']  # Shape: [total_size, sequence_length]
        self.attention_mask = data['attention_mask']  # Shape: [total_size, sequence_length]

    def __len__(self):
        #return len(self.data)
        return len(self.input_ids)

    def __getitem__(self, idx):
        #return self.data[idx]
        return {
            'input_ids': self.input_ids[idx],  # Shape: [sequence_length]
            'attention_mask': self.attention_mask[idx]  # Shape: [sequence_length]
        }


# Split the data into train and validation sets
total_size = len(encoded_inputs['input_ids'])
train_size = int(total_size * 0.8)
print(f"total_size = {total_size}")

# Split each tensor in the dictionary
train_data = {
    'input_ids': encoded_inputs['input_ids'][:train_size],
    'attention_mask': encoded_inputs['attention_mask'][:train_size]
}
val_data = {
    'input_ids': encoded_inputs['input_ids'][train_size:],
    'attention_mask': encoded_inputs['attention_mask'][train_size:]
}
#train_data = encoded_inputs[:train_size]
#val_data = encoded_inputs[train_size:]

train_dataset = SomeDataset(train_data)
val_dataset = SomeDataset(val_data)


# Use the data_collator to prepare inputs and labels for train_loader and val_loader
data_collator = DataCollatorForSpanCorruption(
    tokenizer=tokenizer,
    mlm_probability=MASK_RATIO,
    mean_noise_span_length=3,
    input_length=input_dim
)

# Create a DataLoader for batch processing
# Now we can use data_loader in the training loop
if MASK_RATIO != -1:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Dummy data loader
if USE_DUMMY_TRAINING_DATA:
    train_loader = DataLoader(torch.randn(100, input_dim), batch_size=batch_size, shuffle=True)

# Define noise schedule
noise_schedule = torch.arange(sigma_min, sigma_max, 0.1).to(device)
print(f"noise_schedule = {noise_schedule}")


def validate_walk_jump(ebm, denoiser, val_loader, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule):
    ebm.eval()
    denoiser.eval()

    val_ebm_losses = []
    val_denoiser_losses = []

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    with torch.no_grad():
        for val_data in val_loader:
            # Clear memory before processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):  # Check if MPS backend exists
                torch.mps.empty_cache()

            #if isinstance(train_data, dict):
            if MASK_RATIO != -1:  # will be using data_collator which returns dict format
                input_ids = val_data['input_ids'].to(device)
                target_label = val_data['labels'].to(device)  # this is the target for the masked tokens in which the model should predict to unmask
            else:
                input_ids = val_data['input_ids'].to(device)
                target_label = input_ids.clone()

            if ENABLE_MASK_LEARNING:
                if MASK_RATIO != -1:
                    # We are now using data collator class to deal with masking strategy
                    mask = val_data['mask_indices'].to(device)  # Get mask from data collator

                    # Randomly mask some tokens in the clean val_data
                    #mask = torch.rand(val_data.shape).to(device) < MASK_RATIO  # 15% masking probability
                    mask = mask * 1  # converts True/False into 1/0
                else:
                    # Randomly mask only 1 single token in the clean val_data
                    batch_size, seq_len = val_data['attention_mask'].shape
                    mask = torch.zeros_like(val_data['attention_mask'], dtype=torch.bool).to(device)

                    # Create a boolean mask for non-pad and non-special tokens
                    # (True where tokens are real, and False for [PAD], [CLS], [SEP])
                    non_special_tokens = (val_data['input_ids'] != pad_token_id) & (val_data['input_ids'] != cls_token_id) & (val_data['input_ids'] != sep_token_id)
                    #print(f"non_special_tokens has a shape of {non_special_tokens.shape}")

                    # Sum along the sequence dimension to get the actual length of each sequence (excluding special tokens)
                    actual_seq_len = non_special_tokens.sum(dim=1)
                    #print(f"actual_seq_len has a shape of {actual_seq_len.shape}")

                    # For each sequence in the batch, randomly mask one token
                    # random indexing starts from 1 since we do not want to index the first [CLS] token in each training sequence
                    # Random index per sequence in the batch
                    random_indices = torch.stack([torch.randint(1, length.item(), (1,)) for length in actual_seq_len]).squeeze()
                    #print(f"random_indices = {random_indices}, random_indices.shape = {random_indices.shape}")
                    # Mask the selected tokens at the random indices
                    mask[torch.arange(batch_size), random_indices] = 1

                    mask = mask * 1  # converts True/False into 1/0
                    assert(mask.sum() == batch_size)  # shape : [batch_size, seq_len] , so only 1 masked token for each sequence


                #if not(MASK_RATIO != -1):  # will not be using data_collator which returns dict format
                    # Set non-masked positions in target_label to -100 or tokenizer.pad_token_id
                    # See the use of ignore_index in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    #target_label[~mask.bool()] = CONSTANTS_VALUE_IGNORE

                input_ids = input_ids.float().to(device)

                if ADD_EXTRA_GAUSSIAN_NOISE:
                    noisy_val_data = add_token_noise(input_ids, tokenizer, noise_level=sigma_min).to(device)  # Or your chosen noising function
                else:
                    noisy_val_data = input_ids  # for isolating the extra noise added

                if MASK_RATIO != -1:
                    if USE_PRETRAINED_T5:
                        # We should not put '<extra_id_0>' for all masked tokens, they should be numbered accordingly as in '<extra_id_*>' to indicate ordering
                        # We had used the data_collator to prepare val_loader, so no need to manually modify
                        masked_val_ids = noisy_val_data.clone()  # we want to denoise noisy data to its clean version
                    else:
                        # we are now using data_collator for masking purpose, see _mask_tokens_standard()
                        # We had used the data_collator to prepare val_loader, so no need to manually modify
                        masked_val_ids = noisy_val_data.clone()  # we want to denoise noisy data to its clean version
                        masked_val_ids[mask.bool()] = tokenizer.mask_token_id
                else:
                    if USE_PRETRAINED_T5:
                        masked_val_ids = noisy_val_data.clone()  # we want to denoise noisy data to its clean version
                        masked_val_ids[mask.bool()] = tokenizer.convert_tokens_to_ids('<extra_id_0>')
                    else:
                        masked_val_ids = noisy_val_data.clone()  # we want to denoise noisy data to its clean version
                        masked_val_ids[mask.bool()] = tokenizer.mask_token_id

                #print(f"Masked validation ids = {masked_val_ids}")

            # ebm model and denoiser model are trained independently, no gradient connections between them
            masked_val_data = {
                'input_ids': masked_val_ids,
                'labels': target_label,
                'attention_mask': val_data['attention_mask']
            }

            # Add noise to input data
            noisy_val_ids = val_data['input_ids'].float() + torch.randn_like(val_data['input_ids'].float()) * sigma

            # Scales to range of [0, tokenizer.vocab_size-1]
            noisy_val_ids = noisy_val_ids - noisy_val_ids.min()
            noisy_val_ids = noisy_val_ids / noisy_val_ids.max()
            noisy_val_ids = noisy_val_ids * (tokenizer.vocab_size - 1)

            # checks for potential issues
            assert_sample_range_compliance(noisy_val_ids, tokenizer)

            # Get energy of noisy input
            energy_real = ebm({
                'input_ids': noisy_val_ids,
                'labels': target_label,
                'attention_mask': val_data['attention_mask']
            })

            # for the purpose of more efficient run for the denoiser model
            input_pad_mask = (val_data['input_ids'] == tokenizer.pad_token_id).to(device)

            if ENABLE_MASK_LEARNING:
                # Generate samples using walk-jump
                if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                    noisy_ids, generated_samples, denoised_masked_token_logits, denoised_token_logits = walk_jump_sampling(masked_val_data, mask, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask, target_label=target_label)
                else:
                    noisy_ids, generated_samples, denoised_masked_token_logits = walk_jump_sampling(masked_val_data, mask, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask, target_label=target_label)
            else:
                # Generate samples using walk-jump
                noisy_ids, generated_samples = walk_jump_sampling(val_data, None, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask, target_label=target_label)

            # Analyzing the generated samples during validation
            if ENABLE_SAMPLE_ANALYSIS:
                analyze_samples(generated_samples, tokenizer)

            #energy_fake = ebm(generated_samples)
            energy_fake = ebm({
                'input_ids': noisy_ids, #val_data['input_ids'],
                'labels': target_label,
                'attention_mask': val_data['attention_mask']
            })

            assert not torch.all(energy_fake == 0), "Error: energy_fake contains all zeros!"

            # Compute EBM loss with contrastive divergence
            #val_ebm_loss = (energy_real.mean() - energy_fake.mean()) + ebm_energy_regularization_scale * (energy_real ** 2).mean()  # Added L2 regularization

            # Compute EBM loss with contrastive divergence, log-sum-exp trick and offset
            energies = torch.cat([energy_real, energy_fake])
            mean_energy = torch.mean(energies)
            #val_ebm_loss = log_sum_exp(-(energy_real-mean_energy)) - log_sum_exp(-(energy_fake-mean_energy)) + ebm_energy_regularization_scale * (energy_real ** 2).mean()  # Added L2 regularization
            val_ebm_loss = energy_real.mean() + log_sum_exp(-energy_real).mean() - energy_fake.mean() - log_sum_exp(-energy_fake).mean() + ebm_energy_regularization_scale * (energy_real ** 2).mean()  # Added L2 regularization

            if ENABLE_MASK_LEARNING:
                #print(f"mask = {mask}")
                unmask = 1 - mask  # Inverse of mask

                # Compute the loss for unmasked positions
                if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                    # Compute CrossEntropy loss directly between denoised_token_logits and the correct tokens
                    #print(f"denoised_token_logits.shape = {denoised_token_logits.shape}, val_data.shape = {val_data.shape}")
                    target = val_data['input_ids'].long()
                    val_denoiser_loss = nn.CrossEntropyLoss(ignore_index=-100)(denoised_token_logits.view(-1, denoised_token_logits.size(-1)), target.view(-1))
                else:
                    # Compute the loss for masked positions
                    loss_masked = 0.00
                    if MASK_RATIO == -1:  # loss computation in the case of a single masked token
                        # Assuming that 'mask' is a tensor of shape [batch_size, seq_len] where True indicates a masked position
                        val_data_correct = val_data[mask.bool()].long()

                        #print(f"denoised_masked_token_logits shape: {denoised_masked_token_logits.shape}, dtype: {denoised_masked_token_logits.dtype}")
                        #print(f"val_data_correct shape: {val_data_correct.shape}, dtype: {val_data_correct.dtype}")

                        # Compute CrossEntropy loss directly between denoised_masked_token_logits and the correct tokens
                        loss_masked = nn.CrossEntropyLoss(ignore_index=-100)(denoised_masked_token_logits, val_data_correct)
                        #loss_masked = smooth_CE_loss_fn(denoised_masked_token_logits, val_data_correct)

                        # Both generated_samples and val_data are of tokenized embedding nature, hence use MSELoss() here for now
                        #loss_masked = nn.MSELoss()(generated_samples[mask.bool()], val_data[mask.bool()])
                        #loss_masked = nn.CrossEntropyLoss()(generated_samples[mask.bool()], val_data[mask.bool()])
                    else:
                        if USE_PRETRAINED_T5:
                            val_data_correct = val_data.long()
                            # we do not run CE loss for computing loss_masked because DataCollatorForSpanCorruption does not yet provide masked positions directly
                        else:
                            val_data_correct = val_data[mask.bool()].long()

                            # BERT_MLM model outputs a shape of [batch_size, sequence_length, vocab_size] which is feasible for computing CE loss
                            if USE_PRETRAINED_BERT or USE_PRETRAINED_BERT_MLM:
                                # Compute CrossEntropy loss directly between denoised_token_logits and the correct tokens
                                loss_masked = nn.CrossEntropyLoss(ignore_index=-100)(denoised_token_logits.view(-1, denoised_token_logits.size(-1)), val_data_correct)

                    loss_unmasked = nn.MSELoss()(generated_samples[unmask.bool()], val_data[unmask.bool()])

                    if USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER:
                        alpha = 1.0  # Focus on optimizing loss_masked
                    else:
                        alpha = 0.5  # Adjust alpha as needed

                    print(f"loss_masked = {loss_masked}")
                    print(f"loss_unmasked = {loss_unmasked}")
                    val_denoiser_loss = alpha * loss_masked + (1 - alpha) * loss_unmasked
            else:
                val_denoiser_loss = nn.MSELoss()(generated_samples, val_data)

            #print(f"val_denoiser_loss = {val_denoiser_loss}")

            val_ebm_losses.append(val_ebm_loss.item())
            val_denoiser_losses.append(val_denoiser_loss.item())

        '''
        # Explicitly delete tensors to save memory across training epochs
        del val_data
        del noisy_val_data
        del masked_val_data
        del mask
        del unmask
        del target
        del input_ids
        del target_label
        #del non_special_tokens
        #del random_indices
        del input_pad_mask
        del energy_real
        del energy_fake
        del energies
        del mean_energy
        #del unused_token_mask

        if ENABLE_MASK_LEARNING:
            # Generate samples using walk-jump
            if USE_LOGITS_FOR_THE_ENTIRE_SENTENCE:  # denoised_token_logits will have a shape of [batch_size, sequence_length, vocab_size]
                del generated_samples
                del denoised_masked_token_logits
                del denoised_token_logits
            else:
                del generated_samples
                del denoised_masked_token_logits
        else:
            # Generate samples using walk-jump
            del generated_samples
        '''

    return numpy.mean(val_ebm_losses), numpy.mean(val_denoiser_losses)


# Top-p sampling selects the smallest possible set of tokens whose cumulative probability
# exceeds a threshold p. This allows for more dynamic and contextual selection of tokens
# based on their probabilities.
def top_p_sampling(probabilities, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    threshold_index = (cumulative_probs > p).nonzero(as_tuple=True)[0][0]
    top_p_probs = sorted_probs[:threshold_index+1]
    top_p_indices = sorted_indices[:threshold_index+1]
    return top_p_probs, top_p_indices


# Temperature-based sampling introduces a temperature parameter that controls the randomness
# of the sampling process. Higher temperatures result in more diverse and random samples,
# while lower temperatures produce more deterministic and conservative samples.
def temperature_sampling(probabilities, temperature=1.0):
    tempered_probs = torch.pow(probabilities, 1.0 / temperature)
    tempered_probs /= torch.sum(tempered_probs, dim=-1, keepdim=True)
    # Flatten the tempered probabilities
    tempered_probs = tempered_probs.view(-1)
    sampled_token_index = torch.multinomial(tempered_probs, num_samples=1)
    sampled_token_id = sampled_token_index.item()
    return sampled_token_index, sampled_token_id


# Get all special token IDs
special_token_ids = tokenizer.all_special_ids

# Get all token IDs that are labeled as "unused" in the vocabulary
range_of_unused_token_ids = 1000
unused_token_ids = [tokenizer.convert_tokens_to_ids(f'[unused{i}]') for i in range(range_of_unused_token_ids)]

# Create a set of all token indices
all_token_indices = set(range(tokenizer.vocab_size))

# Subtract the special token indices and unused token indices from the full range
valid_token_indices = sorted(all_token_indices - set(special_token_ids) - set(unused_token_ids))

# Further filter out subword tokens (those that start with '##') due to WordPiece or Byte-Pair Encoding (BPE) scheme in the tokenizer
filtered_valid_token_indices = [idx for idx in valid_token_indices if not tokenizer.convert_ids_to_tokens(idx).startswith('##')]

# Function to generate a random non-special token ID
def generate_random_non_special_token():
    return random.choice(filtered_valid_token_indices)


def infer_walk_jump(input_text, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, max_length=20, top_k=5):
    ebm.eval()
    denoiser.eval()

    # for sliding window operation
    window_length = len(input_text.split())
    print(f"window_length = {window_length}")

    with torch.no_grad():
        print(f"input_text = {input_text}")
        tokenized_input = tokenizer_function(input_text, tokenizer)
        input_ids = tokenized_input['input_ids'].float().to(device)
        print(f"before add_token_noise, input_ids = {analyze_samples(input_ids, tokenizer, skip_special_tokens=True, num_samples=1)}")
        if ADD_EXTRA_GAUSSIAN_NOISE:
            noisy_input_ids = add_token_noise(input_ids, tokenizer, noise_level=sigma_min).to(device)  # Or your chosen noising function
        else:
            noisy_input_ids = input_ids
        print(f"after add_token_noise, noisy_input_ids = {analyze_samples(noisy_input_ids, tokenizer, skip_special_tokens=True, num_samples=1)}")

        current_ids = noisy_input_ids.clone()
        generated_texts = []

        if not GENERATES_OUTPUT_OF_VARYING_LENGTH:
            # we only run the following overlapping rolling diffusion loop ONCE, hence we directly get the output from the first iteration
            max_length = 1

        for current_index in range(max_length):
            print(f"current_index = {current_index}")
            print(f"current_ids has shape of {current_ids.shape}")
            print(f"current_ids = {current_ids}")

            if GENERATES_OUTPUT_OF_VARYING_LENGTH:
                # sliding window diffusion as in [rolling diffusion models](http://arxiv.org/abs/2402.09470)
                # sliding window concept is primarily designed for continuous domains like images or videos,
                # might not directly translate well to NLP tasks. In the image or video domain, frames or pixels
                # can have strong local correlations, making interpolation and context windows effective.
                # However, in NLP, each word or token is discrete and often depends on long-range dependencies
                # that are not easily captured with a simple sliding window.
                #generated_ids = current_ids[:, window_length-1]
                #generated_ids = current_ids[:, -1]
                #generated_ids = tokenizer.mask_token_id
                #generated_ids = generate_random_non_special_token()
                #generated_ids = torch.tensor(generated_ids, dtype=torch.float).unsqueeze(0)
                #print(f"generated_ids = {generated_ids}")
                #print(f"generated_ids is of type {type(generated_ids)}")
                #generated_text = analyze_samples(generated_ids, tokenizer, skip_special_tokens=True, num_samples=1)
                #print(f"generated_text = {generated_text}")
                #print(f"generated_text is of type {type(generated_text)}")

                if ENABLE_MASK_LEARNING:
                    # Create a copy of current_ids and replace the token at the desired position with the mask_token_id
                    masked_ids = current_ids.clone()

                    mask_position = len(input_text.split()) + current_index + 2  # plus 2 because of the B.O.S. token and the newly generated next token
                    sep_position = mask_position + 1

                    print(f"mask_position = {mask_position} , mask_token_id = {tokenizer.mask_token_id}")
                    print(f"sep_position = {sep_position} , sep_token_id = {tokenizer.sep_token_id}")

                    print(f"Before modification, masked_ids = {masked_ids}")

                    masked_ids[0, mask_position] = tokenizer.mask_token_id

                    # Print the specific index to see if it was updated
                    print(f"After modification, token at mask_position: {masked_ids[0, mask_position]}")

                    masked_ids[0, sep_position] = tokenizer.sep_token_id  # indication of sentence phrase partial end-separation

                    print(f"After modification, masked_ids = {masked_ids}")

                    # for the purpose of more efficient run for the denoiser model
                    input_pad_mask = (masked_ids == tokenizer.pad_token_id).to(device)

                    # Pass the masked sequence to the walk-jump model for processing and denoising
                    noisy_ids, generated_samples = walk_jump_sampling(masked_ids, None, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask)
                else:
                    # for the purpose of more efficient run for the denoiser model
                    input_pad_mask = (current_ids == tokenizer.pad_token_id).to(device)

                    # Pass the sequence to the walk-jump model for processing and denoising
                    noisy_ids, generated_samples = walk_jump_sampling(current_ids, None, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask)

                print(f"generated_samples has shape of {generated_samples.shape}")

                # Extract the next predicted token
                generated_text = analyze_samples(generated_samples, tokenizer, skip_special_tokens=True, num_samples=1)
                next_token = generated_text[0].split()[-1]
                print(f"next_token = {next_token}")

                # Append the predicted token to the generated_texts list
                generated_texts.append(next_token)

                # Update current_ids with the predicted token
                current_ids = generated_samples

                # Since current_ids contains some noising and denoising artifact,
                # there is possibility of token switch for the original input_text fed into the model
                decoded_current_ids = analyze_samples(current_ids, tokenizer, skip_special_tokens=True, num_samples=1)
                print(f"decoded_current_ids = {decoded_current_ids}")
                print(f"decoded_current_ids is of type {type(decoded_current_ids)}")
                decoded_current_ids = str(decoded_current_ids[0]).split()
                decoded_current_ids[:len(input_text.split())] = input_text.split()
                print(f"decoded_current_ids after 1st clean-up = {decoded_current_ids}")

                # Remove the first word token
                #decoded_current_ids = str(decoded_current_ids[0]).split()[1:]
                # Sometimes the model will combine punctuation mark into part of words,
                # hence the above operation will remove the first word token together with the punctuation mark
                #if len(decoded_current_ids) < window_length-1:
                    # add back the punctuation mark, and it is usually period mark
                    #decoded_current_ids = list('.') + decoded_current_ids
                #print(f"decoded_current_ids after removing first token = {decoded_current_ids}")
                #print(f"decoded_current_ids after removing first token is of type {type(decoded_current_ids)}")

                # appends the next predicted token
                # Use Entire History, Maintain the entire sequence generated so far as the context for generating the next token.
                # This avoids truncating important context, which is crucial in NLP.
                if current_index > 0:
                    decoded_current_ids[len(input_text.split()):] = generated_texts
                print(f"decoded_current_ids after 2nd clean-up = {decoded_current_ids}")
                print(f"len(decoded_current_ids) = {len(decoded_current_ids)}")

                new_input = decoded_current_ids

                # converts to string
                new_input = ' '.join(str(x) for x in new_input)
                print(f"new_input = {new_input}")

                # prepares the next input iteration for the walk-jump model
                current_ids_retokenized = tokenizer_function(new_input, tokenizer)
                current_ids = current_ids_retokenized['input_ids'].float().to(device)

                # If EOS token is generated, stop generation
                if next_token == tokenizer.eos_token:
                    break
            else:
                # for the purpose of more efficient run for the denoiser model
                input_pad_mask = (current_ids == tokenizer.pad_token_id).to(device)

                # Pass the sequence to the walk-jump model for processing and denoising
                noisy_ids, generated_samples = walk_jump_sampling(current_ids, None, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, input_pad_mask=input_pad_mask)
                print(f"generated_samples has shape of {generated_samples.shape}")

                generated_texts = analyze_samples(generated_samples, tokenizer, skip_special_tokens=True, num_samples=1)
                print(f"generated_texts has a length of {len(generated_texts)}")

    return generated_texts


if not INFERENCE_ONLY:
    # Train and validate models
    if (USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER):
        lr = 1e-4
    else:
        lr = 3e-3
    weight_decay = 1e-5
    eps = 1e-8
    betas = (0.9, 0.999)  # only for Adam

    if USE_PRETRAINED_T5:
        # replace AdamW with Adafactor
        # See https://github.com/PiotrNawrot/nanoT5/blob/1375b389d33ab4f34754a9fca62e4cfa1dd52379/README.md?plain=1#L36
        optimizer_ebm = Adafactor(ebm.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        optimizer_denoiser = Adafactor(denoiser.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    else:
        if USE_ADAM_MINI:  # for saving RAM memory during training
            if not (USE_CUSTOM_TRANSFORMER_ENCODER_DECODER or USE_CUSTOM_TRANSFORMER_ENCODER):
                if USE_PRETRAINED_BERT_MLM:
                    model_dim = denoiser.config.hidden_size
                    num_heads = denoiser.config.num_attention_heads
                else:
                    model_dim = denoiser.config.dim
                    num_heads = denoiser.config.num_attention_heads

            optimizer_ebm = Adam_mini(
                                named_parameters = ebm.named_parameters(),
                                lr = lr,
                                betas = betas,
                                eps = eps,
                                weight_decay = weight_decay,
                                dim = model_dim_ebm,
                                n_heads = num_heads_ebm,
                                n_kv_heads = num_heads_ebm,
                            )

            optimizer_denoiser = Adam_mini(
                                    named_parameters = denoiser.named_parameters(),
                                    lr = lr,
                                    betas = betas,
                                    eps = eps,
                                    weight_decay = weight_decay,
                                    dim = model_dim,
                                    n_heads = num_heads,
                                    n_kv_heads = num_heads,
                                )

            # https://github.com/zyushun/Adam-mini/issues/30
            optimizer_ebm.embd_names.add('embedding') # add the keyword of the embedding layer
            optimizer_ebm.output_names.add('denoise_head') # output layer of EBM model is not using projection layer
            optimizer_denoiser.embd_names.add('embedding') # add the keyword of the embedding layer
            optimizer_denoiser.output_names.add('projection') # projection layer is using weight-tying with embedding layer

            optimizer_ebm.mlp_names = {"self_attn"}
            optimizer_denoiser.mlp_names = {"self_attn"}

            optimizer_ebm.mlp_names.add("attn")
            optimizer_ebm.mlp_names.add("linear")
            optimizer_denoiser.mlp_names.add("attn")
            optimizer_denoiser.mlp_names.add("linear")

            optimizer_denoiser.wqk_names.add("self_attn")  # For query, key, and value combined
            optimizer_denoiser.wqk_names.add("multihead_attn")

        else:
            optimizer_ebm = optim.AdamW(ebm.parameters(), lr=lr, weight_decay=weight_decay)  # Added weight decay
            optimizer_denoiser = optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=weight_decay)  # Added weight decay

    def warmup_schedule(current_step: int):
        warmup_steps = 1000
        step_size = 5 * len(train_loader)  # 5 epochs
        gamma = 0.5

        if current_step < warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # After warmup, apply step decay
            num_steps_after_warmup = current_step - warmup_steps
            num_step_decays = num_steps_after_warmup // step_size
            return gamma ** num_step_decays

    def warmup_cosine_schedule(current_step: int):
        warmup_steps = 500
        total_steps = num_epochs * len(train_loader)
        if current_step < warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # After warmup, apply cosine decay
            return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))


    if USE_PRETRAINED_T5:
        scheduler_ebm = AdafactorSchedule(optimizer_ebm)
        scheduler_denoiser = AdafactorSchedule(optimizer_denoiser)
    else:
        scheduler_ebm = LambdaLR(optimizer_ebm, lr_lambda=warmup_cosine_schedule)
        scheduler_denoiser = LambdaLR(optimizer_denoiser, lr_lambda=warmup_cosine_schedule)

        #scheduler_ebm = optim.lr_scheduler.StepLR(optimizer_ebm, step_size=5, gamma=0.5)
        #scheduler_denoiser = optim.lr_scheduler.StepLR(optimizer_denoiser, step_size=5, gamma=0.5)

    if USE_MIXED_PRECISION_TRAINING and torch.cuda.is_available():
        scaler = GradScaler()  # MPS backend does not have this option yet
    else:
        scaler = None

    best_val_ebm_loss = float('inf')
    best_val_denoiser_loss = float('inf')

    for epoch in range(num_epochs):
        # Train models
        train_ebm_loss, train_denoiser_loss = train_walk_jump(ebm, denoiser, train_loader, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, optimizer_ebm, optimizer_denoiser, scheduler_ebm, scheduler_denoiser, scaler)

        # Validate models
        val_ebm_loss, val_denoiser_loss = validate_walk_jump(ebm, denoiser, val_loader, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train EBM Loss: {train_ebm_loss:.8f}, Train Denoiser Loss: {train_denoiser_loss:.8f}, Val EBM Loss: {val_ebm_loss:.8f}, Val Denoiser Loss: {val_denoiser_loss:.8f}")

        # Save the trained models
        if val_ebm_loss < best_val_ebm_loss:
            best_val_ebm_loss = val_ebm_loss
            torch.save(ebm.state_dict(), 'best_ebm.pth')

        if val_denoiser_loss < best_val_denoiser_loss:
            best_val_denoiser_loss = val_denoiser_loss
            torch.save(denoiser.state_dict(), 'best_denoiser.pth')

        # Early stopping check
        if (val_denoiser_loss < EARLY_STOP_THRESHOLD) and USE_EARLY_STOP:
            print(f"Early stopping triggered. Validation loss ({val_denoiser_loss:.4f}) is below the threshold ({EARLY_STOP_THRESHOLD}).")
            break

    print("Training complete.")


# Inference
ebm.load_state_dict(torch.load('best_ebm.pth', weights_only=True))
denoiser.load_state_dict(torch.load('best_denoiser.pth', weights_only=True))

if not GENERATES_OUTPUT_OF_VARYING_LENGTH:
    #input_text = input_text + "an unfortunate shooting event in one of the Donald Trump's presidential election campaigns"
    if TEST_OVERFIT:
        input_text = dataset['train'][0]['text']
    else:
        input_text = dataset['train'][300]['text']
        #input_text = "Pandemic is inevitable these days, we need to ensure that we follow lockdown policies restrictions"

num_of_words_fed_into_the_model = 6
num_of_words_to_be_generated = len(input_text.split()) - num_of_words_fed_into_the_model
input_text = " ".join(input_text.split()[0:num_of_words_fed_into_the_model-1])
print(f"input_text = {input_text}")

generated_texts = infer_walk_jump(input_text, ebm, denoiser, num_walk_steps, walk_step_size, num_jump_steps, noise_schedule, max_length=num_of_words_to_be_generated)
print("Generated text:", generated_texts)
