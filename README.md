Character-level bigram language model trained on Rumi's poetry. Built from scratch in PyTorch with sampling and self-attention examples.

🧠 Bigram Language Model from Scratch
A beginner-friendly, character-level Bigram Language Model built using PyTorch, trained on Rumi's poetry. This project demonstrates how to implement core NLP concepts such as tokenization, embeddings, loss computation, sampling, and self-attention—all from the ground up.

📚 Overview
This project walks through the complete process of building a simple but functional language model that learns to predict the next character in a sequence. It uses:

Custom tokenization

Manual data preparation

A simple nn.Embedding-based bigram model

Basic sampling for text generation

Training using gradient descent

Toy examples to explore self-attention and layer normalization

🧾 Project Structure
Step 1: Load Dataset
Rumi's poetry text is loaded from a .txt file.

The text is printed and inspected to check size and content.

Step 2: Vocabulary Construction
All unique characters are extracted to form a vocabulary.

Two maps are created:

stoi: character → integer

itos: integer → character

Helper functions encode() and decode() convert between text and token IDs.

Step 3: Data Conversion
The full text is converted to a PyTorch tensor of token IDs.

This tensor is used to train the model.

Step 4: Train/Validation Split
The data is split: 90% for training, 10% for validation.

A simple demonstration shows how input/target pairs are formed for training.

Step 5: Data Batching
A batch generation function randomly samples sequences of a fixed length (block size).

Returns both inputs and targets for each mini-batch.

Step 6: Bigram Language Model
A simple neural network model is defined:

Uses a single embedding layer where each token maps directly to logits for the next token.

Loss is computed using cross-entropy between predicted logits and target tokens.

A generate() function allows sampling new characters autoregressively.

Step 7: Pre-Training Sampling
Before training, the model generates random text (mostly gibberish) using the initial, untrained weights.

Step 8: Model Training
Trains the model for a small number of steps using the AdamW optimizer.

After training, the loss is printed and final predictions are much more coherent.

Step 9: Post-Training Sampling
Generates new text from the trained model.

The generated output resembles Rumi's poetic style at the character level.

🧪 Educational Toy Examples
These sections are designed to illustrate advanced concepts using simple, small matrices.

Step 10: Weighted Averages via Matrix Multiplication
Shows how to compute a running average using lower triangular matrices.

Demonstrates how attention-like operations aggregate values over time.

Step 11: Average Over Time via Matrix Ops
Computes sequence-wise averages in multiple ways:

Naive loop-based method

Matrix multiplication

Softmax-weighted averaging

Step 12: Self-Attention Mechanics
Builds the full self-attention mechanism:

Query, Key, Value linear layers

Attention weights via scaled dot product

Causal masking using tril to prevent "peeking ahead"

Final output computed via weighted sum of values

Step 13: Custom Layer Normalization
Implements LayerNorm from scratch:

Normalizes features per sample

Uses learnable parameters gamma and beta

Demonstrates stability and output statistics

🛠️ Requirements
Python 3.7+

PyTorch

Numpy

(Optional) A text file of Rumi’s poetry named Rumi_poetry.txt

🚀 How to Run
Clone the repo or copy the script.

Place Rumi_poetry.txt in the correct path.

Run the Python script.

Observe:

Tokenized input/output pairs

Pre- and post-training text generations

Visualization of attention and normalization behavior

📌 Key Concepts Learned
Character-level language modeling

Bigram token prediction

Embedding layers

Cross-entropy loss for classification

Gradient-based training (backpropagation)

Text sampling

Self-attention and causal masking

Layer normalization

