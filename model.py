import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset import train_pairs # Import training data

# -----------------------------------------------------------------------------
# Vocab Class
# -----------------------------------------------------------------------------

class Vocab:
    def __init__(self, tokens):
         # Define special token indices for padding, start-of-sentence (sos),
        # end-of-sentence (eos), and unknown tokens.
        self.pad = 0
        self.sos = 1
        self.eos = 2
        self.unk = 3

        # Build a set of unique characters from all tokens in the dataset.
        chars = set()
        for t in tokens:
            chars.update(t)

        # The vocabulary list starts with special tokens and then sorted unique characters.
        self.chars = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(chars)

        # Create a mapping from character to index and vice versa.
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}

    def numericalize(self, text):
        # Convert a string (text) into a list of indices.
        # If a character is not in the vocabulary, it defaults to the unknown token index.
        return [self.stoi.get(c, self.unk) for c in text]

# Create vocabularies for source and target using the training pairs.
src_vocab = Vocab([pair[0] for pair in train_pairs])
tgt_vocab = Vocab([pair[1] for pair in train_pairs])

# -----------------------------------------------------------------------------
# Dataset and DataLoader Setup
# -----------------------------------------------------------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs         # List of (source, target) sentence pairs.
        self.src_vocab = src_vocab # Source language vocabulary.
        self.tgt_vocab = tgt_vocab # Target language vocabulary.

    def __len__(self): 
        # Return the number of sentence pairs.
        return len(self.pairs)

    def __getitem__(self, idx):
         # Get the sentence pair at index idx.
        src, tgt = self.pairs[idx]
        # Convert source and target sentences into tensors of token indices.
        # Each sequence is prepended with a start token and appended with an end token.
        src_tensor = [src_vocab.sos] + src_vocab.numericalize(src) + [src_vocab.eos]
        tgt_tensor = [tgt_vocab.sos] + tgt_vocab.numericalize(tgt) + [tgt_vocab.eos]
        return torch.LongTensor(src_tensor), torch.LongTensor(tgt_tensor)

def collate_fn(batch):
    # Unzip the batch into source and target sequences.
    src_batch, tgt_batch = zip(*batch)
    # Pad sequences in the batch to the same length for both source and target.
    src_padded = pad_sequence(src_batch, padding_value=src_vocab.pad)
    tgt_padded = pad_sequence(tgt_batch, padding_value=tgt_vocab.pad)
    return src_padded, tgt_padded

# Create dataset and DataLoader for training.
dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)


# -----------------------------------------------------------------------------
# Context-Aware Transformer Model Definition
# -----------------------------------------------------------------------------
class ContextAwareTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4,
                 num_layers=3, dropout=0.2):
        super().__init__()
        # Define the encoder part of the Transformer.
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, 256, dropout),
            num_layers
        )
        # Define the decoder part of the Transformer.
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, 256, dropout),
            num_layers
        )
        # Embedding layers for source and target vocabularies.
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
         # Positional encoding is learned here and has a fixed maximum sequence length (100).
        self.pos_encoder = nn.Parameter(torch.randn(100, 1, d_model))
        # Final linear layer to project decoder outputs to target vocabulary space.
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model  # Save the model dimension for scaling

    def forward(self, src, tgt):
        # Embed the source and target sequences.
        src_embed = self.src_embed(src) * torch.sqrt(torch.tensor(self.d_model))
        tgt_embed = self.tgt_embed(tgt) * torch.sqrt(torch.tensor(self.d_model))

        # Add positional encoding to the embeddings.
        src_embed = src_embed + self.pos_encoder[:src.size(0), :]
        tgt_embed = tgt_embed + self.pos_encoder[:tgt.size(0), :]

        # Create a mask for the source sequence so that padding tokens are ignored.
        src_pad_mask = (src == src_vocab.pad).T
        # Encode the source sequence.
        memory = self.encoder(src_embed, src_key_padding_mask=src_pad_mask)

        # Generate a mask for the target sequence to prevent positions from attending
        # to subsequent positions (autoregressive decoding).
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)
        # Create a padding mask for the target sequence.
        tgt_pad_mask = (tgt == tgt_vocab.pad).T

        # Decode the target sequence using the encoder memory.
        output = self.decoder(
            tgt_embed, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        # Final projection to obtain logits over the target vocabulary.
        return self.fc_out(output)

# Determine if GPU is available; otherwise use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate the model with appropriate vocabulary sizes and move it to the device.
model = ContextAwareTransformer(len(src_vocab.chars), len(tgt_vocab.chars)).to(device)
# Define the optimizer (Adam) with learning rate and weight decay.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Define the loss function (cross-entropy) with label smoothing.
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad, label_smoothing=0.1)

# -----------------------------------------------------------------------------
# Training Loop with Early Stopping
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    best_loss = float('inf')
     # Run training for a maximum of 100 epochs.
    for epoch in range(100):
        model.train()
        total_loss = 0
        # Loop over batches provided by the DataLoader.
        for src, tgt in dataloader:
            # Move the batch data to the appropriate device.
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad() # Zero the gradients from the previous step.
            # Forward pass: predict the target sequence (excluding the final token).
            output = model(src, tgt[:-1])
            # Calculate the loss comparing the output with the actual target (shifted by one).
            loss = criterion(output.view(-1, output.shape[-1]), tgt[1:].view(-1))
            loss.backward()# Backpropagation.
             # Clip gradients to avoid exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()# Update model parameters.
            total_loss += loss.item()

        avg_loss = total_loss/len(dataloader)
        # Save the model if the current average loss is the best seen so far.
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pt')

        # Print progress every 10 epochs.
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # After training, load the best model saved during training.
    model.load_state_dict(torch.load('best_model.pt'))


# -----------------------------------------------------------------------------
# Translation Function Using Beam Search
# -----------------------------------------------------------------------------
def translate(model, sentence, src_vocab, tgt_vocab, max_len=20, beam_size=2):
    model.eval()# Set model to evaluation mode.
    # Convert the input sentence to a list of token indices with sos and eos tokens.
    tokens = [src_vocab.sos] + src_vocab.numericalize(sentence) + [src_vocab.eos]
    src = torch.LongTensor(tokens).unsqueeze(1).to(device)# Add batch dimension.

    # Compute the source embeddings and add positional encoding.
    src_embed = model.src_embed(src) * torch.sqrt(torch.tensor(model.d_model))
    src_embed = src_embed + model.pos_encoder[:src.size(0), :]
    # Obtain the encoder's memory representation.
    memory = model.encoder(src_embed)

    # Start beam search with an initial sequence containing only the start token.
    sequences = [[ [tgt_vocab.sos], 0 ]]
    for _ in range(max_len):
        candidates = []
        # Expand each sequence in the beam.
        for seq, score in sequences:
            # If the sequence already ends with the end-of-sentence token, keep it unchanged.
            if seq[-1] == tgt_vocab.eos:
                candidates.append((seq, score))
                continue

            # Prepare the current sequence for the decoder.
            tgt = torch.LongTensor(seq).unsqueeze(1).to(device)
            tgt_embed = model.tgt_embed(tgt) * torch.sqrt(torch.tensor(model.d_model))
            tgt_embed = tgt_embed + model.pos_encoder[:tgt.size(0), :]

            # Pass through the decoder with the encoder memory.
            output = model.decoder(
                tgt_embed,
                memory
            )
            # Compute logits for the next token.
            logits = model.fc_out(output[-1])
            # Convert logits to log-probabilities.
            probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Select the top `beam_size` tokens to expand the sequence.
            top_probs, top_ids = probs.topk(beam_size)
            for i in range(beam_size):
                 # Append the selected token and update the cumulative score.
                candidates.append((seq + [top_ids[0,i].item()], score + top_probs[0,i].item()))

        # Sort candidates by normalized score (score per token length) in descending order.
        candidates.sort(key=lambda x: x[1]/len(x[0]), reverse=True)
        # Keep the best `beam_size` sequences.
        sequences = candidates[:beam_size]

    # Choose the best sequence from the final candidates.
    best_seq = max(sequences, key=lambda x: x[1]/len(x[0]))[0]
    # Convert token indices back to characters, ignoring special tokens
    return ''.join([tgt_vocab.itos[t] for t in best_seq if t not in {tgt_vocab.sos, tgt_vocab.eos, tgt_vocab.pad}])