import re
import torch
from torch import nn
from torch.nn import functional as F
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# Simple tokenizer for processing text and mapping words into ids

class SimpleTokenizer:
    def __init__(self, tokens):
        #tokens: list of string tokens(vocab)
        self.string_to_int = {s:i for i, s in enumerate(tokens)}
        self.int_to_string = {i:s for i, s in enumerate(tokens)}
    @staticmethod
    def preprocess(text):
        """handdling our tokens
        to be ready for encoding and decoding steps"""
        parts = re.split(r'([,.:;?_!"()\\]|--|\s)', text)
        parts = [i.strip() for i in parts if i and i.strip()] # enusre removing spaces
        return parts


    def encode(self, text):
        """returns ids for each coressponding word in the text"""
        toks = self.preprocess(text)
        idx = [self.string_to_int[i] for i in toks]
        return idx

    def decode(self, idx):
        """returns words for each coressponding idx
        which comes back from encoding step
        """
        text = " ".join([self.int_to_string[i] for i in idx])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\\])', r'\1', text)
        return text


# Bigram model (neural: embedding -> logits)
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #Embdding with dims = vocab_size, so each token maps to logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets= None):
        """our Neural net flow :)"""
        logits = self.token_embedding_table(idx) #B*T*C
        if targets is not None:
            B, T, C= logits.shape
            logits2 = logits.view(B*T, C)
            targets2 = targets.view(B*T)
            loss= F.cross_entropy(logits2, targets2)
            return logits2, loss
        return logits, None # Return logits and None for loss when targets is None


    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits_2, _ = self.forward(idx) #(B,T,C)
            logits_last = logits_2[:, -1, :] #(B,C)
            probs = F.softmax(logits_last, dim=1) # raw_logits --> probabilities
            next_id = torch.multinomial(probs, num_samples=1) #probabilities --> ids
            idx = torch.cat((idx, next_id), dim=1) # concatenating previous ids with new generated ones
        return idx


def get_batch(data, batch_size, block_size, device= "cpu"):
    """stacking our data as batches in top of each others with length of block_size (num of tokens)"""
    ix = torch.randint(len(data)-block_size, (batch_size,))
    X = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    X, y = X.to(device), y.to(device)
    return X, y

# train and predict in short corpus

text = "I love learning NLP because it helps me build AI agents."

tokens = SimpleTokenizer.preprocess(text)
vocab_list = sorted(list(tokens))
special_tokens= ["<|endoftext|>", "<|unk|>"] # handling unknown words and endoftext vocab

for s in special_tokens:
    if s not in vocab_list:
        vocab_list.append(s)

tokenizer = SimpleTokenizer(vocab_list)
ids = tokenizer.encode(text)
data = torch.tensor(ids, dtype= torch.long)

# spliting our humble dataset into train and val
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 1   # bigram
batch_size = 1
learning_rate = 1e-2
epochs = 150
vocab_size = len(vocab_list)
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

def train_step(train_data, model, optimizer, epochs, block_size, batch_size, device= device):
    train_loss = 0
    model.to(device)
    for epoch in range(epochs):
        xb, yb = get_batch(train_data, batch_size, block_size, device)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        train_loss+= loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch %10 == 0:
            print(f"Epoch: {epoch} and Train_loss: {loss:.4f}")
    avg_train_loss =train_loss /epochs 
    print(f"final average loss= {avg_train_loss:.4f}")

def val_step(val_data, model, epochs, device, block_size, batch_size):
    val_loss = 0
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for epoch in range(epochs):
            xb, yb = get_batch(val_data, batch_size, block_size, device)
            xb, yb = xb.to(device), yb.to(device)
            logits_val, loss_val = model(xb, yb)
            val_loss+=loss_val.item()
            if epoch %10 == 0:
                print(f"Epoch: {epoch} and val_loss: {loss_val:.4f}")
            avg_loss =val_loss / epochs
        print(f"Final average validation loss = {avg_loss:.2f} \n")
train_step(train_data= train_data, model=model, optimizer= optimizer, epochs = epochs, block_size= block_size, batch_size= batch_size, device = device)
val_step(val_data, model, epochs, device, block_size, batch_size)

def predict(model, 
            context_word, 
            device, 
            max_new_tokens):
  encoded_context = tokenizer.encode(context_word)
  context_tensor = torch.tensor([encoded_context], dtype = torch.long, device= device)
  generated_indices = model.generate(context_tensor, max_new_tokens)[0].tolist()
  predicted_text = context_word +tokenizer.decode(generated_indices[len(encoded_context):])
  print(predicted_text)

predict(model, "I love learning AI", device, max_new_tokens=3)