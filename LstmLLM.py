from torch import nn

class LstmLLm(nn.Module):
    def __init__(self, vocab_size, embeddin_dim, hidden_dim,
                 num_layers, device, dropout_embd=0.05, dropout_rnn=0.05): #*args, **kwargs
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embeddin_dim, device=device)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.dropout = nn.Dropout(p=dropout_embd)
        
        self.lstm = nn.LSTM(embeddin_dim, hidden_dim, num_layers, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        #batch_size, seq_len = x.shape
        print(f"LLM_Class input x_batch.shape : {x.shape}") # [64, 120]
        
        y_hat = self.embedding(x)          # [64, 120, 128]
        #print(f"y_hat_after_embedding {y_hat.shape}")

        #y_hat = self.dropout(self.embedding(x))
        y_hat = self.dropout(y_hat)        # [64, 120, 128]
        #print(f"y_hat_after_dropout {y_hat.shape}")
        
        y_hat, (h_n, c_n) = self.lstm(y_hat)   # [64, 120, 256]
        #print(f"y_hat_after_lstm {y_hat.shape}")

        y_hat = self.fc(y_hat)             # [64, 120, 28782]
        #print(f"y_hat_after_fc {y_hat.shape}")

        return y_hat, (h_n, c_n)