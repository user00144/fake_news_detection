import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
import torch
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertPreTrainedModel
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from einops import rearrange
import random
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#params
batch_size = 8
EPOCHS = 3
PATIENT = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#fix seed
seed = 511
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# #tensorboard_logger_start
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('./runs/clickbait_kobert')
# tb_logging_rate = 100


#Load dataframe
df_train = pd.read_csv('./data/Train_Part1.csv')
df_test = pd.read_csv('./data/Test_Part1.csv')

#define Dataset
class CustomDataset(Dataset) :
    def __init__(self, dataframe, tokenizer, max_length) :
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, x) :
        data = self.dataframe.loc[x].title + '[CLS][PAD][PAD]' + self.dataframe.loc[x].content
        data = self.tokenize(data)
        label = torch.tensor(self.dataframe.loc[x].label)
        input_ids = data.input_ids
        return input_ids, label

    def tokenize(self, data) :
        token = self.tokenizer(data, padding = 'max_length', max_length = self.max_length, truncation = True, return_tensors = "pt")
        del token['token_type_ids']
        return token
    
    def __len__(self) :
        return len(self.dataframe)

#define Model
class HierAttNet(nn.Module):
    def __init__(self, word_dims: int = 32, sent_dims: int = 64, dropout: float = 0.1, num_classes: int = 2, 
                 vocab_len: int = 50002, embed_dims: int = 100):
        super(HierAttNet, self).__init__()

        # word attention
        self.word_attention = WordAttnNet(
            vocab_len  = vocab_len, 
            embed_dims = embed_dims,
            word_dims  = word_dims,
            dropout    = dropout
        )

        # sentence attention
        self.sent_attention = SentAttnNet(
            word_dims = word_dims, 
            sent_dims = sent_dims, 
            dropout   = dropout
        )

        # classifier
        self.fc = nn.Linear(2 * sent_dims, num_classes)

    def init_w2e(self, weights: np.ndarray, nb_special_tokens: int = 0):

        weights = torch.from_numpy(
            np.concatenate([
                weights, 
                np.random.randn(nb_special_tokens, weights.shape[1])
            ]).astype(np.float)
        )
        self.word_attention.w2e = self.word_attention.w2e.from_pretrained(weights)

    def freeze_w2e(self):
        self.word_attention.w2e.weight.requires_grad = False

    def forward(self, input_ids, output_attentions: bool = False):
        # word attention
        words_embed, words_attn_score = self.word_attention(input_ids) 

        # sentence attention
        sents_embed, sents_attn_score = self.sent_attention(words_embed)

        # classification
        out = self.fc(sents_embed)

        if output_attentions:
            return out, words_attn_score, sents_attn_score
        else:
            return out


class WordAttnNet(nn.Module):
    def __init__(self, vocab_len, embed_dims, word_dims, dropout):
        super(WordAttnNet, self).__init__()
        # word to embeding
        self.w2e = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dims)

        # word attention
        self.gru = nn.GRU(embed_dims, word_dims, bidirectional=True, dropout=dropout)
        self.attention = Attention(2 * word_dims, word_dims)

        # layer norm and dropout
        self.layer_norm = nn.LayerNorm(2 * word_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # docs: B x sents x words
        b, s, _ = input_ids.size()
        input_ids = rearrange(input_ids, 'b s w -> (b s) w')

        # input_embed: (B x sents) x words x dims
        input_embed = self.w2e(input_ids)
        input_embed = self.dropout(input_embed)

        # word attention
        words_embed, _ = self.gru(input_embed.float())
        words_embed = self.layer_norm(words_embed)

        words_embed, words_attn_score = self.attention(words_embed)

        words_embed = rearrange(words_embed, '(b s) d -> b s d', b=b, s=s)
        words_embed = self.dropout(words_embed)

        return words_embed, words_attn_score


class SentAttnNet(nn.Module):
    def __init__(self, word_dims, sent_dims, dropout):
        super(SentAttnNet, self).__init__()
        # sentence attention
        self.gru = nn.GRU(2 * word_dims, sent_dims, bidirectional=True, dropout=dropout)
        self.attention = Attention(2 * sent_dims, sent_dims)

        # layer norm and dropout
        self.layer_norm = nn.LayerNorm(2 * sent_dims)

    def forward(self, words_embed):
        # sentence attention
        sents_embed, _ = self.gru(words_embed)
        sents_embed = self.layer_norm(sents_embed)

        sents_embed, sents_attn_score = self.attention(sents_embed)
        
        return sents_embed, sents_attn_score


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(in_dim, out_dim)
        self.context = nn.Linear(out_dim, 1)

    def forward(self, x):
        attn = torch.tanh(self.attention(x))
        attn = self.context(attn).squeeze(2)
        attn_score = torch.softmax(attn, dim=1)

        out = torch.einsum('b n d, b n -> b d', x, attn_score) 

        return out, attn_score


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_states=True)
conf =  AutoConfig.from_pretrained("skt/kobert-base-v1")
model = HierAttNet().to(device)

train_dataset = CustomDataset(dataframe = df_train,
                              tokenizer = tokenizer,
                              max_length = 512)
test_dataset = CustomDataset(dataframe = df_test,
                             tokenizer = tokenizer,
                             max_length = 512)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 4, shuffle = False)



loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    params       = filter(lambda p: p.requires_grad, model.parameters()), 
    lr           = 0.00001, 
    weight_decay = 0.0005
)

warmup_ratio = 0.1

t_total = len(df_train) * EPOCHS * 4

warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)

fold = KFold(n_splits = 5, shuffle = True, random_state = seed)

def acc_fn(y_pred, y_true):
    accuracy = torch.eq(y_pred, y_true).sum().item()/len(y_pred)
    return accuracy

def train_func(model, dataloader, optim, loss_fn, scheduler) :
    t_loss, t_acc = 0, 0
    model.train()
    for batch, (input_ids, label) in enumerate(tqdm(dataloader)) :
        input_ids, label = input_ids.to(device), label.to(device)
        
        outputs = model(input_ids = input_ids).to(device)
        outputs = torch.softmax(outputs, dim=1).to(device)

        loss = loss_fn(outputs, label)
        acc = acc_fn(outputs.argmax(dim=1).to(device), label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        t_loss += loss.item()
        t_acc += acc

    t_loss /= len(dataloader)
    t_acc /= len(dataloader)
    return t_loss, t_acc

def eval_func(model, dataloader, loss_fn) :
    e_loss, e_acc = 0, 0
    model.eval()
    with torch.no_grad() :
        for batch, (input_ids, label) in enumerate(tqdm(dataloader)) :
            input_ids, label = input_ids.to(device), label.to(device)
            outputs = model(input_ids = input_ids).to(device)
            outputs = torch.softmax(outputs, dim=1).to(device)
            
            loss = loss_fn(outputs, label)
            acc = acc_fn(outputs.argmax(dim=1).to(device), label)

            e_loss += loss.item()
            e_acc += acc

        e_loss /= len(dataloader)
        e_acc /= len(dataloader)

    return e_loss, e_acc

def train(train_dataset , model, epochs, optim, loss_fn, patient, scheduler) :
    print('<<TRAIN>>')
    tot_tr_loss, tot_tr_acc = [], []
    tot_val_loss, tot_val_acc = [], []
    for i, (train_idx, val_idx) in enumerate(fold.split(train_dataset)) :
        min_val_loss = 2
        n_patience = 0
        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)

        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 4)
        val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = 4)

        for epoch in range(epochs) :
            
            print(f'\n==FOLD {i+1} || EPOCH {epoch+1}/{epochs}==')
            train_loss, train_acc = train_func(model, train_dl, optim, loss_fn, scheduler)
            val_loss, val_acc = eval_func(model, val_dl, loss_fn)
            
            print(f'Train_loss : {train_loss} || Train_acc : {train_acc}\n Val_loss : {val_loss} || Val_acc : {val_acc}')
            
            if np.round(min_val_loss, 5) > np.round(val_loss, 5) :
                min_val_loss = val_loss
                n_patience = 0
                print(f'Save the best params with val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}')
                torch.save(model.state_dict(), f'./models/best_w_{i}.pth')
            else :
                n_patience += 1
            
            if n_patience >= patient :
                print('Early Stopping')
                break

            tot_tr_loss.append(train_loss)
            tot_tr_acc.append(train_acc)
            tot_val_loss.append(val_loss)
            tot_val_acc.append(val_acc)
            print(f'\n<<FOLD {i+1}>>')
            print(f"\t Train loss {np.mean(tot_tr_loss):.4f} | acc {np.mean(tot_tr_acc):.4f}")
            print(f"\t Valid loss {np.mean(tot_val_loss):.4f} | acc {np.mean(tot_val_acc):.4f}")

            # #tensorboard_logger_close
            # writer.close()


def test(test_dataloader, model , loss_fn) :
    print("<<TEST>>")
    test_loss, test_acc = [],[]
    for i in range(5):
        model.load_state_dict(torch.load(f'./models/best_w_{i}.pth'))
        loss, acc = eval_func(model, test_dataloader, loss_fn)
        print(f"\t <<FOLD{i+1}>> Test loss {loss} | acc {acc}")
        test_loss.append(loss)
        test_acc.append(acc)

    print(f"Average  loss {np.mean(test_loss):.4f} | acc {np.mean(test_acc):.4f}")
    print(f"Variance loss {np.var(test_loss):.4f} | acc {np.var(test_acc):.4f}")

train(train_dataset = train_dataset,
       model = model,
       epochs = EPOCHS,
       optim = optimizer,
       loss_fn = loss_fn,
       patient = PATIENT,
       scheduler = scheduler)

test(test_dataloader = test_dataloader,
     model = model,
     loss_fn = loss_fn)

