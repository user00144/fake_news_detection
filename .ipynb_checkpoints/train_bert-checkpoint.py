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
import random
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#params
batch_size = 8
EPOCHS = 1
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
        input_ids = data.input_ids.squeeze()
        attention_mask = data.attention_mask.squeeze()

        return input_ids, attention_mask, label

    def tokenize(self, data) :
        token = self.tokenizer(data, padding = 'max_length', max_length = self.max_length, truncation = True, return_tensors = "pt")
        del token['token_type_ids']
        return token
    
    def __len__(self) :
        return len(self.dataframe)

#define Model
class Model(BertPreTrainedModel) :
    def __init__(self, output_classes, config) :
        super().__init__(config)
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1", config=config)
        self.output_layer = nn.Linear(config.hidden_size, output_classes)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, attention_mask) :
        hidden_state = self.bert(input_ids = input_ids,
                                 attention_mask = attention_mask)
        data = hidden_state[1]
        output = self.dropout(data)
        output = self.output_layer(output)
        return output

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_states=True)
conf =  AutoConfig.from_pretrained("skt/kobert-base-v1")
model = Model(output_classes = 2, config = conf).to(device)

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
    for batch, (input_ids, attention_mask, label) in enumerate(tqdm(dataloader)) :
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask).to(device)
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
        for batch, (input_ids, attention_mask, label) in enumerate(tqdm(dataloader)) :
            input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask).to(device)
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
            
            print(f'==FOLD {i} / EPOCH {epoch}==')
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
            print(f'<<FOLD {i}>>')
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
        print(f"\t <<FOLD{i}>> Test loss {loss} | acc {acc}")
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

