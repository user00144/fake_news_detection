from sklearn.metrics import f1_score
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

#fix seed
seed = 511
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

df_test = pd.read_csv('./data/Test_Part1.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)


    def forward(self, input_ids, attention_mask) :
        hidden_state = self.bert(input_ids = input_ids,
                                 attention_mask = attention_mask)
        data = hidden_state[1]
        output = self.dropout(data)
        output = self.dense(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.output_layer(output)
        return output



tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_states=True)
conf =  AutoConfig.from_pretrained("skt/kobert-base-v1")
model = Model(output_classes = 2, config = conf).to(device)
test_dataset = CustomDataset(dataframe = df_test,
                             tokenizer = tokenizer,
                             max_length = 512)
test_dataloader = DataLoader(test_dataset, batch_size = 32, num_workers = 4, shuffle = False)

def eval_func(model, dataloader) :
    e_f = 0, 0
    model.eval()
    with torch.no_grad() :
        for batch, (input_ids, attention_mask, label) in enumerate(tqdm(dataloader)) :
            input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask).to(device)
            outputs = torch.softmax(outputs, dim=1).to(device)
            f = f1_score(outputs.argmax(dim=1).cpu(), label.cpu())
            e_f += f

        e_f /= len(dataloader)

    return e_f


def test(test_dataloader, model) :
    print("<<TEST>>")
    test_f = []
    for i in range(5):
        model.load_state_dict(torch.load(f'./models/best_w_{i}.pth'))
        f = eval_func(model, test_dataloader)
        print(f"\t <<FOLD{i+1}>> f1 {f}")
        test_f.append(f)

    print(f"f1 {np.mean(test_f):.4f}")
    print(f"f1 {np.var(test_f):.4f}")



test(test_dataloader = test_dataloader,
     model = model)
