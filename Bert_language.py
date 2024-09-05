print('Start')
import torch
torch.set_num_threads(1)
from transformers import BertTokenizer, BertModel
import wandb
import numpy as np
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

max_num_of_numbers_check = -5

data_path = 'C:/Users/yarom/PycharmProjects/MusikTransformer/DataBase/Language_base_data/'

#wandb.login()


class Head_Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_activation = nn.GELU()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.linear_1 = torch.nn.Linear(768, 1536)
        self.conv1d_1 = torch.nn.Conv1d(1, 4, 3, padding=1)
        self.conv1d_2 = torch.nn.Conv1d(4, 8, 3, padding=1)
        self.conv1d_3 = torch.nn.Conv1d(8, 16, 3, padding=1)
        self.linear_2 = torch.nn.Linear(3072, 2048)

    def forward(self, batch):

        out = self.linear_1(batch)
        out = self.f_activation(out)
        out = torch.unsqueeze(out, 1)

        out = self.conv1d_1(out)
        out = self.f_activation(out)
        out = self.pool(out)

        out = self.conv1d_2(out)
        out = self.f_activation(out)
        out = self.pool(out)

        out = self.conv1d_3(out)
        out = self.f_activation(out)
        out = self.pool(out)

        out = torch.flatten(out, start_dim=1, end_dim=2)

        out = self.linear_2(out)

        return out


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", trainable = True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model_bert = DistilBertModel.from_pretrained(model_name).to(device)
        if trainable == False:
            for m in self.model_bert.modules():
                for name, params in m.named_parameters():
                    params.requires_grad = False
        self.target_token_idx = 0
        self.head = Head_Bert()

    def forward(self, text):
        encoding = self.tokenizer.batch_encode_plus(
            text,  # List of input texts
            padding="max_length",
            max_length=512,  # Pad to the maximum sequence length
            truncation=True,  # Truncate to the maximum sequence length if necessary
            return_tensors='pt',  # Return PyTorch tensors
            add_special_tokens=True  # Add special tokens CLS and SEP
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        out = self.model_bert(input_ids, attention_mask, output_hidden_states=True)
        out = out[0][:,self.target_token_idx,:]
        out = self.head(out)

        return out

class clip_models(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_first = TextEncoder()
        self.model_second = TextEncoder(trainable = False)
        self.ln = nn.LayerNorm([2048])
        self.temp = 0.07

    def forward(self, review, query):
        first_embeding = self.model_first(review)
        second_embeding = self.model_second(query)

        first_embeding = self.ln(first_embeding)
        second_embeding = self.ln(second_embeding)

        similarity = torch.matmul(first_embeding, second_embeding.T) * torch.exp(torch.tensor(self.temp))

        labels = torch.arange(similarity.shape[0]).to(device)# --------------------------------------------

        img_loss = self.cross_entropy_loss(similarity, labels)
        tex_loss = self.cross_entropy_loss(similarity.T, labels)

        loss = (img_loss + tex_loss) / 2


        return loss



class Bert_Dataset(Dataset):
    def __init__(self, data_path):
        self.dataset = []
        a = 0
        for album in range(0, 42):
            with open((data_path + str(album) + "/reviews.txt"), "r", encoding="utf-8") as file:
                reviews = file.read().split('/////')
                reviews = reviews[1:]
            with open((data_path + str(album) + "/query.txt"), "r", encoding="utf-8") as file:
                query = file.read()
                query = self.preprocess_text(query)

        for rewiew in reviews:
            if a < len(rewiew):
                a = len(rewiew)
            rewiew = self.preprocess_text(rewiew)
            self.dataset.append([rewiew, query])


    def preprocess_text(self, text):
        # lowercasing
        lowercased_text = text.lower()

        # cleaning
        import re
        remove_punctuation = re.sub(r'[^\w\s]', '', lowercased_text)
        remove_white_space = remove_punctuation.strip()

        # Tokenization = Breaking down each sentence into an array
        from nltk.tokenize import word_tokenize
        tokenized_text = word_tokenize(remove_white_space)

        # Stemming = Transforming words into their base form
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        stemmed_text = [ps.stem(word) for word in tokenized_text]

        # Putting all the results into a dataframe.
        df = stemmed_text

        df = ' '.join(df)

        return df


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


dataset = Bert_Dataset(data_path)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)



class train_module():
    def __init__(self, model, criterion, epochs, train_dataloader, val_dataloader):
        super().__init__()
        self.model = model.to(device)
        self.criterion = criterion
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.params = [{"params": self.model.model_first.parameters()},
                  {"params": self.model.model_second.parameters()}]
        self.optimizer = torch.optim.Adam(self.params)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=1, factor=0.8)

    def training_step(self, train_batch):
        rewiew, query = train_batch
        loss = self.model(list(rewiew), list(query))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        wandb.log({"loss_train": loss})


    def validation_step(self, val_batch):
        rewiew, query = val_batch
        loss = self.model(list(rewiew), list(query))
        wandb.log({"loss_val": loss})

    def train(self):
        for epoch in range(0, self.epochs):
            self.model.train()
            for batch in self.train_dataloader:
                self.training_step(batch)

            self.model.eval()
            for batch in self.val_dataloader:
                self.validation_step(batch)
        return self.model



model = clip_models()
criterion = torch.nn.CrossEntropyLoss()
epoch = 1

run = wandb.init(
    project="Encoder_train",
    config={
        "config": 'One Bert train and two heads',
    }
)


train_ = train_module(model, criterion, epoch, train_dataloader, val_dataloader)
if __name__ == '__main__':
    model = train_.train()

path = 'C:/Users/yarom/PycharmProjects/MusikTransformer/Text_Encoder.pt'
print("saving model")
torch.save(model.state_dict(), path)



