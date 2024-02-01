# cd c mnt/c/Users/mousa/Desktop/SemiEval/Code
import numpy as np
from datasets import Dataset
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import torch.utils.data
import pandas as pd
import os


GLOBAL_SEED = 10

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# model_name = 'bert-base-uncased'
# model_name = 'roberta-large'
model_name = 'xlm-roberta-base'


classes = [0,1,2,3]

SP_train_data = np.load('SP-train.npy', allow_pickle=True)
data = pd.DataFrame(list(SP_train_data))
df = pd.read_csv('BiRdQA_en_train(new).csv')

train_df, val_df = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
train_df.to_csv("./train_df.csv")
val_df.to_csv("./val_df.csv")
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
train_df.drop(['id', 'answer', 'distractor1', 'distractor2', 'distractor(unsure)','label'], axis=1, inplace=True)
val_df.drop(['id', 'answer', 'distractor1', 'distractor2', 'distractor(unsure)','label'], axis=1, inplace=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_tokenizer_output(tokenizer, text, attr):
    return np.array(tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")[attr])

def get_tokenizer_output(tokenizer, text1_list, text2_list, attr):
    return np.array(tokenizer(text1_list, text2_list, padding='max_length', truncation=True, return_tensors="pt")[attr])

class MultipleChoiceDataset(Dataset):
    def __init__(self, df, tokenizer, classes):
        super(MultipleChoiceDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        selected_item = self.df.iloc[idx]
        tokenized_item = self.tokenizer([selected_item['question']]*4, selected_item['choice_list'], padding='max_length', truncation=True, return_tensors="pt")
        item = {'input_ids': tokenized_item['input_ids'], 'attention_mask':tokenized_item['attention_mask'], 'labels' : selected_item['choice_order'][0]}
        return item
    


choice_list = []
choice_order = []
for i in range(len(df)):
  hold_choice = []
  hold_order = []

  hold_choice.append(df.iloc[i]["choice0"])
  hold_choice.append(df.iloc[i]["choice1"])
  hold_choice.append(df.iloc[i]["choice2"])
  hold_choice.append(df.iloc[i]["choice3"])

  hold_order.append(df.iloc[i]["(index of correct choice)"])
  for j in range(3):
    hold_order.append(-1)

  choice_order.append(hold_order)
  choice_list.append(hold_choice)

data  = {'question':df['riddle'].tolist(),'choice_list':choice_list,'choice_order':choice_order}
df_extend = pd.DataFrame(data)


# train_df= df_extend.append(train_df)
train_df = pd.concat([df_extend, train_df], axis=0)

train_dataset = MultipleChoiceDataset(train_df, tokenizer, classes)
val_dataset = MultipleChoiceDataset(val_df, tokenizer, classes)

print("ٔNumber of samples in train split", len(train_dataset))
print("Number of samples in val split", len(val_dataset))

def worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

model = AutoModelForMultipleChoice.from_pretrained(model_name)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 20
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
best_val_loss = float('inf')

total_steps = len(train_dataloader)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}:")
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, batch in enumerate(train_dataloader):
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        print(f"{(step + 1)} / {total_steps}")

        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs.loss
        _, preds = torch.max(outputs.logits, dim=1)

        total_loss = total_loss + loss.item()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Train: Acc = {total_correct/total_samples}, loss = {total_loss}, avg. loss = {total_loss/total_samples}")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            _, preds = torch.max(outputs.logits, dim=1)

            total_loss = total_loss + loss.item()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    print(f"Validation: Acc = {total_correct/total_samples}, loss = {total_loss}, avg. loss = {total_loss/total_samples}")
    print(50*"=")

    # Update the learning rate using the scheduler
    scheduler.step()

    # Save the model if it has the best validation loss
    if total_loss < best_val_loss:
        best_val_loss = total_loss
        model.save_pretrained('./best_model')


    # save_path_drive = '/content/drive/MyDrive/SemEval2024/Task9/model_checkpoint_epoch_{}.pth'.format(epoch + 1)
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': total_loss,
    #     'accuracy': total_correct / total_samples,
    # }, save_path_drive)


# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')