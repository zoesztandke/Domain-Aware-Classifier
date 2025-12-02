import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import Adam
from torch import nn
import numpy as np
from tqdm.auto import tqdm

# from google.colab import drive
# drive.mount('/content/drive')

if torch.cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'

TRANSFORM = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

DOMAIN_EPOCHS = 60
PRED_EPOCHS = 60
BATCH_SIZE = 16

class MyModel(nn.Module):
    def __init__(self, train_mode: bool = False, domain: bool = False, num_classes: int = 10):
        super().__init__()
        self.prediction_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=None)
        self.prediction_model.classifier[1] = nn.Linear(self.prediction_model.last_channel, num_classes)
        self.prediction_model.to(DEVICE)

        self.domain_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=None)
        self.domain_model.classifier[1] = nn.Linear(self.domain_model.last_channel, 1)
        self.domain_model.to(DEVICE)

        self.train_mode = train_mode
        self.domain = domain

    def forward(self, img):
        # training prediction flow
        if self.train_mode:
            if self.domain:
                prob = self.domain_model(img)
                prob = prob.view(-1, 1)
                return prob
            else:
                return self.prediction_model(img)

        # normal prediction flow
        logit = self.domain_model(img)
        logit = logit.view(1, -1)

        prob = torch.sigmoid(logit)
        predictions = self.prediction_model(img)

        for i, p in enumerate(prob.flatten()):
            if p < 0.5:
                artificial_prediction = torch.tensor([0] * 10)
                rand_num = np.random.randint(0, 10)
                artificial_prediction[rand_num] = 1
                predictions[i] = artificial_prediction

        return predictions

class BinaryDataLoader(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item, _ = self.dataset[index]
        label = self.label
        return item, label

def get_domain_data(in_path, out_path):
    in_data = datasets.ImageFolder(in_path, transform=TRANSFORM)
    out_data = datasets.ImageFolder(out_path, transform=TRANSFORM)

    in_data_loader = BinaryDataLoader(in_data, 1)
    out_data_loader = BinaryDataLoader(out_data, 0)

    combined_dataset = ConcatDataset([in_data_loader, out_data_loader])
    combined_data_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return combined_data_loader


def learn(path_to_in_domain: str, path_to_out_domain: str):
    in_data = datasets.ImageFolder(path_to_in_domain, transform=TRANSFORM)

    in_data_loader = DataLoader(in_data, batch_size=BATCH_SIZE, shuffle=True)
    comb = get_domain_data(path_to_in_domain, path_to_out_domain)

    model = MyModel()

    # train domain
    model.train_mode = True
    model.domain = True

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for e in range(DOMAIN_EPOCHS):
        print(f'epoch: {e}')
        running_loss = 0.
        for i, data in tqdm(enumerate(comb), total=len(comb)):
            inputs, labels = data
            inputs = inputs.to(DEVICE)

            labels = labels.view(-1, 1)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.float(), labels.float())
            running_loss += loss.item()
            loss.backward()

            optimizer.step()

        print(f'running loss for epoch {e}: {running_loss}')

    # train prediction
    model.train_mode = True
    model.domain = False

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for e in range(PRED_EPOCHS):
        print(f'epoch: {e}')
        running_loss = 0.
        for i, data in tqdm(enumerate(in_data_loader), total=len(in_data_loader)):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
        print(f'running loss for epoch {e}: {running_loss}')

    # torch.save(model.state_dict(), 'myModel.pth')
    return model

def accuracy(path_to_eval_folder: str, model) -> float:
    data = datasets.ImageFolder(path_to_eval_folder, transform=TRANSFORM)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    model.train_mode = False
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            outputs = outputs.view(-1, 10)
            outputs = outputs.to(DEVICE)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

model = learn('A4data/in-domain-train', 'A4data/out-domain-train')

new_model = MyModel()
new_model.to(DEVICE)
state_dict = torch.load('myModel.pth', map_location=DEVICE)
new_model.load_state_dict(state_dict)

# helper func
def test_domain_model(model, in_dom, out_dom):
    comb = get_domain_data(in_dom, out_dom)

    model.train_mode = True
    model.domain = True
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(comb, total=len(comb)):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = labels.view(-1, 1)

            logits = model(inputs)
            logits = logits.view(-1, 1)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def test_prediction_model(model, in_dom):
    data = datasets.ImageFolder(in_dom, transform=TRANSFORM)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    model.train_mode = True
    model.domain = False
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            outputs = outputs.view(-1, 10)
            outputs = outputs.to(DEVICE)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

helper_acc_1 = test_domain_model(new_model, 'A4data/in-domain-eval', 'A4data/out-domain-eval')
print(helper_acc_1)

helper_acc_2 = test_prediction_model(new_model, 'A4data/in-domain-eval')
print(helper_acc_2)

acc = accuracy('A4data/in-domain-eval', model)
print(acc)
acc_2 = accuracy('A4data/out-domain-eval', model)
print(acc_2)