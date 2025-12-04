from torch import nn, cuda, hub, sigmoid, tensor, save, load, no_grad, max
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.optim import Adam

from torchvision import datasets, transforms
from numpy.random import randint

if cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'

# global variables
TRANSFORM = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
DOMAIN_EPOCHS = 1
PRED_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

class DualModel(nn.Module):
    def __init__(self, train_mode: bool=False, domain_mode: bool=False, num_classes: int=10):
        super().__init__()
        self.prediction_model = hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=None)
        self.prediction_model.classifier[1] = nn.Linear(self.prediction_model.last_channel, num_classes)
        self.prediction_model.to(DEVICE)

        self.domain_model = hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=None)
        self.domain_model.classifier[1] = nn.Linear(self.domain_model.last_channel, 1)
        self.domain_model.to(DEVICE)

        self.train_mode = train_mode
        self.domain_mode = domain_mode

    def forward(self, imgs):
        """
        If the model is in training mode:
        -> If the model is in domain mode:
        ---> Train domain model
        -> Otherwise:
        ---> Train prediction model

        If the model is in prediction mode:
        -> If the image is predicted to be in-domain:
        ---> Use prediction model to predict class
        -> Otherwise
        ---> Predict a random class
        """
        # training flow
        if self.train_mode:
            # if training domain model
            if self.domain_mode:
                domain_logits = self.domain_model(imgs)
                domain_logits = domain_logits.view(-1, 1)
                return domain_logits
            # if training prediction model
            else:
                class_logits = self.prediction_model(imgs)
                return class_logits

        # normal prediction flow
        domain_logits = self.domain_model(imgs)
        domain_logits = domain_logits.view(-1, 1)
        domain_probs = sigmoid(domain_logits)

        class_logits = self.prediction_model(imgs)

        for i, p in enumerate(domain_probs.flatten()):
            if p < 0.5:
                artificial_prediction = tensor([0] * 10)
                rand_num = randint(0, 10)
                artificial_prediction[rand_num] = 1
                class_logits[i] = artificial_prediction
        return class_logits

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
    """
    Takes the filepath to in-domain and out-domain training data. Combines the in and out domain training data into
    a single, binary data loader. Labels the in-domain data as 1 and the out-domain data as 0. Allows the domain
    model to learn from these labels.
    """
    in_data = datasets.ImageFolder(in_path, transform=TRANSFORM)
    out_data = datasets.ImageFolder(out_path, transform=TRANSFORM)

    in_data_loader = BinaryDataLoader(in_data, 1)
    out_data_loader = BinaryDataLoader(out_data, 0)

    combined_dataset = ConcatDataset([in_data_loader, out_data_loader])
    combined_data_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return combined_data_loader

def learn(path_to_in_domain: str, path_to_out_domain: str):
    """
    Takes file-paths to in-domain and out-domain training data. First, reads in the in-domain and out-domain training
    data, and uses a binary dataloader to train the domain classifier for DOMAIN_EPOCHS epochs. Next, reads the
    in-domain training data into a dataloader with 10 different labels, and trains the prediction model on only this
    data (excludes out-domain). Returns a dual model of class DualModel.
    """
    in_data = datasets.ImageFolder(path_to_in_domain, transform=TRANSFORM)
    in_data_loader = DataLoader(in_data, batch_size=BATCH_SIZE, shuffle=True)

    combined_data_loader = get_domain_data(path_to_in_domain, path_to_out_domain)

    dual_model = DualModel()
    dual_model.train_mode = True

    # train domain
    dual_model.train()
    dual_model.domain_mode = True

    optimizer = Adam(dual_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for e in range(DOMAIN_EPOCHS):
        running_loss = 0.
        for data in combined_data_loader:
            inputs, labels = data
            inputs = inputs.to(DEVICE)

            labels = labels.view(-1, 1)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = dual_model(inputs)

            loss = criterion(outputs.float(), labels.float())
            running_loss += loss.item()
            loss.backward()

            optimizer.step()

    # train prediction
    dual_model.train()
    dual_model.domain_mode = False

    optimizer = Adam(dual_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for e in range(PRED_EPOCHS):
        running_loss = 0.
        for data in in_data_loader:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = dual_model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
    return dual_model

def accuracy(path_to_eval_folder: str, dual_model: DualModel) -> float:
    """
    Takes a folder of images and a DualModel. Predicts the class of the objects in the images, and return an accuracy
    for how many were classified correctly.
    """
    data = datasets.ImageFolder(path_to_eval_folder, transform=TRANSFORM)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    dual_model.train_mode = False
    dual_model.eval()

    correct = 0
    total = 0

    with no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = dual_model(inputs)
            outputs = outputs.view(-1, 10)
            outputs = outputs.to(DEVICE)

            _, predicted = max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total
