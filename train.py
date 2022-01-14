import copy
import os
from time import time

from dvclive import Live
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from ruamel.yaml import YAML
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection

# Initialize DVCLive
dvclive = Live()

# Where the data comes from
data_dir = "./cats-dogs"

# Load params
with open("params.yaml") as f:
    yaml = YAML(typ='safe')
    params = yaml.load(f)

# Model that we want to use from these options: [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = params["model_name"]

# False to fine-tune, True for feature extraction
feature_extract = True

# Detect if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=2, is_inception=False):
    since = time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    total = dvclive.get_step() + num_epochs
    for epoch in range(dvclive.get_step(), total):
        print(f"Epoch {epoch}/{total - 1}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        output, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            epoch_time_elapsed = time() - since

            if phase == 'train':
                torch.save(model.state_dict(), "model.pt")

                dvclive.log('acc', epoch_acc.item())
                dvclive.log('loss', epoch_loss)
                dvclive.log('training_time', epoch_time_elapsed)

            if phase == 'val':
                dvclive.log('val_acc', epoch_acc.item())
                dvclive.log('val_loss', epoch_loss)

                val_acc_history.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        dvclive.next_step()
        print()

    time_elapsed = time() - since

    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    else:
        raise ValueError('Model name must be "alexnet" or "squeezenet"')

    return model_ft, input_size


def train():
    # Initialize model for this run
    model_ft, input_size = initialize_model(model_name, params['num_classes'], feature_extract, use_pretrained=True)

    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # TODO: Add data drift tracking with Evidently
    # animals_data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    # animals_data_drift_dashboard.calculate(image_datasets, image_datasets, column_mapping = None)
    # animals_data_drift_dashboard.show()

    # animals_data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    # animals_data_drift_profile.calculate(image_datasets, image_datasets, column_mapping = None)
    # animals_data_drift_profile.json()

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=params["batch_size"], shuffle=True, num_workers=4) for x in ['train', 'val']}

    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=params['lr'], momentum=params['momentum'])

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=params['num_epochs'], is_inception=(model_name=="inception"))


if __name__ == "__main__":
    train()
