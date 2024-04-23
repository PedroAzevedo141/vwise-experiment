import os
import sys
import torch
import numpy as np
from torchvision.models import mobilenet_v3_large

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader import CustomDataLoader


def train(Train_data, Val_data, device="cpu", patience=5):
    # Definir modelo
    model = mobilenet_v3_large(weights=None)
    model = model.to(device)

    # Definir função de perda e otimizador
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Criar um dicionário de dataloaders
    dataloaders = {"train": Train_data, "val": Val_data}

    # Inicializar a melhor perda como infinito
    best_loss = float("inf")

    # Inicializar contadores
    epochs_no_improve = 0

    # Listas para armazenar perdas
    train_losses = []
    val_losses = []

    for epoch in range(25):  # loop over the dataset multiple times
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # print statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print("Epoch: {} - {} Loss: {:.4f}".format(epoch, phase, epoch_loss))

            # save model if validation loss has decreased
            if phase == "val":
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    print(
                        "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                            best_loss, epoch_loss
                        )
                    )
                    torch.save(model.state_dict(), "model.pt")
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == patience:
                        print("Early stopping!")
                        # Load in the best model
                        model.load_state_dict(torch.load("model.pt"))
                        return model
            else:
                train_losses.append(epoch_loss)

    # Save losses to a .txt file
    with open("train_losses.txt", "w") as f:
        for item in train_losses:
            f.write("%s\n" % item)

    with open("val_losses.txt", "w") as f:
        for item in val_losses:
            f.write("%s\n" % item)

    return model


def test(Test_data, model=None, device="cpu"):
    # Carregar o modelo
    if model is None:
        model = mobilenet_v3_large(weights=None)
        model.load_state_dict(torch.load("model.pt"))
        model = model.to(device)
        model.eval()
    

    # Avaliar o modelo
    correct = 0
    total = 0
    with torch.no_grad():
        for data in Test_data:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))


if __name__ == "__main__":

    # Definir dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Definir o path para o dataset
    data_path = "./../data/custom/"

    # Carregar dados
    data_loader = CustomDataLoader(data_path, 0.7, 0.2, 0.1)
    Train_data = data_loader.get_train_data()
    Test_data = data_loader.get_test_data()
    Val_data = data_loader.get_val_data()

    model = train(Train_data, Val_data, device)
    test(Test_data, model, device)
