import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time


# Define the model
class SkinCancerModel(nn.Module):
    def __init__(self, model_option, loss_option, optimizer_option, model_save_path = 'skin_cancer_model', specified_weights=None):
        super(SkinCancerModel, self).__init__()
        self.model_option = model_option
        self.loss_option = loss_option
        self.optimizer_option = optimizer_option

        self.model = self._get_model()
        # Change the output layer to have 1 output
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        # if specified_weights:
        #     self.model.load_state_dict(torch.load(specified_weights))

        self.criterion = self._get_loss()
        self.optimizer = self._get_optimizer()
    
    def _get_model(self):
        if self.model_option == "resnet18":
            return models.resnet18(pretrained=True)
        elif self.model_option == "resnet34":
            return models.resnet34(pretrained=True)
        elif self.model_option == "resnet50":
            return models.resnet50(pretrained=True)

    
    def _get_loss(self):
        if self.loss_option == "BCELoss":
            return nn.BCELoss()
        elif self.loss_option == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.loss_option == "MSELoss":
            return nn.MSELoss()
        elif self.loss_option == "NLLLoss":
            return nn.NLLLoss()
        elif self.loss_option == "SmoothL1Loss":
            return nn.SmoothL1Loss()
    
    def _get_optimizer(self):
        if self.optimizer_option == "Adam":
            return optim.Adam(self.model.parameters(), lr=0.001)
        elif self.optimizer_option == "SGD":
            return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        elif self.optimizer_option == "RMSprop":
            return optim.RMSprop(self.model.parameters(), lr=0.001)
        elif self.optimizer_option == "Adagrad":
            return optim.Adagrad(self.model.parameters(), lr=0.01)
        elif self.optimizer_option == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=0.001)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))
    

# Train the model
def train_model(model, train_loader, test_loader,  epochs=10, save_period=10, model_save_path='skin_cancer_model'):
    train_losses = []
    test_accuracies = []

    criterion = model.criterion
    optimizer = model.optimizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        batch_idx = 0
        for inputs, labels in train_loader:
            batch_idx += 1
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # breakpoint()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # breakpoint()
            if (batch_idx + 1) % (len(train_loader) // (len(train_loader)*.2)) == 0:
                percent_done = (batch_idx + 1) / len(train_loader) * 100
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}, Percent Done: {percent_done:.2f}%, Time Elapsed: {time.time() - start_time:.2f} seconds")
        
        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        if (epoch + 1) % save_period == 0:
            save_model(model, f"{model_save_path}_save{epoch+1}.pt")
            print(f"----------Model saved at epoch {epoch+1}-----------")
        
        # Evaluate on test set
        test_accuracy, correctly_classified, incorrectly_classified = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)
    
    return train_losses, test_accuracies, correctly_classified, incorrectly_classified

# Evaluate the model
def evaluate_model(model, test_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), eval_only=False):
    if eval_only:
        model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    correctly_classified = []
    incorrectly_classified = []
    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)

            # breakpoint()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correctly_classified.extend(inputs[(predicted == labels).squeeze()])
            incorrectly_classified.extend(inputs[(predicted != labels).squeeze()])
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy, correctly_classified, incorrectly_classified

# Ensemble evaluation NOT TESTED
def ensemble_evaluation(models, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        model.to(device)
        model.eval()

    ensemble_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            predictions = []
            for model in models:
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                predictions.append(predicted)
            ensemble_predictions.append(predictions)

    ensemble_predictions = torch.stack(ensemble_predictions)
    ensemble_statistics = torch.mean(ensemble_predictions, dim=0)

    return ensemble_statistics


# Save the model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model