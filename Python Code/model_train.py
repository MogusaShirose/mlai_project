import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# Define transformations for the training data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Define transformations for the validation data (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths to your datasets with forward slashes
train_data_path = 'C:/Users/seanh/Downloads/MLAI Project/datasets/train_dataset'
val_data_path = 'C:/Users/seanh/Downloads/MLAI Project/datasets/validation_dataset'

# Create datasets
train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)
val_dataset = datasets.ImageFolder(val_data_path, transform=val_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Define classes based on folder names
classes = train_dataset.classes

# Define the CNN model
class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        self.features = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1') # Use weights parameter instead of pretrained
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

def main():
    # Initialize the model
    model = MyCNN(num_classes=len(classes))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += inputs.size(0)
        
        # Print statistics
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds
        val_epoch_loss = val_loss / len(val_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, '
              f'Val Acc: {epoch_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'C:/Users/seanh/Downloads/model.pth')
    print('Training complete. Model saved.')

if __name__ == '__main__':
    main()
