import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# Load your trained model
model_path = 'C:/Users/seanh/Downloads/MLAI Project/models/model.pth' 

# Define the same model architecture as during training
class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        self.features = models.resnet18(pretrained=False)
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

# Create an instance of your model
model = MyCNN(num_classes=3)  # Assuming 3 classes: Apple, Strawberry, Unknown
model.load_state_dict(torch.load(model_path))
model.eval()

# Define transformations for the camera input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(rgb_frame)
    
    # Apply transformations
    img_tensor = transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        top3_prob, top3_idx = torch.topk(probabilities, k=3, dim=1)
    
    # Display labels and confidences
    label_texts = []
    for i in range(3):
        class_index = top3_idx[0][i].item()
        confidence_score = top3_prob[0][i].item()
        
        # Determine class label
        if class_index == 0:
            label = "Apple"
        elif class_index == 1:
            label = "Strawberry"
        else:
            label = "Unknown"
        
        # Format label and confidence
        label_text = f'{label}: {confidence_score:.2f}'
        label_texts.append(label_text)
    
    # Display labels on the frame
    for idx, text in enumerate(label_texts):
        cv2.putText(frame, text, (50, 50 + idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Camera Feed', frame)
    
    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
