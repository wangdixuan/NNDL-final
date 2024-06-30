import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
num_epochs = 100

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model = resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)  # CIFAR-100有100个类别
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

writer = SummaryWriter('logs/runs_cifar100_resnet18')

best_accuracy = 0.0

for epoch in tqdm(range(num_epochs)):
    # 训练阶段
    model.train()
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    scheduler.step()
    
    avg_loss_train = running_loss_train / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    
    # 在TensorBoard中单独记录训练损失
    writer.add_scalar('Train/Loss', avg_loss_train, epoch)
    # 在TensorBoard中单独记录训练准确率
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss_train:.4f}, Train Acc: {train_accuracy:.2f}%')
    
    # 验证阶段
    model.eval()
    running_loss_val = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
        avg_loss_val = running_loss_val / len(test_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # 在TensorBoard中单独记录验证损失
        writer.add_scalar('Val/Loss', avg_loss_val, epoch)
        # 在TensorBoard中单独记录验证准确率
        writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
        
        print(f'Validation Loss: {avg_loss_val:.4f}, Validation Acc: {val_accuracy:.2f}%')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), '/root/hykuang/NNDL/model/cifar100_resnet18.pth')
            
writer.close()