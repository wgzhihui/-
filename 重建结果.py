import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import nibabel as nib
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt


# 定义更复杂的超分辨率模型，例如改进的SRCNN
class ImprovedSRCNN(nn.Module):
    def __init__(self):
        super(ImprovedSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x


# 数据集根目录，根据你的要求修改
data_root = "D:\桌面\IXI-TT1"

# 获取所有图像文件路径
image_paths = []
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            image_paths.append(os.path.join(root, file))

# 划分数据集
random.shuffle(image_paths)
train_size = int(0.7 * len(image_paths))
val_size = int(0.15 * len(image_paths))
test_size = len(image_paths) - train_size - val_size
train_paths = image_paths[:train_size]
val_paths = image_paths[train_size:train_size + val_size]
test_paths = image_paths[train_size + val_size:]


# 自定义数据加载函数，用于加载NIFTI格式的医学图像，并添加去噪处理
def load_nii_image(path):
    img = nib.load(path)
    img_data = img.get_fdata()
    print(f"Loaded img_data: type {type(img_data)}, shape {img_data.shape}")
    # 假设图像是三维的，取其中一个维度作为图像平面（例如取中间切片）
    img_slice = img_data[:, :, img_data.shape[2] // 2]
    if not isinstance(img_slice, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(img_slice)} in load_nii_image")
    # 使用中值滤波去噪
    from scipy.ndimage import median_filter
    img_slice = median_filter(img_slice, size=3)
    print(f"Returned img_slice: type {type(img_slice)}, shape {img_slice.shape}")
    return img_slice


# 数据预处理（与训练时一致），调整归一化方式
def preprocess(x):
    print(f"Input to preprocess: type {type(x)}, shape {x.shape if hasattr(x,'shape') else None}")
    if not isinstance(x, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(x)}")
    try:
        x = torch.from_numpy(x).unsqueeze(0).float()
        # 归一化到0 - 1范围
        x = x / 255.0
    except ValueError as e:
        print(f"Error in converting to tensor: {e}")
        raise
    assert isinstance(x, torch.Tensor), f"Expected tensor, got {type(x)} after conversion"
    return x


def custom_collate_fn(batch):
    new_batch = []
    for item in batch:
        if isinstance(item, tuple):
            for sub_item in item:
                if isinstance(sub_item, torch.Tensor):
                    new_batch.append(sub_item)
                else:
                    raise ValueError(f"Unexpected non - tensor item {type(sub_item)} in batch")
        elif isinstance(item, torch.Tensor):
            new_batch.append(item)
        else:
            raise ValueError(f"Unexpected non - tensor or non - tuple item {type(item)} in batch")
    try:
        return torch.stack(new_batch)
    except RuntimeError as e:
        print(f"Error in custom_collate_fn: {e}")
        raise


# 加载并预处理训练集数据
train_data = []
for path in train_paths:
    img = load_nii_image(path)
    img_tensor = preprocess(img)
    if not isinstance(img_tensor, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(img_tensor)} for path {path}")
    train_data.append(img_tensor)
for i, data in enumerate(train_data):
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Non - tensor data at index {i} in train_data. Type: {type(data)}")
tensor_shapes = [data.shape for data in train_data]
if len(set(tensor_shapes)) > 1:
    print(f"Tensor shapes in train_data are inconsistent: {tensor_shapes}")
    raise ValueError("Inconsistent tensor shapes in train_data.")
train_data = torch.stack(train_data)
train_dataset = torch.utils.data.TensorDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)


# 加载并预处理验证集数据
val_data = []
for path in val_paths:
    img = load_nii_image(path)
    img_tensor = preprocess(img)
    if not isinstance(img_tensor, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(img_tensor)} for path {path}")
    val_data.append(img_tensor)
for i, data in enumerate(val_data):
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Non - tensor data at index {i} in val_data. Type: {type(data)}")
tensor_shapes = [data.shape for data in val_data]
if len(set(tensor_shapes)) > 1:
    print(f"Tensor shapes in val_data are inconsistent: {tensor_shapes}")
    raise ValueError("Inconsistent tensor shapes in val_data.")
val_data = torch.stack(val_data)
val_dataset = torch.utils.data.TensorDataset(val_data)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)


# 定义损失函数和优化器
criterion = nn.MSELoss()
model = ImprovedSRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 训练模型
num_epochs = 5
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(data)} at iteration {i} in training")
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # 在验证集上评估模型
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            if not isinstance(val_data, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(val_data)} at iteration {i} in validation")
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_data)
            val_running_loss += val_loss.item()

    val_epoch_loss = val_running_loss / len(val_loader)
    val_losses.append(val_epoch_loss)

    scheduler.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')


# 保存训练好的模型
torch.save(model.state_dict(),'model.pth')


# 加载训练好的模型
model = ImprovedSRCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()


# 保存重建图像的目录
save_dir ='super_resolution_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


with torch.no_grad():
    for i, path in enumerate(test_paths):
        # 加载低分辨率测试集图像
        low_res_img = load_nii_image(path)
        low_res_tensor = preprocess(low_res_img)
        # 进行超分辨率重建
        super_res_tensor = model(low_res_tensor.unsqueeze(0))
        super_res_tensor = super_res_tensor.clamp(0, 1)
        super_res_img = super_res_tensor.squeeze(0).squeeze(0).cpu().numpy()
        # 将重建后的图像转换为PIL Image并保存为图片格式
        super_res_img = (super_res_img * 255).astype(np.uint8)
        super_res_pil = Image.fromarray(super_res_img)
        save_path_sr = os.path.join(save_dir, f'result_sr_{i}.png')
        super_res_pil.save(save_path_sr)

        # 保存原始低分辨率图像
        low_res_img = (low_res_img * 255).astype(np.uint8)
        low_res_pil = Image.fromarray(low_res_img)
        save_path_lr = os.path.join(save_dir, f'result_lr_{i}.png')
        low_res_pil.save(save_path_lr)

        # 可视化对比
        try:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(low_res_img, cmap='gray')
            plt.title('Low Resolution Image')
            plt.subplot(1, 2, 2)
            plt.imshow(super_res_img, cmap='gray')
            plt.title('Super Resolution Image')
            plt.show()
        except ImportError:
            print("Matplotlib not installed, cannot display image.")
