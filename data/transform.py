from torchvision import transforms


def get_image_transform(img_size):
    return transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),  # 转为张量，范围 [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到 [-1, 1]
    ])

def get_label_transform(img_size):
    return transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.PILToTensor()  # 转为张量，范围 [0, 255]
    ])
