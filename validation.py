import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_val_dataloader(val_dir, batch_size=4, num_workers=4):
    """Get validation dataloader.

    Args:
        val_dir (str): Directory of validation dataset.
        batch_size (int): Batch size for dataloader (default: 4).
        num_workers (int): Number of workers for dataloader (default: 4).

    Returns:
        DataLoader: Dataloader for validation dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(360),
        transforms.CenterCrop(324),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(val_dir, transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return dataloader


def test_time_augmentation(model, inputs, num_augmentations=4):
    """Perform test-time augmentation on inputs.

    Args:
        model (nn.Module): The neural network model.
        inputs (torch.Tensor): Input tensor to augment.
        num_augmentations (int): Number of augmentations to apply (default: 4).

    Returns:
        torch.Tensor: Weighted averaged outputs after augmentation.
    """
    def denormalize(tensor, mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # Denormalize
        return tensor

    def normalize(tensor, mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)  # Normalize
        return tensor

    model.eval()
    aug_trans = [
        transforms.RandomVerticalFlip(p=0.0),
        transforms.RandomRotation(30),
        transforms.RandomRotation(150),
        transforms.RandomRotation(270),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0)
    ]
    weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.005]

    batch_outputs = []
    with torch.no_grad():
        for i in range(num_augmentations):
            aug_inputs = denormalize(inputs.clone())  # Denormalize
            aug_idx = i % len(aug_trans)
            aug_inputs = aug_trans[aug_idx](aug_inputs)  # Apply augmentation
            aug_inputs = normalize(aug_inputs)  # Re-normalize
            outputs = model(aug_inputs)
            batch_outputs.append(outputs * weights[i])  # Weighted output

    return torch.mean(torch.stack(batch_outputs), dim=0)


if __name__ == '__main__':
    data_root = "./dataset/data/validation"  # Modify to your dataset path
    test_dir = os.path.join(data_root, "")
    test_loader = get_val_dataloader(test_dir)

    # Dictionary order class names
    class_names_dict_order = [
        '0', '1', '10', '11', '12', '13', '14', '15',
        '16', '17', '18', '19', '2', '20', '21', '22',
        '23', '24', '25', '26', '27', '28', '29', '3',
        '30', '31', '32', '33', '34', '35', '36', '37',
        '38', '39', '4', '40', '41', '42', '43', '44',
        '45', '46', '47', '48', '49', '5', '50', '51',
        '52', '53', '54', '55', '56', '57', '58', '59',
        '6', '60', '61', '62', '63', '64', '65', '66',
        '67', '68', '69', '7', '70', '71', '72', '73',
        '74', '75', '76', '77', '78', '79', '8', '80',
        '81', '82', '83', '84', '85', '86', '87', '88',
        '89', '9', '90', '91', '92', '93', '94', '95',
        '96', '97', '98', '99'
    ]

    # Numerical order class names
    class_names_num_order = sorted(class_names_dict_order,
                                   key=lambda x: int(x))

    # Create index mapping (dict order index -> num order index)
    index_mapping = {
        class_names_dict_order.index(name): class_names_num_order.index(name)
        for name in class_names_dict_order
    }

    print(index_mapping)  # Test output mapping

    device = 'cuda:0'
    class_names = [x for x in range(100)]

    model_ft = models.resnext101_64x4d(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, len(class_names)),
                                nn.LogSoftmax())

    model_weight_name = 'cool_model'
    model_ft.load_state_dict(
        torch.load(
            os.path.join('./models', model_weight_name + '.pth'),
            weights_only=True
        )
    )
    model_ft.eval()
    model_ft = model_ft.to(device)

    # Define loss function
    criterion = nn.NLLLoss()

    # For calculating overall statistics
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Map true labels to numerical order
            mapped_labels = torch.tensor([index_mapping[label.item()]
                                          for label in labels]).to(device)

            outputs = test_time_augmentation(model_ft, images)
            loss = criterion(outputs, mapped_labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            # Correct prediction indices
            corrected_preds = torch.tensor(
                [index_mapping[pred.item()] for pred in preds.cpu().numpy()]
            ).to(device)

            total += labels.size(0)
            correct += (corrected_preds == mapped_labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
