import os
import torch
from data import get_dataloader
from model import MaskRCNNModel
from evaluation import validate
from train import train, get_optimizer
from train import set_seed
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

BATCH_SIZE = 1
EPOCHS = 100
PATIENCE = 20
NAMING = "src_a12"


def main(seed=77):
    set_seed(seed)
    device = torch.device('cuda') \
        if torch.cuda.is_available() \
        else torch.device('cpu')

    root = '../dataset/train'
    all_folders = [os.path.join(root, d) for d in os.listdir(root)]

    train_folders, valid_folders = train_test_split(all_folders,
                                                    test_size=0.1,
                                                    random_state=seed)

    train_loader, valid_loader = get_dataloader(
        train_folders=train_folders,
        valid_folders=valid_folders,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    model = MaskRCNNModel(num_classes=5, pretrained=True).to(device)
    scaler = torch.amp.GradScaler()
    optimizer = get_optimizer(model)

    no_improvement_epochs = 0
    train_losses = []
    valid_maps = []
    max_map = 0.0

    for epoch in range(EPOCHS):
        train_loss = train(model, optimizer, train_loader, device, scaler)
        valid_map = validate(model, valid_loader, device)

        train_losses.append(train_loss)
        valid_maps.append(valid_map)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation mAP: {valid_map:.4f}")

        no_improvement_epochs += 1
        if valid_map > max_map:
            print(f"Saving model, Best mAP: {valid_map:.4f}")
            torch.save(model.state_dict(),
                       f'../models/{NAMING}_seed{seed}.pth')
            max_map = valid_map
            no_improvement_epochs = 0

        if no_improvement_epochs >= PATIENCE:
            print("Early stopping")
            break

    print(f"train_losses: {train_losses}")
    print(f"mAP: {valid_maps}")


if __name__ == "__main__":
    main(seed=77)
