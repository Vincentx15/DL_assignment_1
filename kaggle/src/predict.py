from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import argparse
from models import baseline


def make_predictions(data_dir, out_dir, model_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4897, 0.4547, 0.4160),
                             std=(0.25206208, 0.24510874, 0.24726304))
    ])

    model = torch.load(model_path, map_location=device)
    model.eval()
    dataset = ImageFolder(root='../data/testset',
                          transform=transform)

    predictions = []
    test_loader = DataLoader(dataset,
                             batch_size=128)

    for inputs, _ in test_loader:
        predictions.append(model(inputs.to(device)).argmax(dim=1))
        print(predictions)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/testset')
    parser.add_argument('--out_dir', default='Submissions/')
    parser.add_argument(
        '--model_path', default='results/base_wr_lr01_wflipbest_model.pth')
    args = parser.parse_args()
    make_predictions(args.data_dir, args.out_dir, args.model_path)
