from torchvision import transforms
from utils.dataset import KaggleDataset
from torch.utils.data import DataLoader
import torch
import argparse
from models import baseline
import pandas as pd


def make_predictions(data_dir, out_dir, model_path):

    mapping = {0: 'Cat', 1: 'Dog'}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop(60),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    model = torch.load(model_path, map_location=device)
    model.eval()
    dataset = KaggleDataset(data_dir='../data/testset/',
                            transform=transform)

    predictions = []
    ids = []
    test_loader = DataLoader(dataset,
                             batch_size=128)

    for sample in test_loader:

        inputs, id = sample['image'], sample['id']
        # Move to device
        inputs = inputs.to(device)
        predictions.append(model(inputs.to(device)).argmax(dim=1))
        ids.append(id)

    predictions = torch.cat(predictions).cpu().numpy()
    ids = torch.cat(ids).cpu().numpy()

    results = pd.DataFrame(data={'id': ids, 'label': [mapping[p]
                                                      for p in predictions]}
                           )
    results.index.name = 'id'
    results.to_csv(out_dir+'results.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/testset')
    parser.add_argument('--out_dir', default='Submissions/')
    parser.add_argument(
        '--model_path', default='results/base_wr_lr01best_model.pth')
    args = parser.parse_args()
    make_predictions(args.data_dir, args.out_dir, args.model_path)
