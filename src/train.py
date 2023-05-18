import torch.nn as nn
from src.model import MobileNetV1, Network, MobileNetV2
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
import hydra
from omegaconf import DictConfig, OmegaConf

def train(train_set, cfg, in_channels = 3, num_classes = 10):

    loss_function = nn.CrossEntropyLoss()
    
    network = Network(in_channels = in_channels, num_classes = num_classes)

    network.train()

    if cfg['show_model_summary']:
        summary(network, (in_channels,224,224))

    optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)

    train_dataloader = DataLoader(train_set, batch_size=20)
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:
                # print (img.shape)
                optimizer.zero_grad() 
                out = network(imgs.to(device))
                loss = loss_function(out, labels.to(device))
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
    print("training done")
    torch.save(network, cfg['save_model_path'])

    return network

if __name__ == "__main__":

    torch.manual_seed(42)

    from src.dataset import get_load_data
    cfg = {"save_model_path": "model_weights/model_weights.pt",
           "epochs": 10}
    train_set, test_set = get_load_data(root = "../data", dataset = "Flowers102")
    train(train_set = train_set, cfg = cfg, in_channels = 3, num_classes = 102)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    