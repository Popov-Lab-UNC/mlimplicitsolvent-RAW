from tqdm import tqdm
from config import CONFIG
import terrace as ter
from datasets.bigbind_solv import BigBindSolvDataset

def train():

    train_dataset = BigBindSolvDataset("train")
    train_loader = ter.DataLoader(train_dataset,
                                  batch_size=CONFIG.batch_size,
                                  shuffle=True)
    
    for batch in tqdm(train_loader):
        pass

if __name__ == "__main__":
    train()