import os
import json
import numpy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import math
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from sklearn.metrics import classification_report

product_list = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

mapping_product = {}
for i, prod in enumerate(product_list):
    mapping_product[i] = prod

def get_submit_file(idx, output):
    with open('submission.csv', 'w') as f:
        f.write('ncodpers,added_products')
        f.write('\n')
        for id, item in enumerate(output):
            item_added = [mapping_product[i] for i, label in enumerate(item) if label == 1]
            f.write(f"""{idx[id]}, {' '.join(item_added)}\n""")
        f.close()

class SantanderDataset(Dataset):
    def __init__(self, X, y=None):
        self.trainset = torch.tensor(X)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.trainset[idx], self.label[idx].float()
    
class SantanderTestset(Dataset):
    def __init__(self, X):
        self.trainset = torch.tensor(X)
        self.y = torch.zeros(self.trainset.size(0))
        
    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.trainset[idx], self.y[idx]
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BST(pl.LightningModule):
    def __init__(self, num_embedding, num_feature_past, n_item=24):
        super().__init__()
        
        self.feature_dimmension = int(n_item//2) 
        self.embedding_matrix = torch.nn.ModuleList()
        
        for i in num_embedding:
            self.embedding_matrix.append(nn.Embedding(i, int(math.sqrt(i))+1))
        
        self.positional_encoding = PositionalEncoding(self.feature_dimmension)
        self.item_embedding = nn.Embedding(n_item+1, self.feature_dimmension)
        
        self.other_dimession = sum([int(math.sqrt(i))+1 for i in num_embedding])
        self.transformers = nn.TransformerEncoderLayer(d_model=self.feature_dimmension, nhead=4, dropout=0.2)
        
        self.out_tranformers_dimession = self.other_dimession + num_feature_past*self.feature_dimmension
        
        self.linear = nn.Sequential(
            nn.Linear(
                self.out_tranformers_dimession,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_item),
        )
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.validation_step_outputs = []
        self.test = []
        self.eval_result, self.eval_label = [], []
        self.save_hyperparameters()
    def forward(self, batch):
        X, y = batch        
        other_feature, item_feature = X[:,:-20], X[:,-20:]
        expand_other_feature =  torch.cat([e_m(other_feature[:,i]) for i, e_m in enumerate(self.embedding_matrix)], dim=1)

        item_feature = self.item_embedding(item_feature)

        item_feature = self.positional_encoding(item_feature)

        transformer_output =  torch.flatten(self.transformers(item_feature), start_dim=1)
        feature = torch.cat((transformer_output, expand_other_feature),dim=1)
        
        output = self.linear(feature)
        output = self.sigmoid(output)
        return output, y
    
    def training_step(self, batch, batch_idx):
        out, target_purchase = self(batch)
        loss = self.bce(out, target_purchase)
        mse = self.mse(out, target_purchase)
        mae = self.mae(out, target_purchase)
        rmse = torch.sqrt(mse)
        self.log(
            "train/mae", mae, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        
        self.log(
            "train/rmse", rmse, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log("train/step_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out, target_movie_rating = self(batch)
        loss = self.bce(out, target_movie_rating)
        
        batch_result = out.tolist()
        
        mae = self.mae(out, target_movie_rating)
        mse = self.mse(out, target_movie_rating)
        rmse = torch.sqrt(mse)
        loss_dict = {"val_loss": loss, "mae": mae.detach(), "rmse": rmse.detach()}
        
        self.validation_step_outputs.append(loss_dict)
        
        self.eval_label.extend(target_movie_rating.cpu())
        self.eval_result.extend([[1 if i>0.5 else 0 for i in j] for j in batch_result])

        return loss_dict

    def on_validation_epoch_end(self):
        print('***Evaluate on validation set***')
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        print('\n')
        print("*"*20)
        print(f'Result on validation set:\nLoss:{avg_loss}\nMAE:{avg_mae}\nRMSE:{avg_rmse}')
        print("*"*20)
        self.log("val/loss", avg_loss)
        self.log("val/mae", avg_mae)
        self.log("val/rmse", avg_rmse)
        self.validation_step_outputs.clear()
        print(classification_report(self.eval_label, self.eval_result))
        self.eval_label.clear()
        self.eval_result.clear()
        
    def test_step(self, batch, batch_idx):
        out, _ = self(batch)
        batch_result = out.tolist()  ###size (batch_size, result)
        self.test.extend([[1 if i>0.5 else 0 for i in j] for j in batch_result])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)
    
if __name__ == "__main__": 
    X, y = np.load('data/X_train.npy'), np.load('data/y_train.npy')
    X_test, X_idx = np.load('data/X_test.npy'), np.load('data/test_idx.npy')
    
    X_array = np.array(X)
    
    num_embedding = [max(np.unique(X_array[:,i]))+1 for i in range(X_array.shape[1]-20)]

    dataset = SantanderDataset(X, y)

    generator = torch.Generator().manual_seed(42)
    train, val = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train, batch_size=512, shuffle=True)
    val_loader = DataLoader(val, batch_size=512, shuffle=True)

    bst = BST(num_embedding=num_embedding, num_feature_past=20)

    callbacks = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='val/loss', dirpath= 'model_ckpt/',
                                                                                mode='min',
                                                                                every_n_epochs=1,
                                                                                auto_insert_metric_name=False,
                                                                                verbose=True,
                                                                                save_top_k=1)
    trainer = pl.Trainer( 
                         max_epochs=5,
                         callbacks = [callbacks,]
                        )
    trainer.fit(model=bst, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint("example.ckpt")

    bst = BST(num_embedding=num_embedding, num_feature_past=20)
    
    for file in os.listdir('model_ckpt/'):
        if file.endswith('.ckpt'):
            break
            
    bst = bst.load_from_checkpoint('model_ckpt/'+ file)
    
    test_set = DataLoader(SantanderTestset(X_test), batch_size=256, shuffle=False)
    trainer.test(bst, test_set)

    get_submit_file(X_idx, bst.test)