import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from src.utils import AdaptiveScheduler, NSELoss, sample, nKLDivLoss


class MLP(nn.Module):
    """
    Generic MLP with Dropout
    Last layer is softmax to compute attention scores
    """
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidd_layers,
                 act = nn.LeakyReLU(),
                 drop_p = 0.4):
    
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidd_layers = hidd_layers
        self.act = act
        self.drop_p = drop_p

        ### network architecture
        layers = [nn.Linear(self.input_dim, self.hidd_layers[0])]
        batch_norms = [nn.BatchNorm1d(self.hidd_layers[0])]
        for i in range(len(hidd_layers)-1):
            layers.append(nn.Linear(self.hidd_layers[i], self.hidd_layers[i+1]))
            batch_norms.append(nn.BatchNorm1d(self.hidd_layers[i+1]))
           
        self.out = nn.Linear(self.hidd_layers[-1], self.output_dim)
        self.layers = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.dropout = nn.Dropout(self.drop_p, inplace = False)
        self.out_bn = nn.BatchNorm1d(self.output_dim)
        # initialize weights
        self._initialize_weights()

        print("MLP initialized")
        # for e, layer in enumerate(self.layers):
        #     print(f"Weight/bias layer {e}", layer.weight,layer.bias)

        # print(f"Weight/bias out layer", self.out.weight,self.out.bias)

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            #torch.nn.init.normal_(layer.weight, mean=0,std=1e-4)
            torch.nn.init.normal_(layer.bias, mean=0.0, std=1e-4)

        # set first hidden layer out to be the identities
        #nn.init.eye_(self.layers[0].weight)
        #self.layers[0].weight = nn.Parameter(self.layers[0].weight.add(torch.randn_like(self.layers[0].weight)*1e-4))
        #nn.init.eye_(self.out.weight)
        #self.out.weight = nn.Parameter(self.out.weight.add(torch.randn_like(self.out.weight)*1e-4))
        nn.init.kaiming_normal_(self.out.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.normal_(self.out.bias, mean=0.0, std=1e-4)

    def forward(self, x):
        # comment first act and batch norm layers in enca_26 for runs 4,5
        # pass first layer
        x = self.layers[0](x)
        x = self.act(x)
        x = self.batch_norms[0](x)
        x = self.dropout(x)
        #print(torch.mean(x, dim=0),torch.std(x, dim=0) )
        
        # pass the network
        for i in range(1,len(self.layers)):
            residual = x
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.act(x)
            x = self.dropout(x)
            #print(torch.mean(x, dim=0),torch.std(x, dim=0) )
            x = (x + residual) / torch.sqrt(torch.tensor(2.0, device=x.device))
        
        x = self.out(x)
        x = self.out_bn(x)
        
        return x
    

class MLP_Linear(nn.Module):
    """
    Generic MLP with Dropout
    Last layer is softmax to compute attention scores
    """
    def __init__(self, 
                 input_dim,
                 output_dim):
    
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
       
        ### network architecture
        self.out = nn.Linear(self.input_dim, self.output_dim)
        
        # initialize weights
        self._initialize_weights()

        print("MLP initialized")
        # for e, layer in enumerate(self.layers):
        #     print(f"Weight/bias layer {e}", layer.weight,layer.bias)

        # print(f"Weight/bias out layer", self.out.weight,self.out.bias)

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.out.weight, mode='fan_in', nonlinearity='linear')
        torch.nn.init.normal_(self.out.bias, mean=0.0, std=1e-4)

    def forward(self, x):
        # comment first act and batch norm layers in enca_26 for runs 4,5
        x = self.out(x)
        
        return x

class Hydro_Attention(pl.LightningModule):
    """
    Autoencoder with a MLP encoder and a LSTM decoder
    """
    def __init__(self,
                 input_dim,
                 output_dim, 
                 hidden_layers,
                 lstm_hidden_units = 256, 
                 initial_forget_bias = 3,
                 act = nn.LeakyReLU(), 
                 drop_p = 0.4, 
                 seq_length = 365,
                 lr = 1e-5,
                 weight_decay = 0.0,
                 milestones = {0 : 1e-3},
                ):
          
        
        super().__init__()
        self.save_hyperparameters(ignore=['act']) # save hyperparameters for chekpoints
        
        # Parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.initial_forget_bias = initial_forget_bias
        self.drop_p = drop_p
        self.seq_length = seq_length
        self.lstm_hidden_units = lstm_hidden_units
       
    
          
        ### LSTM decoder
        if self.output_dim == None: # 15 forcing + input_dim attributes
            self.lstm = nn.LSTM(input_size= 15 + self.input_dim, 
                           hidden_size=self.lstm_hidden_units,
                           num_layers=1,
                           batch_first=True,
                          bidirectional=False)
        elif self.output_dim >= 0: # 15 forcing only + output_dim encoded attributes
            if self.output_dim > 0:
                if len(hidden_layers) == 0:
                    self.encoder = MLP_Linear(input_dim, output_dim)
                else:
                    self.encoder = MLP(input_dim, output_dim, hidden_layers, act, drop_p)
        
            self.lstm = nn.LSTM(input_size= 15 + self.output_dim, 
                           hidden_size=self.lstm_hidden_units,
                           num_layers=1,
                           batch_first=True,
                          bidirectional=False)
        else:
            raise ValueError("output_dim should be either None, or greater equal than 0")
        
        # reset weigths
        self._initialize_weights()

        # out layer
        self.dropout = nn.Dropout(drop_p, inplace = False)
        self.out = nn.Linear(lstm_hidden_units, 1)
        print("Attention LSTM initialized")

    def _initialize_weights(self):
        # set forget bias to specified value (as Kratzert et al., 2021)
        if self.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.lstm_hidden_units:2 * self.lstm_hidden_units] = self.initial_forget_bias


    def forward(self, x, attr, random_features=None):
        enc = None
        if self.output_dim == None:
            attr_expanded = attr.unsqueeze(1).expand(-1, self.seq_length, -1) # expand dimension
            x = torch.cat((x, attr_expanded),dim=-1) # concat data

        elif self.output_dim > 0:
            if random_features is not None:
                if random_features.shape != (self.output_dim,):
                    raise ValueError(
                        "random_features must have shape "
                        f"({self.output_dim},), got {tuple(random_features.shape)}"
                    )
                # Static attributes must remain constant across all windows in a basin.
                enc = random_features.to(device=attr.device, dtype=attr.dtype)
                enc = enc.unsqueeze(0).expand(attr.shape[0], -1)
            else:
                enc = self.encoder(attr) # shape (batch_size, output_dim)
            enc_expanded = enc.unsqueeze(1).expand(-1, self.seq_length, -1) # expand dimension
            x = torch.cat((x, enc_expanded),dim=-1) # concat data
        
        # LSTM layers
        x, recurrent = self.lstm(x)
        x = self.dropout(x[:,-1,:]) # take the last value predicted
        rec = self.out(x)
        
        return enc, rec, recurrent
        
    def training_step(self, batch, batch_idx):        
        x, attr, y, q_stds = batch # Unpack batch
        _, rec, _ = self.forward(x, attr.squeeze())# forward pass

        # Logging to TensorBoard by default
        train_loss = NSELoss(rec, y, q_stds)
        
        self.log("train_loss", train_loss, on_step=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):        
        x, attr, y, q_stds = batch # Unpack batch
        _, rec, _ = self.forward(x, attr.squeeze())# forward pass

        # Logging to TensorBoard by default
        val_loss = NSELoss(rec, y, q_stds)
        self.log("val_loss", val_loss, on_step=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        #print("Parsms group:", optimizer.param_groups)
        self.lr_scheduler = AdaptiveScheduler(optimizer, milestones=self.milestones)
        #print("Last lr scheduler:", self.lr_scheduler.get_last_lr())
        return {"optimizer":optimizer, "lr_scheduler":self.lr_scheduler}


    def on_train_epoch_end(self):
        # Retrieve the optimizer
        optimizer = self.optimizers()
        
        # Log the learning rate for each param group
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log(f'lr_param_group_{i}', lr, prog_bar=True)
            print(f"Learning rate for param group {i}: {lr}")
