import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import timeit


class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, save_name,mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])

        self.save_name = save_name

        if num_pred_classes==2:
            self.final = nn.Linear(mid_channels * 2, 1)
        else:
            self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))

    def fit(self, train_dataloader, test_dataloader ,num_epochs=100,
            learning_rate=0.001,patience=10,save_best_model=True):

        # useful variables
        patience_counter = 0
        val_acc_hist = []
        train_loss_hist = []

        # optimizer and loss
        optimizer=torch.optim.RMSprop(self.parameters(),lr=learning_rate,weight_decay=0.0001,momentum=0.9)
        binary_case = (self.input_args['num_pred_classes']==2)
        if binary_case:
            loss_fn = nn.BCEWithLogitsLoss( reduction='mean')
        else:
            loss_fn = nn.CrossEntropyLoss( reduction='mean')

        for current_epoch in range(num_epochs):
            epoch_train_loss = []
            # train loop
            for  X_train,y_train in train_dataloader:
                optimizer.zero_grad()
                train_output = self(X_train)
                train_loss = loss_fn(train_output, y_train.float().reshape(-1,1)) if binary_case else  loss_fn(train_output, y_train)
                epoch_train_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
            train_loss_hist.append(np.mean(epoch_train_loss))


            epoch_val_loss = []
            #self.model.eval()
            true_list = []
            pred_list = []

            # validation step
            for X_val,y_val in test_dataloader:
                with torch.no_grad():
                    val_output = self(X_val)
                    val_loss = loss_fn(val_output, y_val.float().reshape(-1,1)) if binary_case else loss_fn(val_output,y_val)
                    epoch_val_loss.append(val_loss)

                    #TODO can I improve the validation process?
                    true_list.append(y_val.cpu().numpy())
                    preds= torch.squeeze( torch.sigmoid(val_output).round() ) if binary_case \
                        else torch.softmax(val_output,dim=-1)
                    pred_list.append(preds.cpu().numpy())
            true_np,preds_np = np.concatenate(true_list), np.concatenate(pred_list)


            # TODO Can I remove this line of code?
            preds_np= np.argmax(preds_np,axis=-1) if not binary_case else preds_np
            val_acc = accuracy_score(true_np,preds_np)
            val_acc_hist.append(val_acc)


            # early stopping
            best_val_acc = max(val_acc_hist[:-1]) if len(val_acc_hist)>1 else 0
            if best_val_acc>=val_acc:
                val_acc_hist.append(val_acc)
                patience_counter+=1
                if patience_counter==patience:
                    return best_val_acc
            else:
                val_acc_hist.append(val_acc)
                patience_counter=0
                torch.save( self.state_dict(), "saved_model/synths/"+self.save_name)

            #self.val_acc.append(val_acc)
            #self.val_loss.append(np.mean(epoch_val_loss))


            # TODO save best model!
            """
            print(f'Epoch: {epoch + 1}, '
                  f'Train loss: {round(self.train_loss[-1], 4)}, '
                  f'Val loss: {round(self.val_loss[-1], 4)}, ',
                  f'Val acc: {round(val_acc, 4)} ')

            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
                # if self.val_acc[-1] > best_val_acc:
                #     best_val_acc = self.val_acc[-1]
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    return

        if save_best_model == True:
            savepath = './model/{self.ds}_best.pkl'
            torch.save(self.model,savepath)
        """

    #TODO do i really need this function?
    def evaluate(self,):
        data = LocalDataLoader(self.datapath,self.ds)
        test_loader,_ = data.get_loaders(mode='test')
        self.model.eval()
        true_list,pred_list = [],[]

        for x,y in test_loader:
            with torch.no_grad():
                true_list.append(y.detach().numpy())
                preds = self.model(x)
                if y.shape[-1] == 2:
                    preds = torch.sigmoid(preds)
                else:
                    preds=torch.softmax(preds,dim=-1)
                pred_list.append(preds.detach().numpy())
        true_np,preds_np = np.concatenate(true_list), np.concatenate(pred_list)

        true_np = np.argmax(true_np,axis=-1)
        preds_np= np.argmax(preds_np,axis=-1)
        self.test_result = accuracy_score(true_np,preds_np)

        print(f'Accuracy score: {round(self.test_result, 4)}')


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)












class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)
