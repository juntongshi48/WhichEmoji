import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

import pdb

class myTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = cfg.device

    def train(self, optimizer, epoch):
        self.model.train()
        losses = []
        targets = None
        predictions = None
        for x, eos, y in self.train_loader:
            x = x.to(self.device)
            eos = eos.to(self.device)
            y = y.to(self.device)
            loss = self.model.loss(x, eos, y, epoch)
            
            optimizer.zero_grad()
            loss.backward()
            if hasattr(self.cfg, "grad_clip"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            y_pred = self.model.predict(x, eos)
            if targets is None:
                targets = y.cpu()
            else:
                targets = np.concatenate((targets, y.cpu()), axis=0)
            if predictions is None:
                predictions = y_pred
            else:
                predictions = np.concatenate((predictions, y_pred), axis=0)

            # desc = f'Epoch {epoch}'
            losses.append(loss.cpu().item()) # BUG: transfer loss from cuda to cpu
            # for k, v in out.items():
            #     if k not in losses:
            #         losses[k] = []
            #     losses[k].append(v.item())
            #     avg_loss = np.mean(losses[k][-50:])
            #     desc += f', {k} {avg_loss:.4f}'

        #     if not quiet:
        #         pbar.set_description(desc)
        #         pbar.update(x.shape[0])
        # if not quiet:
        #     pbar.close()
        # isTrue = predictions==targets
        # if len(targets.shape) > 1:
        #     # import pdb
        #     # pdb.set_trace()
        #     isTrue = np.all(predictions==targets, axis=1)
        # accuracy = np.sum(isTrue) / len(self.train_loader.dataset)
        accuracy = self.eval_acc(predictions, targets)
        print(f'train_accuracy: {accuracy}')
        train_metrics = dict(loss=losses, accuracy=accuracy)
        return train_metrics


    def eval(self, data_loader):
        self.model.eval()
        total_loss = 0
        num_batch = 0
        predictions = None
        # targets = np.zeros(0)
        with torch.no_grad():
            for x, eos, y in data_loader:
                x = x.to(self.device)
                eos = eos.to(self.device)
                y = y.to(self.device)
                loss = self.model.loss(x, eos, y, 100).cpu().item()
                total_loss += loss
                num_batch += 1
                y_pred = self.model.predict(x, eos)
                if predictions is None:
                    predictions = y_pred
                else:
                    predictions = np.concatenate((predictions, y_pred), axis=0)
                # targets = np.concatenate((targets, y.cpu()))
                # for k, v in out.items():
                #     total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

            # desc = 'Test '
            # for k in total_losses.keys():
            #     total_losses[k] /= len(data_loader.dataset)
            #     desc += f', {k} {total_losses[k]:.4f}'
            # if not quiet:
            #     print(desc)
        loss = total_loss / num_batch
        targets = data_loader.dataset.labels
        # isTrue = predictions==targets
        # if len(targets.shape) > 1:
        #     isTrue = np.all(predictions==targets, axis=1)
        # accuracy = np.sum(isTrue) / len(data_loader.dataset)
        accuracy = self.eval_acc(predictions, targets)
        f1_macro = f1_score(targets, predictions, average='macro')
        if self.cfg.num_output_labels == 1:
            confus_matrix = confusion_matrix(targets, predictions, normalize='true')
            eval_metrics = dict(loss=loss, accuracy=accuracy, f1_macro=f1_macro, confusion_matrix=confus_matrix)
        else:
            eval_metrics = dict(loss=loss, accuracy=accuracy, f1_macro=f1_macro)
        return eval_metrics

    def eval_acc(self, predictions, targets):
        isTrue = None
        if len(targets.shape) > 1:
            if self.cfg.metrics.relaxed:
                isTrue = np.all((predictions-targets) >= 0, axis=1)
            else:
                isTrue = np.all(predictions==targets, axis=1)
        else:
            isTrue = predictions==targets
        accuracy = np.mean(isTrue)
        return accuracy
    

    # def evaluate_accuracy(self, self.model, data_loader):
    #         self.model.eval()
    #         total_correct = 0
    #         with torch.no_grad():
    #             for x, eos, y in data_loader:
    #                 x = x.cuda()
    #                 eos = eos.cuda()
    #                 y = y.cuda()
    #                 y_pred = self.model.predict(x, eos)
    #                 total_correct += torch.sum((y_pred == y))

    #         return total_correct / len(data_loader.dataset)


    def train_epochs(self):
        epochs, lr = self.cfg.epochs, self.cfg.lr
        grad_clip = self.cfg.grad_clip if hasattr(self.cfg, "grad_clip") else None
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_losses, val_losses, train_accuracies, val_accuracies, test_metrics_ = [], [], [], [], []
        for epoch in tqdm(range(epochs)):
            self.model.epoch = epoch
            train_metrics = self.train(optimizer, epoch)
            val_metrics = self.eval(self.val_loader)
            test_metrics = self.eval(self.test_loader)

            train_losses.extend(train_metrics['loss'])
            train_accuracies.append(train_metrics['accuracy'])
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            test_metrics_.append(test_metrics)
            print('epoch: %d, avg_train_loss: %0.8f, avg_val_loss: %0.8f, val_accuracy: %0.4f' %\
                (epoch, sum(train_metrics['loss'])/len(train_metrics['loss']), val_metrics['loss'], val_metrics['accuracy']))
            
        return train_losses, val_losses, train_accuracies, val_accuracies, test_metrics_