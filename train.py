import os
import sys
import time
import torch
import random
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from models import TransDelta
from torch.utils.data import DataLoader
from loss_func import CE_Loss
from data_utils import MyDataset, build_tokenizer, build_embedding_matrix

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        self._set_seed(opt.seed)
        opt.zeta = opt.zeta if 'M' in opt.method else 0.0
        opt.tokenizer = build_tokenizer(domains=opt.domains, fnames=opt.dataset_file.values())
        embedding_matrix = build_embedding_matrix(domains=opt.domains, vocab=opt.tokenizer.vocab['word'])
        self.trainset = MyDataset(side='main', tasks=opt.tasks, domains=opt.domains, fname=opt.dataset_file['train'], tokenizer=opt.tokenizer)
        self.testset = MyDataset(side='main', tasks=opt.tasks, domains=opt.domains, fname=opt.dataset_file['test'], tokenizer=opt.tokenizer)
        self.auxset = MyDataset(side='aux', tasks=opt.tasks, domains=opt.domains, fname=opt.dataset_file['aux'], tokenizer=opt.tokenizer)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.info = f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}\ntraining arguments:\n"
        self.info += '\n'.join([f">>> {arg}: {getattr(self.opt, arg)}" for arg in vars(self.opt)])
        if self.opt.device.type == 'cuda':
            print(f"cuda memory allocated: {torch.cuda.memory_allocated(self.opt.device.index)}")
        print(self.info)
    
    def _reset_records(self):
        self.records = {
            'best_epoch': 0,
            'best_test_acc': 0,
            'best_test_f1': 0,
            'train_loss': list(),
            'train_acc': list(),
            'test_loss': list(),
            'test_acc': list(),
            'checkpoints': list()
        }
    
    def _update_records(self, epoch, train_loss, test_loss, train_acc, test_acc, test_f1):
        if test_acc > self.records['best_test_acc']:
            path = f"./state_dict/{self.opt.model_name}_{'_'.join(self.opt.domains.values())}_acc{test_acc:.4f}_temp{str(time.time())[-6:]}.pt"
            torch.save(self.model.state_dict(), path)
            self.records['checkpoints'].append(path)
            self.records['best_test_acc'] = test_acc
            self.records['best_test_f1'] = test_f1
            self.records['best_epoch'] = epoch
        self.records['train_loss'].append(train_loss)
        self.records['test_loss'].append(test_loss)
        self.records['train_acc'].append(train_acc)
        self.records['test_acc'].append(test_acc)
    
    def _draw_records(self, transferred_instances=None, logs=None):
        timestamp = f"{time.time():.7f}"
        print(f"best epoch: {self.records['best_epoch']}")
        print(f"best test acc: {self.records['best_test_acc']:.4f}, best test f1: {self.records['best_test_f1']:.4f}")
        print(f"best train loss: {min(self.records['train_loss']):.4f}, best test loss: {min(self.records['test_loss']):.4f}")
        if len(self.records['checkpoints']):
            os.rename(self.records['checkpoints'][-1], f"./state_dict/{self.opt.model_name}_{'_'.join(self.opt.domains.values())}_f{self.records['best_test_acc']:.4f}_{timestamp}.pt")
            for path in self.records['checkpoints'][0:-1]:
                os.remove(path)
        # Draw figures
        plt.figure()
        trainloss, = plt.plot(self.records['train_loss'])
        testloss, = plt.plot(self.records['test_loss'])
        plt.legend([trainloss, testloss], ['train', 'test'], loc='upper right')
        plt.title(f"{timestamp} loss curve")
        plt.savefig(f"./figs/{timestamp}_loss.png", format='png', transparent=True, dpi=300)
        plt.figure()
        trainacc, = plt.plot(self.records['train_acc'])
        testacc, = plt.plot(self.records['test_acc'])
        plt.legend([trainacc, testacc], ['train', 'test'], loc='upper left')
        plt.title(f"{timestamp} acc curve")
        plt.savefig(f"./figs/{timestamp}_acc.png", format='png', transparent=True, dpi=300)
        # Save report
        report_key = '\t'.join(['test_acc', 'test_f1', 'best_train_loss', 'best_test_loss', 'best_epoch', 'timestamp'])
        report_value = '\t'.join([f"{self.records['best_test_acc']:.4f}", f"{self.records['best_test_f1']:.4f}", f"{min(self.records['train_loss']):.4f}", f"{min(self.records['test_loss']):.4f}", f"{self.records['best_epoch']}", f"{timestamp}"])
        report = f"{report_key}\n{report_value}\n{self.info}"
        open(f"./logs/{timestamp}_log.txt", 'w').write(report)
        print(f"report saved: ./logs/{timestamp}_log.txt")
        if transferred_instances is not None:
            instances = '\n'.join([f"{weight:.2f}\t{text}" for weight, text in transferred_instances])
            open(f"./logs/{timestamp}_casestudy.txt", 'w').write(instances)
            print(f"case study saved: ./logs/{timestamp}_casestudy.txt")
        if logs is not None:
            log_outputs = '\n'.join([','.join(log_aspect) for log_aspect in logs])
            open(f"./logs/{timestamp}_logs.csv", 'w').write(log_outputs)
            print(f"logs saved: ./logs/{timestamp}_logs.csv")
    
    def _train(self, gamma, optimizer, criterion, train_dataloader, aux_dataloader):
        if 'M' in self.opt.method:
            train_loss, n_train = {'main': 0, 'aux': 0}, {'main': 0, 'aux': 0}
        else:
            train_loss, n_train = {'main': 0}, {'main': 0}
        n_correct = 0
        batch_num = min(len(train_dataloader), len(aux_dataloader))
        train_iter = {'main': iter(train_dataloader), 'aux': iter(aux_dataloader)}
        self.model.train()
        for i_batch in range(batch_num):
            optimizer.zero_grad()
            sample_batched, inputs, labels = dict(), dict(), dict()
            for side in ['main', 'aux']:
                sample_batched[side] = next(train_iter[side])
                inputs[side] = [sample_batched[side][col].to(self.opt.device) for col in self.opt.input_cols]
                labels[side] = sample_batched[side]['polarity'].to(self.opt.device)
            for loss_key in train_loss.keys():
                if 'delta' in self.opt.model_name:
                    self.model.update_params(loss_key)
                outputs = {'main': None, 'aux': None} if 'D' in self.opt.method else {loss_key: None}
                batch_loss = 0
                for output_key in outputs.keys():
                    outputs[output_key] = self.model(inputs[output_key], loss_key)
                    batch_loss += criterion(inputs[output_key], outputs[output_key], labels[output_key], self.opt.tasks[loss_key], self.model, gamma, pseudo=(loss_key!=output_key))
                batch_loss.backward()
                if 'delta' in self.opt.model_name:
                    final_loss = self.model.compute_final_loss(loss_key)
                    final_loss.backward()
                train_loss[loss_key] += batch_loss.item() * len(labels[loss_key])
                n_train[loss_key] += len(labels[loss_key])
                if loss_key == 'main':
                    n_correct += (torch.argmax(outputs['main'], -1) == labels['main']).sum().item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.clip_norm, norm_type=2)
            optimizer.step()
            ratio = int((i_batch+1)*50/batch_num)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{batch_num} {(i_batch+1)*100/batch_num:.2f}%")
            sys.stdout.flush()
        print()
        train_loss = sum([train_loss[k] / n_train[k] for k in train_loss.keys()])
        train_acc = n_correct / n_train['main']
        return train_loss, train_acc
    
    def _evaluate(self, criterion, test_dataloader):
        test_loss, test_acc, test_f1 = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            n_correct, n_test = 0, 0
            inputs_all, targets_all, outputs_all = None, None, None
            if 'delta' in self.opt.model_name:
                self.model.update_params('main')
            for sample_batched in test_dataloader:
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs = self.model(inputs, 'main')
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test += len(targets)
                inputs_all = [torch.cat((inputs_all[i], inputs[i]), dim=0) if inputs_all is not None else inputs[i] for i in range(len(inputs))]
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
            test_loss = criterion(inputs_all, outputs_all, targets_all, self.opt.tasks['main']).item()
            test_acc = n_correct / n_test
            test_f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_loss, test_acc, test_f1
    
    def _evaluate_aux(self, criterion, aux_dataloader, casestudy=False, threshold=0.5):
        transferred_count, total_count, transferred_instances, batch_num = 0, 0, list(), len(aux_dataloader)
        self.model.eval()
        with torch.no_grad():
            if 'delta' in self.opt.model_name:
                self.model.update_params('main')
            for i_batch, sample_batched in enumerate(aux_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                text_len = torch.sum(inputs[0]!=0, dim=-1)
                outputs = self.model(inputs, 'main')
                _, weight = criterion(inputs, outputs, targets, self.model, gamma=1, pseudo=True, inference=True)
                total_count += len(targets)
                if casestudy:
                    for i in range(len(targets)):
                        if weight[i] > threshold:
                            transferred_count += 1
                            text = ' '.join([self.opt.tokenizer.vocab['word'].id_to_word(t.item()) for t in inputs[0][i]][:text_len[i]])
                            transferred_instances.append((weight[i].item(), text))
                else:
                    transferred_count += torch.where(weight>threshold, torch.ones_like(weight), torch.zeros_like(weight)).sum().item()
                ratio = int((i_batch+1)*50/batch_num)
                sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{batch_num} {(i_batch+1)*100/batch_num:.2f}%")
                sys.stdout.flush()
            print()
            print(f"Total aux data count: {total_count}, transferred aux data count: {transferred_count}.")
            print(f"{transferred_count / total_count * 100:.2f}% aux data transferred.")
            return transferred_instances, transferred_count, transferred_count / total_count
    
    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        criterion = CE_Loss(opt=self.opt)
        train_dataloader = DataLoader(dataset=self.trainset, batch_size=opt.batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=self.testset, batch_size=opt.batch_size, shuffle=False)
        aux_dataloader = DataLoader(dataset=self.auxset, batch_size=opt.batch_size, shuffle=True)
        self._reset_records()
        accs, f1s, betas, log_model_paths = [], [], [], []
        transferred_instances = None
        for epoch in range(self.opt.num_epoch):
            gamma = epoch / self.opt.num_epoch * self.opt.gamma if self.opt.change_gamma else self.opt.gamma
            train_loss, train_acc = self._train(gamma, optimizer, criterion, train_dataloader, aux_dataloader)
            test_loss, test_acc, test_f1 = self._evaluate(criterion, test_dataloader)
            self._update_records(epoch, train_loss, test_loss, train_acc, test_acc, test_f1)
            print(f"{100*(epoch+1)/self.opt.num_epoch:6.2f}% > loss: {train_loss:.4f}, acc: {train_acc:.4f}, test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}")
            if self.opt.save_log:
                accs.append(f"{test_acc:.6f}")
                f1s.append(f"{test_f1:.6f}")
                betas.append(f"{torch.sigmoid(self.model.threshold).item():.6f}")
                if self.opt.save_log_freq and (epoch + 1) % self.opt.save_log_freq == 0:
                    log_model_paths.append(f"./state_dict/log_{self.opt.model_name}_{'_'.join(self.opt.domains.values())}_acc{test_acc:.4f}_temp{str(time.time())[-6:]}.pt")
                    torch.save(self.model.state_dict(), log_model_paths[-1])
        trans_counts, trans_ratios = [], []
        if self.opt.save_log:
            print("evaluating transfer ratio during training...")
            for log_model_path in log_model_paths:
                self.model.load_state_dict(torch.load(log_model_path))
                _, trans_count, trans_ratio = self._evaluate_aux(criterion, aux_dataloader)
                trans_counts.append(f"{trans_count}")
                trans_ratios.append(f"{trans_ratio:.4f}")
                os.remove(log_model_path)
        if self.opt.inference:
            print("evaluating final transfer ratio...")
            self.model.load_state_dict(torch.load(self.records['checkpoints'][-1]))
            transferred_instances, _, _ = self._evaluate_aux(criterion, aux_dataloader, casestudy=True)
        if self.opt.save_log:
            self._draw_records(transferred_instances=transferred_instances, logs=(accs, f1s, betas, trans_counts, trans_ratios))
        else:
            self._draw_records(transferred_instances=transferred_instances)
        return self.records['best_test_acc']
    
    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ''' if you are using cudnn '''
        torch.backends.cudnn.deterministic = True # Deterministic mode can have a performance impact
        torch.backends.cudnn.benchmark = False
    

if __name__ == '__main__':
    
    model_classes = {
        'transdelta': TransDelta
    }
    
    input_colses = {
        'transdelta': ['text', 'aspect', 'aspect_mask']
    }
    
    dataset_files = {
        'res14': {
            'train': 'Restaurants_Train.json',
            'test': 'Restaurants_Test.json'
        },
        'laptop': {
            'train': 'Laptops_Train.json',
            'test': 'Laptops_Test.json'
        },
        'twitter': {
            'train': 'Tweets_Train.json',
            'test': 'Tweets_Test.json'
        },
        'res16': {
            'train': 'Restaurants16_Train.json',
            'test': 'Restaurants16_Test.json'
        },
        'yelp': {
            'train': 'Yelp_Train.json'
        },
        'elec': {
            'train': 'Elec_Train.json'
        },
        'res14_ae': {
            'train': 'Restaurants_Train.json'
        },
        'laptop_ae': {
            'train': 'Laptops_Train.json'
        }
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,            # default lr=0.1
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    ''' Basic '''
    parser.add_argument('--model_name', default='transdelta', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--output_layer', default='mean_pooling', type=str, help='mean_pooling, attention')
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=str)
    ''' For transfer '''
    parser.add_argument('--method', default='', type=str, help='M, D')
    parser.add_argument('--alpha', default=10, type=float)
    parser.add_argument('--beta', default=0.75, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--zeta', default=0.01, type=float)
    parser.add_argument('--change_gamma', default=True, type=bool)
    parser.add_argument('--learned_beta', default=True, type=bool)
    ''' For dataset '''
    parser.add_argument('--main', default='res14', type=str, help='res14, laptop, res16')
    parser.add_argument('--aux', default='yelp', type=str, help='res14, laptop, res16, yelp, elec, res14_ae, laptop_ae')
    ''' For model '''
    parser.add_argument('--layer_num', default=1, type=int)
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--rnn_type', default='GRU', type=str)
    ''' For training '''
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--clip_norm', default=20, type=float)
    ''' Others '''
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--backend', default=True, type=bool)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--automl', default=False, type=bool)
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--save_log', default=False, type=bool)
    parser.add_argument('--save_log_freq', default=5, type=int)
    parser.add_argument('--hard_M', default=False, type=bool)
    parser.add_argument('--hard_D', default=False, type=bool)
    ''' Bayesian Optimization '''
    parser.add_argument('--opt_num_rounds', default=20, type=int)
    parser.add_argument('--opt_lr', default=False, type=bool)
    parser.add_argument('--opt_alpha', default=False, type=bool)
    parser.add_argument('--opt_beta', default=False, type=bool)
    parser.add_argument('--opt_gamma', default=False, type=bool)
    parser.add_argument('--opt_zeta', default=False, type=bool)
    
    opt = parser.parse_args()
    opt.model_class = model_classes[opt.model_name]
    opt.input_cols = input_colses[opt.model_name]
    opt.optimizer = optimizers[opt.optimizer]
    opt.initializer = initializers[opt.initializer]
    opt.method = [m.strip().upper() for m in opt.method.split(',')]
    opt.domains = {'main': opt.main, 'aux': opt.aux}
    opt.tasks = {
        'main': 'asc',
        'aux': 'ae' if 'ae' in opt.aux else 'asc' if opt.aux in ['res14', 'laptop', 'res16'] else 'dsc'
    }
    opt.dataset_file = {
        'train': os.path.join('datasets', opt.tasks['main'], dataset_files[opt.domains['main']]['train']),
        'test': os.path.join('datasets', opt.tasks['main'], dataset_files[opt.domains['main']]['test']),
        'aux': os.path.join('datasets', opt.tasks['aux'], dataset_files[opt.domains['aux']]['train'])
    }
    opt.seed = opt.seed if opt.seed else random.randint(0, 4294967295)
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    assert 0 <= opt.seed < 4294967296
    
    for folder in ['figs', 'logs', 'dats', 'state_dict']:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    if opt.backend: # Disable the matplotlib window
        plt.switch_backend('Agg')
    
    warnings.simplefilter("ignore")
    
    def run_and_evaluate():
        all_test_acc = list()
        for i in range(opt.repeat):
            opt.seed = random.randint(0, 4294967295) if i != 0 else opt.seed # change seed
            print(f"repeat: {i+1}")
            ins = Instructor(opt)
            test_acc = ins.run()
            all_test_acc.append(test_acc)
            print('#' * 50)
        print(f"Max test acc: {max(all_test_acc):.4f}")
        print(f"Average test acc: {sum(all_test_acc) / len(all_test_acc):.4f}")
        return sum(all_test_acc) / len(all_test_acc)
    
    if opt.automl:
        # from ax.service.managed_loop import optimize
        from ax.service.ax_client import AxClient
        opt_parameters = []
        opt_dests = []
        if opt.opt_lr:
            opt_dests.append('lr')
            opt_parameters.append({'name': 'lr', 'type': 'range', 'bounds': [5e-5, 1e-3], 'value_type': 'float', 'log_scale': True})
        if opt.opt_alpha:
            opt_dests.append('alpha')
            opt_parameters.append({'name': 'alpha', 'type': 'range', 'bounds': [1e-2, 100], 'value_type': 'float', 'log_scale': True})
        if opt.opt_beta:
            opt_dests.append('beta')
            opt_parameters.append({'name': 'beta', 'type': 'range', 'bounds': [0.8, 0.9999], 'value_type': 'float'})
        if opt.opt_gamma:
            opt_dests.append('gamma')
            opt_parameters.append({'name': 'gamma', 'type': 'range', 'bounds': [0.1, 2.0], 'value_type': 'float'})
        if opt.opt_zeta:
            opt_dests.append('zeta')
            opt_parameters.append({'name': 'zeta', 'type': 'range', 'bounds': [0.01, 100], 'value_type': 'float', 'log_scale': True})
        def tune(parameters):
            for name in opt_dests:
                setattr(opt, name, parameters.get(name))
            return run_and_evaluate()
        ax = AxClient()
        ax.create_experiment(name='acc_optimization', parameters=opt_parameters, objective_name='Ave. Test Acc')
        for _ in range(opt.opt_num_rounds):
            parameters, idx = ax.get_next_trial()
            ax.complete_trial(trial_index=idx, raw_data=tune(parameters))
        best_parameters, values = ax.get_best_parameters()
        # best_parameters, values, experiment, model = optimize(
        #     parameters=opt_parameters,
        #     evaluation_function=tune,
        #     objective_name='Ave. Test Acc',
        # )
        print(best_parameters)
        print(values)
    else:
        run_and_evaluate()
