from src.utils import LabelledDS, UnlabelledDS, cf_labelled, cf_unlabelled, DrawDS, draw_legend, draw_teacher, read_sses, plot_latent
from src.distributions_gru import Generator, Inference, Classifier, Prior
from pixyz.losses import ELBO
from pixyz.models import Model
import torch_optimizer as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
from PIL import Image
from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
from glob import glob
from pathlib import Path
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
import cv2
import seaborn
from tensorboardX import SummaryWriter

class GMVAE:
    def __init__(self, patch_dir, label_dir, z_dim=4, batch_size=20, drop_out_rate=0, lr=1e-3, device="cuda", num_workers=1):
        self.patch_dir = patch_dir
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.classes = [Path(n).name for n in glob(patch_dir + "/labelled/*")]

        _, labels = read_sses(label_dir, (9999,9999))
        self.labels = labels
        self.label_dir = label_dir

        # Setting DataLoaders
        labelled = LabelledDS(patch_dir + "/labelled")
        train_indices, val_indices = train_test_split(list(range(len(labelled.dataset.targets))), test_size=0.2, stratify=labelled.dataset.targets)
        train_dataset = torch.utils.data.Subset(labelled, train_indices)
        val_dataset = torch.utils.data.Subset(labelled, val_indices)
        x, y = train_dataset[0]
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=cf_labelled)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cf_labelled)
        
        unlabelled = UnlabelledDS(patch_dir + "/unlabelled")
        self.unlabelled_loader = DataLoader(unlabelled, batch_size, shuffle=True, num_workers=num_workers, collate_fn=cf_unlabelled)

        self.class_to_idx = self.train_loader.dataset.dataset.dataset.class_to_idx
        self.x_dim = x.shape[2]
        self.y_dim = len(self.classes)
        self.seq_length = x.shape[1]

        self.p = Generator(self.x_dim, z_dim, self.seq_length, device).to(device)
        self.q = Inference(self.x_dim, z_dim, y_dim=len(self.classes), drop_out_rate=drop_out_rate).to(device)
        self.f = Classifier(self.x_dim, y_dim=len(self.classes), drop_out_rate=drop_out_rate).to(device)
        self.prior = Prior(z_dim, y_dim=len(self.classes)).to(device)
        self.p_joint = self.p * self.prior
        
        # distributions for unsupervised learning
        _q_u = self.q.replace_var(x="x_u", y="y_u")
        p_u = self.p.replace_var(x="x_u")
        f_u = self.f.replace_var(x="x_u", y="y_u")
        prior_u = self.prior.replace_var(y="y_u")
        q_u = _q_u * f_u
        p_joint_u = p_u * prior_u
        
        p_joint_u.to(device)
        q_u.to(device)
        f_u.to(device)
        
        elbo_u = ELBO(p_joint_u, q_u)
        elbo = ELBO(self.p_joint, self.q)
        nll = -self.f.log_prob() #-LogProb(f)
        
        rate = 1 * (len(self.unlabelled_loader) + len(self.train_loader)) / len(self.train_loader)
        
        self.loss_cls = -elbo_u.mean() -elbo.mean() + (rate * nll).mean()
        self.loss_cls_test = nll.mean()
        self.model = Model(self.loss_cls,test_loss=self.loss_cls_test,
                      distributions=[self.p, self.q, self.f, self.prior], optimizer=optim.RAdam, optimizer_params={"lr":lr})
        print("Model:")
        print(self.model)
    
    def _train(self, epoch):
        train_loss = 0
        labelled_iter = self.train_loader.__iter__()
        for x_u in tqdm(self.unlabelled_loader):
            try: 
                x, y = labelled_iter.next()
            except StopIteration:
                labelled_iter = self.train_loader.__iter__()
                x, y = labelled_iter.next()
            
            y = torch.eye(self.y_dim)[y].to(self.device)
            x = x.to(self.device)
            x_u = x_u.to(self.device)
            
            loss = self.model.train({"x": x, "y": y, "x_u": x_u})
            train_loss += loss.detach().item()

        train_loss = train_loss * self.unlabelled_loader.batch_size / len(self.unlabelled_loader.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss
    
    def _val(self, epoch):
        test_loss = 0
        total = [0 for _ in range(len(self.classes))]
        tp = [0 for _ in range(len(self.classes))]
        fp = [0 for _ in range(len(self.classes))]
        for x, _y in self.val_loader:
            y = torch.eye(self.y_dim)[_y].to(self.device)
            x = x.to(self.device)
            loss = self.model.test({"x": x, "y": y})
            test_loss += loss.detach().item()
            self.f.eval()
            with torch.no_grad():
                pred_y = self.f.sample_mean({"x": x}).argmax(dim=1)
            for c in range(len(self.classes)):
                pred_yc = pred_y[_y==c]
                _yc = _y[pred_y==c]
                total[c] += len(_y[_y==c])
                tp[c] += len(pred_yc[pred_yc==c])
                fp[c] += len(_yc[_yc!=c])
        
        test_loss = test_loss * self.val_loader.batch_size / len(self.val_loader.dataset)
        test_recall = [100 * c / t for c,t in zip(tp, total)]
        test_precision = []
        for _tp,_fp in zip(tp, fp):
            if _tp + _fp == 0:
                test_precision.append(0)
            else:
                test_precision.append(100 * _tp / (_tp + _fp))
        c = self.class_to_idx
        recall = {}
        prec = {}
        for _, row in self.labels.iterrows():
            index = row[0]
            name = row[1]
            recall[name] = test_recall[c[str(index)]]
            prec[name] = test_precision[c[str(index)]]
        
        print("Test Loss:", str(test_loss), "Test Recall:", str(recall), "Test Precision:", str(prec))
        return test_loss, recall, prec
    
    def train(self, epochs, log_dir):
        self.best_loss = 9999
        self.best_f = "./runs/" + log_dir + "/best.tp"
        writer = SummaryWriter("./runs/" + log_dir)
        x = []
        _y = [] 
        for xx, yy in self.val_loader:
            x.append(xx)
            _y.append(yy)
        
        x = torch.cat(x, 0)
        _y = torch.cat(_y, 0)
        y = torch.eye(len(self.classes))[_y].to(self.device).squeeze()
        x = x.to(self.device)

        cmap = plt.get_cmap("tab20", self.y_dim)
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            #self.scheduler.step()
            val_loss, recall, precision = self._val(epoch)
            if val_loss < self.best_loss:
                torch.save(self.f.state_dict(), self.best_f)
                self.best_recall = recall
                self.best_prec = precision
                self.best_loss = val_loss
            writer.add_scalar("test_loss", val_loss, epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            for label in recall:
                writer.add_scalar("test_recall_" + label, recall[label], epoch)
                writer.add_scalar("test_precision_" + label, precision[label], epoch)
            #latent = plot_latent(self.f, self.q, x, y, cmap)
            #writer.add_images("Image_latent", latent, epoch)
    def draw(self, image_dir, out_path, kernel_size, batch_size):
        with Image.open(glob(image_dir+"/*")[0]) as img:
            w, h = img.size
        self.load_f(self.best_f)
        dataset = DrawDS(image_dir, kernel_size)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        confs = []
        self.f.eval()
        with torch.no_grad():
            for x in tqdm(loader):
                x = x.to(self.device)
                y = self.f.sample_mean({"x": x}).detach().cpu()
                pred_y = y.argmax(1)
                conf = y.max(1)[0]
                pred_ys.append(pred_y)
                confs.append(conf)
        
        seg_image = torch.cat(pred_ys).reshape([h,w]).numpy()
        confs = torch.cat(confs).reshape([h,w]).numpy()
        cmap = plt.get_cmap("tab20", self.y_dim)
        plt.imsave(out_path, seg_image, cmap = cmap)
        out_path2 = Path(out_path).stem + "_conf.png"
        fig = plt.figure(dpi = 100, figsize = (10,6))
        ax = seaborn.heatmap(confs, linewidth = 0.0, vmin = 0.0, vmax = 1.0, )
        plt.savefig(out_path2)
        plt.cla()
        return seg_image, confs
    
    def draw_teacher(self, out_path, image_size):
        draw_teacher(out_path, self.label_dir, self.class_to_idx, image_size)
    
    def draw_legend(self, out_path):
        draw_legend(out_path, self.label_dir, self.class_to_idx)
    
    def load_f(self, classifier_path):
        self.f.load_state_dict(torch.load(classifier_path))