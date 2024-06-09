import torch
from model import C_AutoEncoder, Deep_SVDD
from torch import optim
from torchsummary import summary
import time

class TrainerDeepSVDD:
    def __init__(self, args, dataloaders, device):
        self.args = args
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.device = device


    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1 and classname != 'Conv':
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Linear") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def pretrain(self):
        ae = C_AutoEncoder(self.args.latent_dim).to(self.device)
        ae.apply(self.weights_init_normal)
        # summary(ae, (1, 200, 13, 13))
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        start_time = time.time()
        # best_loss = 9999
        # early_count = 0
        
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            ae.train()
            for x, _ in (self.train_loader):
                img = x.float().to(self.device)
                optimizer.zero_grad()
                img_hat = ae(img)
                reconst_loss = ae.loss_func(img.cpu(), img_hat.cpu())
                total_loss += reconst_loss.item()
                reconst_loss.backward()
                optimizer.step()
                
                scheduler.step()
            elapsed = time.time() - start_time
            print('Time: {} , Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   elapsed, epoch, total_loss/(len(self.train_loader) * self.args.batch_size)))
            
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)
        net = Deep_SVDD(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, './weights/test/pretrained_parameters.pth')
    

    def set_c(self, model, dataloader, eps=0.1):
        
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                img = x.float().to(self.device)
                z = model.encoder(img)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self):
        
        net = Deep_SVDD(self.args.latent_dim).to(self.device)
        best_loss = 9999
        loss_count = 0
        self.loss = []
        
        if self.args.train==True:
            state_dict = torch.load('./weights/test/DSVDD_parameters.pth')
            net.load_state_dict(state_dict)
            state_dict = torch.load('./weights/test/pretrained_parameters.pth')
            c = torch.Tensor(state_dict['center']).to(self.device)
            self.net = net
            self.c = c
            return

        if self.args.pretrain==True:
            state_dict = torch.load('./weights/test/pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(self.weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        start_time = time.time()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in (self.train_loader):
                img = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(img)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            elasped = time.time() - start_time
            print('Time: {}, Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(elasped, epoch, total_loss/(len(self.train_loader))))
            self.loss.append(total_loss/(len(self.train_loader)))


            torch.save(net.state_dict(), './weights/test/DSVDD_parameters.pth')
            # if best_loss > total_loss/(len(self.train_loader)):
            #     best_loss = total_loss/(len(self.train_loader))
            # else:
            #     loss_count += 1

            # if loss_count > 5:
            #     torch.save(net.state_dict(), './weights/indian_pines/ori/1/DSVDD_parameters.pth')
            #     break
        self.net = net
        self.c = c