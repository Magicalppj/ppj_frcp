import sys

from tqdm import tqdm
import torch


def train_one_epoch(model, loss_function, optimizer, data_loader, device, trained_epochs, end_epoch=None):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader) # 打印进度条
    for step, data in enumerate(data_loader):
        images, labels = data
        if isinstance(labels, list):
            labels = [label.to(device, dtype=torch.float32) for label in labels]
        else:
            labels = labels.to(device, dtype=torch.float32)
        pred = model(images.to(device),labels[-1]) # 将Rt label加入网络的forward过程

        loss = loss_function(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        data_loader.desc = "[epoch {}/{}] mean loss {}".format(trained_epochs , end_epoch,
                                                               round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, loss_function, data_loader, device):
    model.eval()
    # 在进程0中打印验证进度
    data_loader = tqdm(data_loader)
    mean_loss = torch.zeros(1, ).to(device)
    for step, data in enumerate(data_loader):
        images, labels = data
        if isinstance(labels, list):
            labels = [label.to(device, dtype=torch.float32) for label in labels]
        else:
            labels = labels.to(device, dtype=torch.float32)
        pred = model(images.cuda(device, non_blocking=True),labels[-1])
        criternion = loss_function(pred, labels)
        mean_loss = (mean_loss * step + criternion.detach()) / (step + 1)  # update mean losses
    model.train()  # 改回训练模式
    return mean_loss.item()
