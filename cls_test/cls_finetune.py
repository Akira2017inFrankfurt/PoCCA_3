from cls_model_4_compare import DG_Tail, get_encoder, DGCNN_CLS_Encoder_1
from cls_eval_data import ModelNet10, ModelNet40, ScanObjectNN
import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def log_func(log_file):
    if not os.path.exists(log_file):
        os.makedirs(log_file)
        print("Make Log File! ", log_file)
    else:
        print("Log File Already Exists")
    tensorboard_writer = SummaryWriter(log_dir=log_file)
    return tensorboard_writer


def get_data_loader(args):
    if args.dataset is 'ModelNet40':
        choice_class = ModelNet40
    elif args.dataset is 'ModelNet10':
        choice_class = ModelNet10
    else:
        choice_class = ScanObjectNN
    train_set = choice_class(num_points=1024, partition='train', percent=args.data_percentage)
    test_set = choice_class(num_points=1024, partition='test', percent=args.data_percentage)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader


def get_model(args):
    model_0 = DGCNN_CLS_Encoder_1().to(device)
    encoder = get_encoder(model_0, args.network_weight_path)
    # network_choice = 1 if args.network is 'DGCNN' else 0
    model = DG_Tail(encoder=encoder).to(device)
    return model


def train_1_epoch(optimizer, criterion, loader, model):
    mean_correct = []
    train_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels[:, 0].to(device)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, labels.long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(labels.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(inputs.shape[0]))
    train_acc = np.mean(mean_correct)
    return train_acc, train_loss


@torch.no_grad()
def evaluate(criterion, loader, model):
    model.eval()
    test_mean_correct = []
    test_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels[:, 0].to(device)
        pred = model(inputs)
        loss = criterion(pred, labels.long())
        test_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(labels.long().data).cpu().sum()
        test_mean_correct.append(correct.item() / float(inputs.size()[0]))
    test_acc = np.mean(test_mean_correct)
    return test_acc, test_loss / len(loader)


def train_and_test(args):
    classifier = get_model(args)
    criterion = nn.CrossEntropyLoss()
    # args.opt is 'sgd' by default
    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    if args.opt is 'adamw':
        optimizer = optim.AdamW(classifier.parameters(),
                                lr=args.lr, betas=(0.9, 0.999),
                                weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    train_loader, test_loader = get_data_loader(args)
    tb_writer = log_func(args.log_file)
    tags = ["train_acc", "learning_rate", "train_loss", "test_acc", "test_loss"]
    # record the best information
    best_test_epoch = 0
    best_test_acc = 0.0
    pair_train_acc = 0.0

    print('Start training...')
    for epoch in range(args.epochs):
        print('\n')
        print('---Training epoch: ', epoch, ' ---')
        train_acc, train_loss = train_1_epoch(optimizer=optimizer,
                                              criterion=criterion,
                                              loader=train_loader,
                                              model=classifier)
        print('+ Train acc is: ', train_acc)
        scheduler.step()
        tb_writer.add_scalar(tags[0], train_acc, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], train_loss / len(train_loader), epoch)
        test_acc, test_loss = evaluate(criterion=criterion, loader=test_loader, model=classifier)
        print('+ Test acc is: ', test_acc)
        tb_writer.add_scalar(tags[3], test_acc, epoch)
        tb_writer.add_scalar(tags[4], test_loss, epoch)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print("best acc: ", best_test_acc)
            pair_train_acc = train_acc
            best_test_epoch = epoch
            print('Save model at {} epoch...'.format(epoch))
            save_path = 'best_cls_model.pth'
            print('Saving at %s' % save_path)
            state = {
                'epoch': best_test_epoch,
                'test_acc': best_test_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if epoch == args.epochs - 1:
            print('Save the last model at {} epoch...'.format(epoch))
            save_path = 'last_cls_model.pth'
            print('Saving at %s' % save_path)
            state = {
                'epoch': best_test_epoch,
                'test_acc': best_test_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

    print("Best test_acc is {}, train_acc is {}, epoch is {}".format(best_test_acc, pair_train_acc, best_test_epoch))
    print("***********\n")


if __name__ == "__main__":
    model_path = r'/home/haruki/下载/SimAttention/scripts/weights/'
    model_name = 'cls_dg_1205_cross_99.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_path = r"/home/haruki/下载/SimAttention/cls_test/test_events"
    log_file_name = 'test_4_' + model_name[:-4]
    for percent in [100]:
        parser = argparse.ArgumentParser(description='Point Cloud Train.')
        parser.add_argument('--network', type=str, default='DGCNN', choices=['PointNet'])
        parser.add_argument('--network_weight_path', type=str, default=os.path.join(model_path, model_name))
        parser.add_argument('--use_pretrain', type=bool, default=True)
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet10', 'Scan'])
        parser.add_argument('--data_percentage', type=int, default=percent)
        parser.add_argument('--train_batch_size', type=int, default=16)
        parser.add_argument('--test_batch_size', type=int, default=8)
        parser.add_argument('--opt', type=str, default='sgd', choices=['adamw'])
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--epochs', type=int, default=250)
        parser.add_argument('--log_file', type=str, default=os.path.join(log_path, log_file_name))

        para = parser.parse_args()
        train_and_test(para)
