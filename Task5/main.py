import torch
import torch.optim as optim
import pandas as pd
import argparse

import aggmodel
import data


def train_and_evaluate(model, train_dataloader, val_dataloader, device, num_epochs, lr):
    """训练模型并在验证集上评估性能"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_model = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        avg_loss, cnt = 0.0, 0.0

        for img, text, label, _ in train_dataloader:
            loss = -((model(img, text) + 1e-80) / (1 + 1e-80)).log()
            loss = loss[torch.arange(0, label.size(0)), label].mean()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            avg_loss = (avg_loss * cnt + loss.item()) / (cnt + 1)
            cnt += 1

        # 验证阶段
        model.eval()
        classification = []
        with torch.no_grad():
            for img, text, label, _ in val_dataloader:
                out = model(img, text).argmax(dim=-1).cpu().numpy()
                label = label.cpu().numpy()
                classification += [{"out": out[i], "tag": label[i]} for i in range(out.shape[0])]

        df = pd.DataFrame(classification)
        acc_now = ((df['out'] == df['tag']) * 1.0).mean()
        if acc_now > best_acc:
            best_model = model
            best_acc = acc_now
        print(f'Epoch {epoch}, avg_loss: {avg_loss:.4f}, acc_now: {acc_now:.4f}, best_acc: {best_acc:.4f}')

    return best_model, best_acc


def output_test_results(model, test_dataloader, device, lr):
    """在测试集上运行并保存结果"""
    model.eval()
    classification = []
    label_map = ["positive", "neutral", "negative"]

    with torch.no_grad():
        for img, text, uid in test_dataloader:
            out = model(img, text).argmax(dim=-1).cpu().numpy()
            uid = uid  # 将uid转换为numpy数组
            classification += [{"guid": uid[i], "tag": label_map[out[i]]} for i in range(out.shape[0])]

    res = pd.DataFrame(classification)
    res.to_csv(f'answer_lr{lr}.txt', index=False)  # 不将guid设置为索引，直接保存两列

def run_experiment(model, batch_size, num_epochs, lr, mode="train"):
    """运行完整的实验流程"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = data.FitDataset.load("train", device)
    val_data = data.FitDataset.load("val", device)
    train_dataloader = train_data.to_dataloader(batch_size)
    val_dataloader = val_data.to_dataloader(batch_size)

    if mode == "train":
        test_data = data.TestSet.load(device)
        test_dataloader = test_data.to_dataloader(batch_size)

        # 训练和测试
        best_model, _ = train_and_evaluate(model, train_dataloader, val_dataloader, device, num_epochs, lr)
        output_test_results(best_model, test_dataloader, device, lr)
    else:
        # 消融实验
        train_and_evaluate(model, train_dataloader, val_dataloader, device, num_epochs, lr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a multi modal sentiment analysis model')
    parser.add_argument('--model', type=str, default='agg', choices=['agg', 'text', 'image'], help='choose a model')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=1, help='num_epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

    args = parser.parse_args()
    if args.model == "agg":
        print("agg")
        selected_model = aggmodel.AttentionModel()
    elif args.model == "text":
        print("text")
        selected_model = aggmodel.OnlyTextModel()
    elif args.model == "image":
        print("image")
        selected_model = aggmodel.OnlyImageModel()
    else:
        print("输入不正确")

    run_experiment(selected_model, batch_size=args.batch_size, num_epochs=args.num_epochs, lr=args.lr, mode="train")

