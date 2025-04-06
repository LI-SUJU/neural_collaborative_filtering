import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 固定随机种子保证结果复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def preprocess_data(file_path, min_rating=4, num_negatives=2, test_ratio=0.15, val_ratio=0.15):
    """
    加载 MovieLens 数据集，将评分 >= min_rating 视为正反馈，
    对每个正反馈生成 num_negatives 个负反馈（从用户未交互物品中采样），
    并将正反馈随机划分为训练、验证和测试集。

    返回：
      train_samples: [(user, item, label), ...]
      val_samples: [(user, item, 1), ...]
      test_samples: [(user, item, 1), ...]
      num_users: 用户总数
      num_items: 物品总数
      train_positive_by_user: 训练集中用户的正反馈集合
    """
    ratings_df = pd.read_csv(file_path, sep='::', engine='python',
                             names=['user', 'item', 'rating', 'timestamp'])
    positive_df = ratings_df[ratings_df['rating'] >= min_rating]
    num_users = int(ratings_df['user'].max())
    num_items = int(ratings_df['item'].max())

    positive_interactions = list(zip(positive_df['user'], positive_df['item']))
    random.shuffle(positive_interactions)

    total = len(positive_interactions)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    test_pos = positive_interactions[:n_test]
    val_pos = positive_interactions[n_test:n_test + n_val]
    train_pos = positive_interactions[n_test + n_val:]

    train_positive_by_user = {}
    for user, item in train_pos:
        train_positive_by_user.setdefault(user, set()).add(item)

    # 预先构建每个用户的负样本候选集
    all_items = set(range(1, num_items + 1))
    neg_candidates_by_user = {user: list(all_items - items) for user, items in train_positive_by_user.items()}

    train_samples = []
    for user, pos_item in train_pos:
        train_samples.append((user, pos_item, 1))
        neg_candidates = neg_candidates_by_user[user]
        negatives = (random.sample(neg_candidates, num_negatives)
                     if len(neg_candidates) >= num_negatives
                     else [random.choice(neg_candidates) for _ in range(num_negatives)])
        for neg_item in negatives:
            train_samples.append((user, neg_item, 0))

    val_samples = [(user, item, 1) for user, item in val_pos]
    test_samples = [(user, item, 1) for user, item in test_pos]
    return train_samples, val_samples, test_samples, num_users, num_items, train_positive_by_user


class NCFDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, item, label = self.samples[idx]
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(label, dtype=torch.float))


class NCF(nn.Module):
    """
    Neural Collaborative Filtering 模型，包含 GMF（线性交互）和 MLP（非线性交互）分支，
    融合后输出用户对物品的交互概率。
    """
    def __init__(self, num_users, num_items,
                 gmf_dim=8, mlp_dim=8, mlp_hidden=[16, 8, 4], dropout=0.1):
        super().__init__()
        self.gmf_user_embed = nn.Embedding(num_users + 1, gmf_dim)
        self.gmf_item_embed = nn.Embedding(num_items + 1, gmf_dim)

        self.mlp_user_embed = nn.Embedding(num_users + 1, mlp_dim)
        self.mlp_item_embed = nn.Embedding(num_items + 1, mlp_dim)

        mlp_layers = []
        input_dim = mlp_dim * 2
        for hidden_size in mlp_hidden:
            mlp_layers.append(nn.Linear(input_dim, hidden_size))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        self.mlp_net = nn.Sequential(*mlp_layers)

        fusion_input = gmf_dim + mlp_hidden[-1]
        self.fusion = nn.Linear(fusion_input, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embed.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embed.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embed.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embed.weight, std=0.01)
        for layer in self.mlp_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.fusion.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user_ids, item_ids):
        user_gmf = self.gmf_user_embed(user_ids)
        item_gmf = self.gmf_item_embed(item_ids)
        gmf_out = user_gmf * item_gmf

        user_mlp = self.mlp_user_embed(user_ids)
        item_mlp = self.mlp_item_embed(item_ids)
        mlp_input = torch.cat((user_mlp, item_mlp), dim=1)
        mlp_out = self.mlp_net(mlp_input)

        fusion_input = torch.cat((gmf_out, mlp_out), dim=1)
        output = self.sigmoid(self.fusion(fusion_input))
        return output.view(-1)


def recall_at_k(ranked, ground_truth, k):
    hits = sum(1 for item in ranked[:k] if item in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(ranked, ground_truth, k):
    dcg = sum(1.0 / math.log2(idx + 2) for idx, item in enumerate(ranked[:k]) if item in ground_truth)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / ideal if ideal > 0 else 0


def evaluate_model(model, train_positive_by_user, test_samples, num_users, num_items, k=10, num_negatives=99, device='cpu'):
    test_positive_by_user = {}
    for user, item, _ in test_samples:
        test_positive_by_user.setdefault(user, set()).add(item)

    model.eval()
    recall_list, ndcg_list = [], []
    with torch.no_grad():
        for user in test_positive_by_user:
            ground_truth = test_positive_by_user[user]
            candidates = list(ground_truth)
            negs = set()
            while len(negs) < num_negatives:
                candidate = random.randint(1, num_items)
                if candidate in train_positive_by_user.get(user, set()) or candidate in ground_truth:
                    continue
                negs.add(candidate)
            candidates += list(negs)
            user_tensor = torch.tensor([user] * len(candidates), dtype=torch.long, device=device)
            item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model(user_tensor, item_tensor).cpu().numpy()
            ranked = [x for x, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]
            recall_list.append(recall_at_k(ranked, ground_truth, k))
            ndcg_list.append(ndcg_at_k(ranked, ground_truth, k))
    return np.mean(recall_list), np.mean(ndcg_list)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=20, early_stop_patience=3):
    best_val_loss = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = []
        for user, item, label in train_loader:
            user, item, label = user.to(device), item.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(user, item), label)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        avg_train_loss = np.mean(epoch_train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for user, item, label in val_loader:
                user, item, label = user.to(device), item.to(device), label.to(device)
                loss = criterion(model(user, item), label)
                epoch_val_loss.append(loss.item())
        avg_val_loss = np.mean(epoch_val_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            no_improve = 0
            print("no_improve", no_improve)
            print("best_val_loss", best_val_loss)

        else:
            no_improve += 1
            print("no_improve", no_improve)
            if no_improve >= early_stop_patience:
                print("Early stopping.")
                model.load_state_dict(best_state)
                break
    return model, train_losses, val_losses


def plot_loss_curve(train_losses, val_losses, recall, ndcg, mlp_hidden, save_path):
    # print the recall and ndcg on the plot

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to {save_path}")


def main():
    data_file = "dataset/ratings.dat"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} 不存在，请下载 MovieLens 1M 数据集。")
    num_negatives = 3
    train_samples, val_samples, test_samples, num_users, num_items, train_positive_by_user = preprocess_data(
        data_file, min_rating=4, num_negatives=num_negatives, test_ratio=0.15, val_ratio=0.15)

    print(f"train samples: {len(train_samples)}, val samples: {len(val_samples)}, test samples: {len(test_samples)}")
    print(f"#users: {num_users}, #item: {num_items}")

    train_dataset = NCFDataset(train_samples)
    val_dataset = NCFDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_hidden=[128, 63, 32]
    model = NCF(num_users, num_items, gmf_dim=16, mlp_dim=64, mlp_hidden=mlp_hidden, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, device,
                                                  epochs=40, early_stop_patience=5)

    recall, ndcg = evaluate_model(model, train_positive_by_user, test_samples, num_users, num_items,
                                  k=10, num_negatives=99, device=device)
    print(f"Test Recall@10: {recall:.4f}, Test NDCG@10: {ndcg:.4f}, mlp_hidden: {mlp_hidden}, num_negatives: {num_negatives}")
    plot_loss_curve(train_losses, val_losses, recall, ndcg, mlp_hidden, save_path=f"loss_curve_{mlp_hidden}_negatives_{num_negatives}.png")


if __name__ == "__main__":
    main()
