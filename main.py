import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


theme2id = {
    '动力': 0, '价格': 1, '内饰': 2, '配置': 3, '安全性': 4,
    '外观': 5, '操控': 6, '油耗': 7, '空间': 8, '舒适性': 9
}
sentiment2id = {'-1': 0, '0': 1, '1': 2}  # 负, 中, 正


class MultiTaskBertDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                text = parts[0]
                labels = parts[1:]

                # 初始化
                theme_labels = [0] * len(theme2id)  # 多标签
                sentiment_sum = 0  # 情感累加
                count = 0

                for lab in labels:
                    if '#' in lab:
                        theme, senti = lab.split('#')
                        if theme in theme2id:
                            theme_labels[theme2id[theme]] = 1
                        sentiment_sum += int(senti)
                        count += 1

                # 规则：情感之和 >0 为正，<0 为负，==0 为中性
                if count == 0:
                    sentiment = sentiment2id['0']
                else:
                    avg_senti = sentiment_sum
                    if avg_senti > 0:
                        sentiment = sentiment2id['1']
                    elif avg_senti < 0:
                        sentiment = sentiment2id['-1']
                    else:
                        sentiment = sentiment2id['0']
                self.data.append((text, theme_labels, sentiment))
        self.data=self.data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, theme_labels, sentiment = self.data[idx]
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'theme_labels': torch.tensor(theme_labels, dtype=torch.float),
            'sentiment_label': torch.tensor(sentiment, dtype=torch.long)
        }



#
# dataset = MultiTaskBertDataset('train.txt', tokenizer, max_len=128)
#
# # DataLoader 示例
#
# loader = DataLoader(dataset, batch_size=16, shuffle=True)

# for batch in loader:
#     print(batch['sentiment'])
#     print(batch['input_ids'].shape)
#     print(batch['attention_mask'].shape)
#     print(batch['theme_labels'].shape)  # 多标签
#     print(batch['sentiment'].shape)     # 多分类
#     break



class BertMultiTaskModel(nn.Module):
    def __init__(self, bert_path, num_sentiment_classes=3, num_topic_classes=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.5)

        # 情感识别头（单标签多分类）
        self.sentiment_fc = nn.Linear(hidden_size, num_sentiment_classes)

        # 主题识别头（多标签分类）
        self.topic_fc = nn.Linear(hidden_size, num_topic_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)  # shape: [batch_size, hidden_size]

        sentiment_logits = self.sentiment_fc(pooled_output)
        topic_logits = self.topic_fc(pooled_output)

        return sentiment_logits, topic_logits







def train(model, train_loader, dev_loader, config):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始训练")
    device = config["device"]
    model = model.to(device)

    # 损失函数
    criterion_sentiment = nn.CrossEntropyLoss()
    criterion_topic = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss, total_acc = 0, 0


        for idx, batch in enumerate(train_loader):
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)  # shape: [B]
            topic_labels = batch["theme_labels"].to(device).float()  # shape: [B, C]

            optimizer.zero_grad()
            sentiment_logits, topic_logits = model(input_ids, attention_mask)

            loss1 = criterion_sentiment(sentiment_logits, sentiment_labels)
            loss2 = criterion_topic(topic_logits, topic_labels)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(sentiment_logits, dim=1)
            total_acc += (preds == sentiment_labels).sum().item()


            print(f"[{current_time}] Epoch [{epoch + 1}/{config['epochs']}], "
                  f"Batch [{idx + 1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / (len(train_loader.dataset))
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        val_loss, val_acc = evaluate(model, dev_loader, criterion_sentiment, criterion_topic, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 可视化 Loss & Accuracy
    plot_curve(train_losses, val_losses, 'Loss')
    plot_curve(train_accs, val_accs, 'Accuracy')


def evaluate(model, loader, criterion_sentiment, criterion_topic, device):
    model.eval()
    total_loss, total_acc = 0, 0
    # 初始化存储所有预测和标签：
    all_sentiment_preds = []
    all_sentiment_labels = []

    all_topic_preds = []
    all_topic_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            topic_labels = batch["theme_labels"].to(device).float()

            sentiment_logits, topic_logits = model(input_ids, attention_mask)
            # 损失
            loss1 = criterion_sentiment(sentiment_logits, sentiment_labels)
            loss2 = criterion_topic(topic_logits, topic_labels)
            loss = loss1 + loss2
            total_loss += loss.item()

            # 情感（多分类）
            preds = torch.argmax(sentiment_logits, dim=1)
            all_sentiment_preds.extend(preds.cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

            # 主题（多标签）
            topic_pred_binary = torch.sigmoid(topic_logits) > 0.5  # 转为0/1
            all_topic_preds.extend(topic_pred_binary.cpu().numpy())
            all_topic_labels.extend(topic_labels.cpu().numpy())

            preds = torch.argmax(sentiment_logits, dim=1)
            total_acc += (preds == sentiment_labels).sum().item()

    # === 情感指标 ===
    print("\n🎯 Sentiment Classification (情感识别)")
    print("Accuracy:", accuracy_score(all_sentiment_labels, all_sentiment_preds))
    print("Precision:", precision_score(all_sentiment_labels, all_sentiment_preds, average='macro'))
    print("Recall:", recall_score(all_sentiment_labels, all_sentiment_preds, average='macro'))
    print("F1 Score:", f1_score(all_sentiment_labels, all_sentiment_preds, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(all_sentiment_labels, all_sentiment_preds))

    # === 主题多标签指标 ===
    print("\n📚 Topic Classification (主题识别)")
    print("Accuracy:", accuracy_score(all_topic_labels, all_topic_preds))
    print("Precision:", precision_score(all_topic_labels, all_topic_preds, average='macro', zero_division=0))
    print("Recall:", recall_score(all_topic_labels, all_topic_preds, average='macro', zero_division=0))
    print("F1 Score:", f1_score(all_topic_labels, all_topic_preds, average='macro', zero_division=0))

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader.dataset)
    return avg_loss, avg_acc


def plot_curve(train_values, val_values, metric):
    plt.plot(train_values, label=f"train_{metric}")
    plt.plot(val_values, label=f"val_{metric}")
    plt.title(metric)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()


config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 2e-5,
    "epochs": 2,
    "batch_size":32,
}

bert_path = 'bert-base-chinese'  # 可换为本地模型路径
model = BertMultiTaskModel(bert_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_dataset = MultiTaskBertDataset('train.txt', tokenizer, max_len=64)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

val_dataset = MultiTaskBertDataset('train.txt', tokenizer, max_len=64)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
train(model, train_loader, val_loader, config)


