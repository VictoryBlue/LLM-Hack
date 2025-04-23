import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import argparse

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
    print("config:", ", ".join(f"{k}={v}" for k, v in config.items()))
    device = config["device"]
    model = model.to(device)

    # 损失函数
    criterion_sentiment = nn.CrossEntropyLoss()
    criterion_topic = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config["epochs"]):
        model.train()


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
            loss = loss1 if config["task"] == "sentiment" else loss2
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if config["task"] == "sentiment":
                preds = torch.argmax(sentiment_logits, dim=1)
                train_accs.append(accuracy_score(sentiment_labels.cpu().numpy(), preds.cpu().numpy()))
            elif config["task"] == "topic":
                topic_pred_binary = torch.sigmoid(topic_logits) > 0.5  # 转为0/1
                train_accs.append(accuracy_score(topic_labels.cpu().numpy(),topic_pred_binary.cpu().numpy()))
            # if config["task"] == "sentiment":
            #     preds = torch.argmax(sentiment_logits, dim=1)
            #     train_accs.append(accuracy_score(sentiment_labels.cpu().numpy(), preds.cpu().numpy()))
            # elif config["task"] == "topic":
            #     topic_pred_binary = torch.sigmoid(topic_logits) > 0.5  # 转为0/1
            #     train_accs.append(accuracy_score(topic_labels.cpu().numpy(), topic_pred_binary.cpu().numpy()))

            if (idx+1)%5==0:
                print(f"[{current_time}] Epoch [{epoch + 1}/{config['epochs']}], "
                      f"Complete [{idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")


        val_loss, val_acc = evaluate(model, dev_loader, criterion_sentiment, criterion_topic, device, config)
        val_losses.extend(val_loss)
        val_accs.extend(val_acc)


    # 可视化 Loss & Accuracy
    plot_curve(train_losses, f'{config["task"]}_train',  'Loss')
    plot_curve(val_losses, f'{config["task"]}_val','Loss')
    plot_curve(train_accs, f'{config["task"]}_train', 'Accuracy')
    plot_curve(val_accs, f'{config["task"]}_val','Accuracy')


def evaluate(model, loader, criterion_sentiment, criterion_topic, device, config):
    model.eval()
    val_losses, val_accs = [], []

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
            loss = loss1 if config["task"]=="sentiment" else loss2


            # # 情感（多分类）
            # preds = torch.argmax(sentiment_logits, dim=1)
            # all_sentiment_preds.extend(preds.cpu().numpy())
            # all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

            # # 主题（多标签）
            # topic_pred_binary = torch.sigmoid(topic_logits) > 0.5  # 转为0/1
            # all_topic_preds.extend(topic_pred_binary.cpu().numpy())
            # all_topic_labels.extend(topic_labels.cpu().numpy())


            val_losses.append(loss.item())
            if config["task"] == "sentiment":
                preds = torch.argmax(sentiment_logits, dim=1)
                all_sentiment_preds.extend(preds.cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
                val_accs.append(accuracy_score(sentiment_labels.cpu().numpy(), preds.cpu().numpy()))
            elif config["task"] == "topic":
                topic_pred_binary = torch.sigmoid(topic_logits) > 0.5  # 转为0/1
                all_topic_preds.extend(topic_pred_binary.cpu().numpy())
                all_topic_labels.extend(topic_labels.cpu().numpy())
                val_accs.append(accuracy_score(topic_labels.cpu().numpy(),topic_pred_binary.cpu().numpy()))

    if config["task"]=="sentiment":
        # === 情感指标 ===
        print("\n🎯 Sentiment Classification (情感识别)")
        print("Accuracy:", accuracy_score(all_sentiment_labels, all_sentiment_preds))
        print("Precision:", precision_score(all_sentiment_labels, all_sentiment_preds, average='macro'))
        print("Recall:", recall_score(all_sentiment_labels, all_sentiment_preds, average='macro'))
        print("F1 Score:", f1_score(all_sentiment_labels, all_sentiment_preds, average='macro'))
        print("Confusion Matrix:\n", confusion_matrix(all_sentiment_labels, all_sentiment_preds))
    elif config["task"]=="topic":
        # === 主题多标签指标 ===
        print("\n📚 Topic Classification (主题识别)")
        print("Accuracy:", accuracy_score(all_topic_labels, all_topic_preds))
        print("Precision:", precision_score(all_topic_labels, all_topic_preds, average='macro', zero_division=0))
        print("Recall:", recall_score(all_topic_labels, all_topic_preds, average='macro', zero_division=0))
        print("F1 Score:", f1_score(all_topic_labels, all_topic_preds, average='macro', zero_division=0))

    return val_losses, val_accs

def plot_curve(values, mode, metric, save_dir="plots"):
    plt.plot(values, label=f"{mode}_{metric}",  color='blue')
    plt.title(f"{mode}_{metric}")

    # 智能判断间距
    max_ticks = 10
    total_points = len(values)
    step = max(1, total_points // max_ticks)  # 至少间隔为1
    plt.xticks(range(0, total_points, step))
    plt.xlabel("batch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"📁 创建目录成功: {save_dir}")
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            save_dir = '.'  # 回退到当前目录
    filename = os.path.join(save_dir, f"{mode}_{metric}.png")
    plt.savefig(filename)
    plt.close()


parser = argparse.ArgumentParser(description="命令行参数传入脚本")
parser.add_argument('--task', type=str, required=True, help="sentiment｜topic")
parser.add_argument('--env', type=str, required=True, help="online｜offline")

args = parser.parse_args()


# env
if args.env == "online":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-5,
        "epochs": 15,
        "batch_size": 128,
        "task": args.task,
        "max_len": 128,
        "weight_decay": 1e-5
    }
elif args.env == "offline":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-5,
        "epochs": 1,
        "batch_size": 5,
        "task": args.task,
        "max_len": 128,
        "weight_decay": 1e-5
    }


bert_path = 'bert-base-chinese'  # 可换为本地模型路径
model = BertMultiTaskModel(bert_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_dataset = MultiTaskBertDataset('train.txt', tokenizer, max_len=config["max_len"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

val_dataset = MultiTaskBertDataset('test.txt', tokenizer, max_len=config["max_len"])
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
train(model, train_loader, val_loader, config)

#云端运行 HF_ENDPOINT=https://hf-mirror.com  python main.py --task topic --env online 2>&1 | tee output.log
