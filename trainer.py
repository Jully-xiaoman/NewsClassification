import torch
import swanlab

import torch
import swanlab

class Trainer:
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config

    def evaluate(self, dataloader):
        self.model.eval()  # 切换到评估模式，关闭dropout

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():  # 评估模式下不计算梯度
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                logits = outputs.logits  # (batch_size, num_labels)
                loss = outputs.loss
                # 取最大值的类别作为预测结果
                # 按行返回最大值的下标，dim=1表示按照行
                preds = torch.argmax(logits, dim=1)
                total_loss += loss.item()
                # 统计正确数量
                # .sum()：等于1的地方全部加起来
                total_correct += (preds == labels).sum().item()
                # 统计张量第0维的长度
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataloader)

        return accuracy, avg_loss

    def train(self, train_dataloader, dev_dataloader):
        best_dev_acc = 0.0

        for epoch in range(self.config["epochs"]):
            self.model.train()
            total_loss = 0

            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            dev_acc, dev_loss = self.evaluate(dev_dataloader)

            # 记录每个 epoch 的训练/验证指标
            swanlab.log({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
                "dev/loss": dev_loss,
                "dev/acc": dev_acc
            })

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_dev_acc": best_dev_acc,
                }, self.config["best_model_path"])
                print("最佳模型已保存！")

    def test(self, test_dataloader):
        self.model.load_state_dict(torch.load(self.config["best_model_path"], map_location=self.device))
        self.model.to(self.device)

        test_acc, test_loss = self.evaluate(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
