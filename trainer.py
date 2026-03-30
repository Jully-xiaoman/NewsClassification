import torch

def evaluate(model, dataloader, device):
    model.eval()  # 切换到评估模式，关闭dropout

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # 不计算梯度（更快更省内存）
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
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

def train(model, train_dataloader, dev_dataloader, optimizer, device, config):
    best_dev_acc = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        dev_acc, dev_loss = evaluate(model, dev_dataloader, device)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), config["best_model_path"])
            print("最佳模型已保存！")

def test(model, test_dataloader, device, config):
    model.load_state_dict(torch.load(config["best_model_path"], map_location=device))
    model.to(device)

    test_acc, test_loss = evaluate(model, test_dataloader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
