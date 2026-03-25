from transformers import BertForSequenceClassification

def create_model(config):
    model = BertForSequenceClassification.from_pretrained(
        config["pretrained_model"],
        num_labels=config["num_labels"]
    )
    return model