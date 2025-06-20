import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import random


class Config:
    MODEL_PATH = r"Japanese_L-12_H-768_A-12_E-30_BPE_WWM"
    DATA_PATH = r"bert_text.xlsx"
    RESULT_PATH = r"kyotoresult/"
    NUM_LABELS = 2  
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    SEED = 42
    CLASS_NAMES = {0: "label", 1: "label2"} 
    TEST_RATIO = 0.2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_and_split_data():
    
    df = pd.read_excel(Config.DATA_PATH)
    
    
    assert 'text' in df.columns, "数据文件缺少'text'列"
    assert 'target' in df.columns, "数据文件缺少'target'列"
    assert set(df['target'].unique()).issubset({0,1}), "target值必须为0/1"  
    
    df = df.dropna(subset=['text', 'target'])
    df = df[df['text'].str.strip().astype(bool)]
    df['tokenized_text'] = df['text'].str.split()
    
    print("\n原始数据分布:")
    print(df['target'].value_counts().sort_index())
    
    min_samples = Config.NUM_LABELS * 3
    if len(df) < min_samples:
        raise ValueError(f"数据集过小（总样本数={len(df)}），需要至少{min_samples}个样本")
    
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=Config.TEST_RATIO,
            stratify=df['target'],
            random_state=Config.SEED
        )
    except ValueError:
        print(" 分层分割失败，使用随机分割")
        train_df, val_df = train_test_split(
            df,
            test_size=Config.TEST_RATIO,
            random_state=Config.SEED
        )
    
    return train_df, val_df

def initialize_model():
    """初始化模型"""
    model_dir = Path(Config.MODEL_PATH)
    
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"缺失模型文件: {missing}")
    
    config = AutoConfig.from_pretrained(
        model_dir,
        num_labels=Config.NUM_LABELS,  
        id2label=Config.CLASS_NAMES,
        hidden_dropout_prob=0.2
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        do_basic_tokenize=False
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = torch.nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已加载到 {device}")
    
    return tokenizer, model

class JLPTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_data(tokenizer, texts):
    return tokenizer(
        texts,
        truncation=True,
        max_length=Config.MAX_LENGTH,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True,
        is_split_into_words=True
    )

def compute_metrics(p):
    predictions = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, predictions),
        **classification_report(
            labels, predictions,
            target_names=Config.CLASS_NAMES.values(),
            output_dict=True,
            zero_division=0
        )['macro avg']
    }

def plot_confusion_matrix(labels, preds, save_path=None):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=Config.CLASS_NAMES.values(),
                yticklabels=Config.CLASS_NAMES.values(),
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    set_seed(Config.SEED)
    train_df, val_df = load_and_split_data()
    print(f"\n总样本数：{len(train_df)+len(val_df)}")
    print(f"训练样本：{len(train_df)} | 验证样本：{len(val_df)}")
    
    tokenizer, model = initialize_model()
    
    train_encodings = tokenize_data(tokenizer, train_df['tokenized_text'].tolist())
    val_encodings = tokenize_data(tokenizer, val_df['tokenized_text'].tolist())
    
    train_dataset = JLPTDataset(train_encodings, train_df['target'].values)
    val_dataset = JLPTDataset(val_encodings, val_df['target'].values)

    training_args = TrainingArguments(
        output_dir=os.path.join(Config.MODEL_PATH, "training_results"),
        evaluation_strategy="epoch",
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=3e-5,
        weight_decay=0.01,
        num_train_epochs=Config.NUM_EPOCHS,
        logging_dir=os.path.join(Config.MODEL_PATH, "logs"),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1-score",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=Config.SEED,
        report_to="none",
        gradient_accumulation_steps=2,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        lr_scheduler_type='cosine_with_restarts',
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    try:
        print("\n开始训练...")
        trainer.train()
        
        best_model_path = os.path.join(Config.RESULT_PATH, "best_model_kyoto_L12")
        trainer.save_model(best_model_path)
        print(f"\n最佳模型已保存至：{best_model_path}")
        
    except Exception as e:
        print(f"\n训练异常：{str(e)}")
        emergency_path = os.path.join(Config.RESULT_PATH, "emergency_model_kyoto")
        model.save_pretrained(emergency_path)
        print(f"紧急模型已保存至：{emergency_path}")
        raise

    print("\n生成评估报告...")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    report = classification_report(
        labels, preds, 
        target_names=Config.CLASS_NAMES.values(),
        digits=4
    )
    print("\n详细分类报告：")
    print(report)
    
    plot_confusion_matrix(labels, preds, 
                         save_path=os.path.join(Config.RESULT_PATH, "confusion_matrix_kyoto.png"))
    print("混淆矩阵已保存")
    
    report_df = pd.DataFrame(classification_report(
        labels, preds,
        target_names=Config.CLASS_NAMES.values(),
        output_dict=True
    )).transpose()
    report_path = os.path.join(Config.RESULT_PATH, "classification_report_kyoto.csv")
    report_df.to_csv(report_path)
    print(f"评估报告已保存至：{report_path}")

if __name__ == "__main__":
    main()      500

    
#accuracy:0.6810