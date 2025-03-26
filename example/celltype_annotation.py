import scvi
import scanpy as sc
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from model import MambaCellClassifier, DataCollatorForCellClassification

class PBMCDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx].tolist(), 
            "label": self.y[idx]
        }

adata = scvi.data.pbmc_dataset(save_path="data/")
adata.obs['batch'] = adata.obs['batch'].astype('category')

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)


X = adata.X.toarray()
y = adata.obs['batch'].cat.codes.values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


train_dataset = PBMCDataset(X_train, y_train)
test_dataset = PBMCDataset(X_test, y_test)

collator = DataCollatorForCellClassification(max_length=2000, pad_to_multiple_of=8)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator)

# 定义模型
vocab_size = 2048
num_labels = len(adata.obs['batch'].cat.categories)  

model = MambaCellClassifier(
    vocab_size=vocab_size,
    num_labels=num_labels,
    d_model=128,  
    n_layer=12,  
    dropout=0.02
).to("cuda" if torch.cuda.is_available() else "cpu")


optimizer = AdamW(model.parameters(), lr=5e-4)
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
           
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
       
            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            
            total_correct += (predictions == batch["labels"]).sum().item()
            total_samples += len(batch["labels"])
    

    accuracy = total_correct / total_samples
    
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    return accuracy, macro_f1


device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(100):
    print(f"Epoch {epoch + 1}")
    
    train_loss = train(model, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")
    
    accuracy, macro_f1 = evaluate(model, test_loader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")