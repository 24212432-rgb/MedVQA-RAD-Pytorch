import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from transformers import AutoTokenizer
from torchvision import transforms

# === Key Imports ===
from src.dataset_advanced import VQARADSeqDataset 
from src.model_advanced import VQAModelAdvanced
from src import config

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(predictions, ground_truths):
    """Calculate accuracy based on exact string match."""
    correct = 0
    total = len(predictions)
    for p, g in zip(predictions, ground_truths):
        # Basic cleaning
        p_clean = str(p).lower().strip().replace('.', '')
        g_clean = str(g).lower().strip().replace('.', '')
        if p_clean == g_clean or g_clean in p_clean:
            correct += 1
    acc = correct / total if total > 0 else 0
    return acc

def run_full_evaluation(model_path):
    print(f"Loading model from: {model_path} ...")
    
    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Initialize Model
    # Must match training configuration
    try:
        model = VQAModelAdvanced(vocab_size=len(tokenizer.vocab), 
                                 hidden_dim=config.HIDDEN_DIM, 
                                 dropout_p=0.3).to(DEVICE)
    except AttributeError:
        print("Warning: config.HIDDEN_DIM not found, using default 512.")
        model = VQAModelAdvanced(vocab_size=len(tokenizer.vocab), 
                                 hidden_dim=512, 
                                 dropout_p=0.3).to(DEVICE)

    # 3. Load Weights
    if not os.path.exists(model_path):
        print(f"❌ Error: File {model_path} not found!")
        return

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    model.eval()
    
    # 4. Prepare Test Data
    print("Preparing Test Dataset...")
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=test_transform
    )
    
    # Replicate Split Logic (Seed 42)
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(full_dataset))
    test_indices = indices[split:] 
    
    test_subset = Subset(full_dataset, test_indices)
    # Batch size must be 1 for easy evaluation logic with generate_answer
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    print(f"\n[2/4] Starting Full Inference (Test Set Size: {len(test_indices)})...")
    
    predictions = []
    ground_truths = []
    images_for_viz = []
    error_log = []

    # Get special tokens for generation
    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id
    
    with torch.no_grad():
        for i, (img, q_ids, a_ids) in enumerate(test_loader):
            img = img.to(DEVICE)
            q_ids = q_ids.to(DEVICE)
            
            # Decode Question and GT Answer
            q_text = tokenizer.decode(q_ids[0], skip_special_tokens=True)
            a_text = tokenizer.decode(a_ids[0], skip_special_tokens=True)
            
            # === FIX: Use generate_answer instead of model() ===
            try:
                # generate_answer returns a list of lists of token IDs
                generated_seqs = model.generate_answer(img, q_ids, bos_idx, eos_idx)
                
                # Decode the first sequence (since batch_size=1)
                pred_text = tokenizer.decode(generated_seqs[0], skip_special_tokens=True)
            
            except Exception as e:
                print(f"Error generating answer for ID {i}: {e}")
                pred_text = "<error>"

            # Post-processing
            pred_text = pred_text.replace('[PAD]', '').strip()

            predictions.append(pred_text)
            ground_truths.append(a_text)
            
            # Log Errors
            if pred_text.lower() not in a_text.lower():
                error_log.append({"ID": i, "Question": q_text, "GT": a_text, "Prediction": pred_text})
            
            # Save random samples for visualization
            if len(images_for_viz) < 5 and random.random() < 0.1:
                images_for_viz.append((img.cpu(), q_text, a_text, pred_text))
                
            if i % 50 == 0:
                print(f"Processed {i}/{len(test_indices)}...")

    # 5. Calculate Metrics
    print("\n[3/4] Calculating Metrics...")
    acc = calculate_metrics(predictions, ground_truths)
    print(f"🎯 Final Accuracy: {acc*100:.2f}%")
    
    # 6. Generate Reports
    print("\n[4/4] Generating Reports...")
    
    if error_log:
        df = pd.DataFrame(error_log)
        df.to_csv("error_analysis_report.csv", index=False)
        print(f"📝 Error report saved to: error_analysis_report.csv ({len(error_log)} errors found)")
        print("Sample Errors:")
        print(df.head(3))
        
    if images_for_viz:
        plt.figure(figsize=(15, 6))
        for idx, (img_tensor, q, gt, pred) in enumerate(images_for_viz):
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            ax = plt.subplot(1, 5, idx+1)
            ax.imshow(img_np)
            ax.axis('off')
            
            color = 'green' if pred.lower() in gt.lower() else 'red'
            q_short = (q[:30] + '..') if len(q) > 30 else q
            
            ax.set_title(f"Q: {q_short}\nGT: {gt}\nPred: {pred}", color=color, fontsize=9)
        
        plt.tight_layout()
        plt.savefig('final_visual_report.png')
        print("🖼️  Visual report saved to: final_visual_report.png")

if __name__ == "__main__":
    run_full_evaluation('medvqa_ultimate_final.pth')
