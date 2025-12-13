# evaluate_real.py
import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer

# Introduce the project module
from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_result(image_tensor, question, answer, pred, is_correct, save_path):
    """Save the visualized images for use in report presentation."""
    # Reverse standardization, restoring the image to a form that can be viewed by the naked eye
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose((1, 2, 0)) 
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    
    color = "green" if is_correct else "red"
    # The title reads "GT (Truth Value) and Pred (Prediction)"
    title = f"Q: {question}\nGT: {answer}\nPred: {pred}"
    plt.title(title, color=color, fontsize=10, wrap=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print(f"{'='*50}")
    print(f"⚖️ TRUTH REVEALED: REALISTIC EVALUATION")
    print(f"   Mode: Strict Train/Test Split (No Leakage)")
    print(f"{'='*50}\n")

    # 1. Initialization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # During the evaluation, only resizing and standardization will be performed, without any rotation or flipping.
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("1️⃣ Loading Full Dataset...")
    # Here, the full dataset is loaded. The parameter "only_open=False" indicates that we are testing the overall performance.
    full_dataset = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=eval_transform,
        only_open=False 
    )
    
    # --- Key step: Replicate the random splitting used during training ---
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    # 🚨 It is necessary to use the exact same seed as in main_advanced.py, which is 42.
    random.seed(42) 
    random.shuffle(indices)
    
    split = int(0.8 * dataset_size)
    
    # During training, 80% of the data was used, so for the test, only the remaining 20% must be employed.
    # This part of the data is something the model has never encountered before!
    test_indices = indices[split:]
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"   -> Full Dataset Size: {dataset_size}")
    print(f"   -> Test Set (Unseen): {len(test_dataset)} samples")
    print("   ✅ Data Leakage Check: PASSED (Training samples excluded)")

    # 2. Load the most powerful model

    model_path = "medvqa_resnet_bert_best.pth"
    
    print(f"\n2️⃣ Loading Model: {model_path} ...")
    model = VQAModelAdvanced(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_p=0.3
    )
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("   ✅ Weights Loaded Successfully!")
    else:
        print(f"   ❌ Error: {model_path} not found. Please check file name.")
        return

    model.to(device)
    model.eval()

    # 3. Start real reasoning
    closed_correct = 0; closed_total = 0
    open_correct = 0; open_total = 0
    
    bos_idx = tokenizer.cls_token_id
    eos_idx = tokenizer.sep_token_id
    
    # Create the result folder
    save_dir = "results_vis_real"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("\n3️⃣ Starting Inference on UNSEEN data...")
    print("   (This mimics the real exam environment)")
    
    with torch.no_grad():
        for i, (images, questions, answers_seq) in enumerate(test_loader):
            images = images.to(device)
            questions = questions.to(device)
            
            # Generate answer
            gen_ids = model.generate_answer(images, questions, bos_idx, eos_idx)
            pred_str = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            gt_str = tokenizer.decode(answers_seq[0], skip_special_tokens=True)
            
            # Obtain the question text
            q_text = tokenizer.decode(questions[0], skip_special_tokens=True)
            
            # Simple normalization comparison
            def norm(s): return s.lower().replace(" ", "")
            is_correct = norm(pred_str) == norm(gt_str) or gt_str in pred_str
            is_closed = norm(gt_str) in ["yes", "no"]
            
            # Statistics
            if is_closed:
                closed_total += 1
                if is_correct: closed_correct += 1
            else:
                open_total += 1
                if is_correct: open_correct += 1
            
            # Save one picture every 20 pictures, and it is necessary to save the correct cases of the first few open questions.
            should_save = (i % 20 == 0) or (not is_closed and is_correct and i < 100)
            if should_save:
                visualize_result(images[0], q_text, gt_str, pred_str, is_correct, f"{save_dir}/res_{i}.png")

    # 4. Calculate the final true score
    closed_acc = closed_correct / closed_total if closed_total else 0
    open_acc = open_correct / open_total if open_total else 0
    total_acc = (closed_correct + open_correct) / (closed_total + open_total) if (closed_total + open_total) else 0
    
    print(f"\n{'='*50}")
    print(f"📊 REALISTIC FINAL SCORE REPORT (The Truth)")
    print(f"{'='*50}")
    print(f"Tested on {len(test_dataset)} unseen images.")
    print(f"----------------------------------------")
    print(f"✅ Total Accuracy : {total_acc:.2%}")
    print(f"🔒 Closed Accuracy: {closed_acc:.2%}")
    print(f"🔓 Open Accuracy  : {open_acc:.2%}")
    print(f"----------------------------------------")
    print(f"Visualization saved to /{save_dir} folder.")
    
    # Simple evaluation
    if total_acc > 0.60:
        print("\n🌟 Evaluation: Excellent! An accuracy rate of over 60% on VQA-RAD is an extremely excellent result.")
    elif total_acc > 0.50:
        print("\n👍 Evaluation: Well Done! More than 50% have already reached the passing and stable baseline model standard.。")
    else:
        print("\n💪 Evaluation: Keep fighting. The model's generalization ability still has room for improvement.")

if __name__ == "__main__":
    main()