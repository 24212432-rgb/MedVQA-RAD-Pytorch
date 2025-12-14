import torch
import torch.nn as nn

def train_model(model, train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    num_epochs = config.NUM_EPOCHS
    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Training loop
        for images, questions, labels in train_loader:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, questions)           # Forward propagation
            loss = criterion(outputs, labels)           # Calculate the loss of the current batch
            loss.backward()                             # Backpropagation
            # Preventing gradient explosion: Clipping gradient norm
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            # The number of correct statistical predictions
            _, predicted = torch.max(outputs, dim=1)    # Take the index of the maximum value in each row as the predicted category.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total

        # Testing and Evaluation Phase
        model.eval()
        total_test = 0
        correct_test = 0
        closed_total = closed_correct = 0
        open_total = open_correct = 0
        # Since it might be necessary to determine the type for each sample individually, an index is used here to track the position of the samples.
        sample_index = 0
        with torch.no_grad():
            for images, questions, labels in test_loader:
                images, questions, labels = images.to(device), questions.to(device), labels.to(device)
                outputs = model(images, questions)
                _, predicted = torch.max(outputs, dim=1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                # Traverse each sample in the current batch to calculate the accuracy rates of closed type/open type.
                for j in range(labels.size(0)):
                    # Based on the original data, determine whether this sample is a yes/no issue
                    ans_text = test_loader.dataset.data[sample_index + j]["answer"].strip().lower()
                    if ans_text in ["yes", "no"]:
                        closed_total += 1
                        if predicted[j].item() == labels[j].item():
                            closed_correct += 1
                    else:
                        open_total += 1
                        if predicted[j].item() == labels[j].item():
                            open_correct += 1
                sample_index += labels.size(0)
        test_acc = correct_test / total_test if total_test > 0 else 0.0
        closed_acc = closed_correct / closed_total if closed_total > 0 else 0.0
        open_acc = open_correct / open_total if open_total > 0 else 0.0

        # Print the results of this round
        print(f"Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Test Acc: {test_acc:.4f}  Closed Acc: {closed_acc:.4f}  Open Acc: {open_acc:.4f}\n")
        # Save the model with the best performance
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "medvqa_baseline_best.pth")
            print(f"✅ Save the best model (Epoch {epoch}, Test Acc={best_test_acc:.4f})\n")
    # Training completed. Save the model of the last round.
    torch.save(model.state_dict(), "medvqa_baseline_last.pth")
    print("Training completed! Best test set accuracy: {:.4f}".format(best_test_acc))

