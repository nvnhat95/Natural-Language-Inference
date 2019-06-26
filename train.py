import torch
import random

def evaluate(model, data, device):
    corrects = 0
    num_samples = 0
    batches = data.get_batches()
    for i, batch in enumerate(batches):
        source, target, label, source_POS, target_POS = batch
        source = source.to(device)
        target = target.to(device)
        label = label.to(device)
        source_POS = source_POS.to(device)
        target_POS = target_POS.to(device)
        
        y_hat = model(source, target, source_POS, target_POS)
        corrects += torch.sum(torch.argmax(y_hat, dim=1) == label).item()
        num_samples += target.shape[0]
    return corrects / num_samples
        

def run(model, optimizer, train_data, dev_data, test_data, max_epochs, device, model_path):
    accs = []
    best_acc_test = 0
    for epoch in range(1, max_epochs + 1):
        print("epoch: ", epoch)
        train_batches = train_data.get_batches()
        random.shuffle(train_batches)
        model.train()
        corrects = 0
        num_samples = 0
        
        for i, batch in enumerate(train_batches):
            source, target, label, source_POS, target_POS = batch
            source = source.to(device)
            target = target.to(device)
            label = label.to(device)
            source_POS = source_POS.to(device)
            target_POS = target_POS.to(device)

            optimizer.zero_grad()
            y_hat = model(source, target, source_POS, target_POS)
            
            # calculate the accuracy on training batch
            corrects += torch.sum(torch.argmax(y_hat, dim=1) == label).item()
            num_samples += target.shape[0]
            
            loss = torch.nn.CrossEntropyLoss()(y_hat, label)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print("{}/{}\tLoss: {:.5f}\tAccuracy: {:.5f}".format(i, len(train_batches), loss, corrects / num_samples))
        
        # Calculate val accuracy        
        model.eval()
        print("Validation accuracy: {:.5f}".format(evaluate(model, dev_data, device)))
        
        if corrects / num_samples > best_acc_test:
            print("Save model at {}".format(model_path))
            torch.save(model, model_path)
            
    model = torch.load(model_path)
    print("Test accuracy: {:.5f}".format(evaluate(model, test_data, device)))