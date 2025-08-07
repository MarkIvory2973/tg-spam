from . import *

def run(root, batch_size, learning_rate, gamma, epochs):
    dataset_train = datasets.load(os.path.join(root, "data/tg-spam/"), True, batch_size)
    dataset_eval = datasets.load(os.path.join(root, "data/tg-spam/"), False, batch_size)
    
    model_c = models.Model_C().to(device)
    
    criterion_c = nn.BCELoss()
    
    optimizer_c = optim.Adam(model_c.parameters(), learning_rate)
    
    scheduler_c = lr_scheduler.ExponentialLR(optimizer_c, gamma)
    
    metric_train_loss_c = metrics.Metric_Loss()
    metric_train_accuracy_c = metrics.Metric_Accuracy()
    metric_eval_loss_c = metrics.Metric_Loss()
    metric_eval_accuracy_c = metrics.Metric_Accuracy()
    
    for epoch in range(epochs):
        model_c.train()
        
        metric_train_loss_c.reset()
        metric_train_accuracy_c.reset()
        
        for step, (inputs, targets) in enumerate(dataset_train):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs_c = model_c(inputs)
            
            loss_c = criterion_c(outputs_c, targets.reshape(-1, 1))
            
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
            
            metric_train_loss_c.step(loss_c)
            metric_train_accuracy_c.step(outputs_c, targets)
            
            print(f"(train) (epoch {epoch+1}/{epochs}) (step {step+1}/{len(dataset_train)}) loss_c={metric_train_loss_c.calc_mean()} accuracy_c={metric_train_accuracy_c.calc_accuracy()}", end="\t\r")
            
        scheduler_c.step()
        
        model_c.eval()
        
        metric_eval_loss_c.reset()
        metric_eval_accuracy_c.reset()
        
        for step, (inputs, targets) in enumerate(dataset_eval):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs_c = model_c(inputs)
            
            loss_c = criterion_c(outputs_c, targets.reshape(-1, 1))
            
            metric_eval_loss_c.step(loss_c)
            metric_eval_accuracy_c.step(outputs_c, targets)
            
            print(f"(eval) (epoch {epoch+1}/{epochs}) (step {step+1}/{len(dataset_eval)}) loss_c={metric_eval_loss_c.calc_mean()} accuracy_c={metric_train_accuracy_c.calc_accuracy()}", end="\t\r")
        
        ckpt = {
            "epoch": epoch + 1,
            "models": {
                "c": model_c.state_dict()
            },
            "optimizers": {
                "c": optimizer_c.state_dict()
            },
            "scheduler": {
                "c": scheduler_c.state_dict()
            },
            "metrics": {
                "train": {
                    "loss": {
                        "c": metric_train_loss_c.calc_mean()
                    },
                    "accuracy": {
                        "c": metric_train_accuracy_c.calc_accuracy()
                    }
                },
                "eval": {
                    "loss": {
                        "c": metric_eval_loss_c.calc_mean()
                    },
                    "accuracy": {
                        "c": metric_eval_accuracy_c.calc_accuracy()
                    }
                }
            }
        }
        torch.save(ckpt, os.path.join(root, f"ckpt/tg-spam/ckpt.{epoch+1}.pth"))
