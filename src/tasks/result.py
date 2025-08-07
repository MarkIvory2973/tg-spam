from . import *

def run(root):
    epochs = len(list(os.scandir(os.path.join(root, "ckpt/tg-spam/"))))
    
    metrics_train_loss_c = []
    metrics_train_accuracy_c = []
    metrics_eval_loss_c = []
    metrics_eval_accuracy_c = []
    
    for epoch in range(epochs):
        ckpt = torch.load(os.path.join(root, f"ckpt/tg-spam/ckpt.{epoch+1}.pth"), device)
        metric_train_loss_c = ckpt["metrics"]["train"]["loss"]["c"]
        metric_train_accuracy_c = ckpt["metrics"]["train"]["accuracy"]["c"]
        metric_eval_loss_c = ckpt["metrics"]["eval"]["loss"]["c"]
        metric_eval_accuracy_c = ckpt["metrics"]["eval"]["accuracy"]["c"]
        
        ckpt = None
        metrics_train_loss_c.append(metric_train_loss_c)
        metrics_train_accuracy_c.append(metric_train_accuracy_c)
        metrics_eval_loss_c.append(metric_eval_loss_c)
        metrics_eval_accuracy_c.append(metric_eval_accuracy_c)
    
    figure, axes = plt.subplots(1, 2)
    
    figure.suptitle("TG Spam Results")
    
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel("loss")
    axes[0].set_xlim(1, epochs)
    axes[0].grid()
    
    axes[0].plot(range(1, epochs+1), metrics_train_loss_c)
    axes[0].plot(range(1, epochs+1), metrics_eval_loss_c)
    axes[0].scatter(range(1, epochs+1), metrics_train_loss_c)
    axes[0].scatter(range(1, epochs+1), metrics_eval_loss_c)
    axes[0].legend(["Train", "Test"])
    
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel("accuracy")
    axes[1].set_xlim(1, epochs)
    axes[1].grid()
    
    axes[1].plot(range(1, epochs+1), metrics_train_accuracy_c)
    axes[1].plot(range(1, epochs+1), metrics_eval_accuracy_c)
    axes[1].scatter(range(1, epochs+1), metrics_train_accuracy_c)
    axes[1].scatter(range(1, epochs+1), metrics_eval_accuracy_c)
    axes[1].legend(["Train", "Test"])
    
    plt.show()
