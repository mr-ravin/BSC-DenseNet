import matplotlib.pyplot as plt
import pandas as pd

def plot_graph(
    root, x_data,
    densenet_epoch_tr_loss, bsc_densenet_epoch_tr_loss,
    densenet_epoch_vl_loss, bsc_densenet_epoch_vl_loss,
    densenet_epoch_tr_acc, bsc_densenet_epoch_tr_acc,
    densenet_epoch_vl_acc, bsc_densenet_epoch_vl_acc
):
    tag_1 = "Densenet121"
    tag_2 = "BSC-Densenet121"

    # Create a DataFrame with all values
    df = pd.DataFrame({
        'Epoch': x_data,
        f'{tag_1}_Train_Loss': densenet_epoch_tr_loss,
        f'{tag_2}_Train_Loss': bsc_densenet_epoch_tr_loss,
        f'{tag_1}_Val_Loss': densenet_epoch_vl_loss,
        f'{tag_2}_Val_Loss': bsc_densenet_epoch_vl_loss,
        f'{tag_1}_Train_Acc': densenet_epoch_tr_acc,
        f'{tag_2}_Train_Acc': bsc_densenet_epoch_tr_acc,
        f'{tag_1}_Val_Acc': densenet_epoch_vl_acc,
        f'{tag_2}_Val_Acc': bsc_densenet_epoch_vl_acc,
    })

    # Save to CSV
    csv_path = './evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)

    # Plotting as before
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(50,50))    
    ax1.set_title("Train Losses")
    ax1.set_xlabel("Epochs")
    ax1.set_xticks(x_data)
    ax1.plot(densenet_epoch_tr_loss, "r--", label=tag_1+" Train Loss")
    ax1.plot(bsc_densenet_epoch_tr_loss, "g--", label=tag_2+" Train Loss")
    ax1.plot(densenet_epoch_vl_loss, "r-o", label=tag_1+" Val Loss")
    ax1.plot(bsc_densenet_epoch_vl_loss, "g-o", label=tag_2+" Val Loss")
    ax1.legend(loc='upper right')

    ax2.set_title("Val Losses")
    ax2.set_xlabel("Epochs")
    ax2.set_xticks(x_data)
    ax2.plot(densenet_epoch_vl_loss, "r-o", label=tag_1+" Val Loss")
    ax2.plot(bsc_densenet_epoch_vl_loss, "g-o", label=tag_2+" Val Loss")
    ax2.legend(loc='upper right')

    ax3.set_title("Train Accuracy")
    ax3.set_xlabel("Epochs")
    ax3.set_xticks(x_data)
    ax3.plot(densenet_epoch_tr_acc, "r--", label=tag_1+" Train Acc")
    ax3.plot(bsc_densenet_epoch_tr_acc, "g--", label=tag_2+" Train Acc")
    ax3.legend(loc='upper right')

    ax4.set_title("Val Accuracy")
    ax4.set_xlabel("Epochs")
    ax4.set_xticks(x_data)
    ax4.plot(densenet_epoch_vl_acc, "r--", label=tag_1+" Val Acc")
    ax4.plot(bsc_densenet_epoch_vl_acc, "g--", label=tag_2+" Val Acc")
    ax4.legend(loc='upper right')

    f.tight_layout(pad=2.0)
    plt.savefig(root + 'overall_analysis.png')
