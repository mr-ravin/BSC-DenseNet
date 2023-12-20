import matplotlib.pyplot as plt

def plot_graph(root, x_data, densenet_epoch_tr_loss, bsc_densenet_epoch_tr_loss, densenet_epoch_vl_loss, bsc_densenet_epoch_vl_loss, densenet_epoch_tr_acc, bsc_densenet_epoch_tr_acc, densenet_epoch_vl_acc, bsc_densenet_epoch_vl_acc):
    tag_1="Densenet 121"
    tag_2="BSC-Densenet 121"
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(50,50))    
    ax1.set_title("Train Losses")
    ax1.set_xlabel("Epochs")
    ax1.set_xticks(x_data)
    ax1.plot(densenet_epoch_tr_loss, "r--", label=tag_1+" Train Loss")
    ax1.plot(bsc_densenet_epoch_tr_loss, "g--", label=tag_2+" Train Loss")
    ax1.plot(densenet_epoch_vl_loss, "r-o", label=tag_1+" Val Loss")
    ax1.plot(bsc_densenet_epoch_vl_loss, "g-o", label=tag_2+" Val Loss")
    ax1.legend([tag_1+" Train Loss", tag_2+" Train Loss", tag_1+" Val Loss", tag_2+" Val Loss"], loc='upper right')

    ax2.set_title("Val Losses")
    ax2.set_xlabel("Epochs")
    ax2.set_xticks(x_data)
    ax2.plot(densenet_epoch_vl_loss, "r-o", label=tag_1+" Val Loss")
    ax2.plot(bsc_densenet_epoch_vl_loss, "g-o", label=tag_2+" Val Loss")
    ax2.legend([tag_1+" Val Loss", tag_2+" Val Loss"], loc='upper right')

    ax3.set_title("Train Accuracy")
    ax3.set_xlabel("Epochs")
    ax3.set_xticks(x_data)
    ax3.plot(densenet_epoch_tr_acc, "r--", label=tag_1+" Train Acc")
    ax3.plot(bsc_densenet_epoch_tr_acc, "g--", label=tag_2+" Train Acc")
    ax3.legend([tag_1+" Train Acc", tag_2+" Train Acc"], loc='upper right')

    ax4.set_title("Val Accuracy")
    ax4.set_xlabel("Epochs")
    ax4.set_xticks(x_data)
    ax4.plot(densenet_epoch_vl_acc, "r--", label=tag_1+" Val Acc")
    ax4.plot(bsc_densenet_epoch_vl_acc, "g--", label=tag_2+" Val Acc")
    ax4.legend([tag_1+" Val Acc", tag_2+" Val Acc"], loc='upper right')
    f.tight_layout(pad=2.0)
    plt.savefig(root+'overall_analysis.png')