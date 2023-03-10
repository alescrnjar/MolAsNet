import numpy as np
import matplotlib.pyplot as plt

def plot_losses(Loss_mean, output_directory):
    fig = plt.figure(1, figsize=(4, 4))
    plt.plot(np.array(Loss_mean)[:, 0], np.array(Loss_mean)[:, 1],lw=1,c='C0')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(output_directory+'Loss.png',dpi=150)
    #plt.show()                                                                                                 
    plt.clf()

def plot_confusion_matrix(confusion_matrix, all_amino, output_directory):
    fig = plt.figure(1, figsize=(4, 4))
    plt.imshow(confusion_matrix, interpolation='none')
    plt.xticks(np.arange(len(all_amino)), all_amino)
    plt.yticks(np.arange(len(all_amino)), all_amino)
    cbar = plt.colorbar(boundaries = np.linspace(min(np.array(confusion_matrix).reshape(-1)),max(np.array(confusion_matrix).reshape(-1)),10))
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    cbar.set_label('Norm. prediction freq.')
    fig.savefig(output_directory+'Confusion_Matrix.png',dpi=150)
    #plt.show()                                                                                                                                                            
    plt.clf()
