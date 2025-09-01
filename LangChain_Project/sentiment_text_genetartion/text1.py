cm = confusion_matrix(df['target'], preds, normalize='true')
def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    plot_confusion_matrix(cm, ['negative', 'positive'])