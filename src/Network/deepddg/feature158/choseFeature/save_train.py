from matplotlib import pyplot as plt
def loss_plot(history_dict, outpth):
    loss = history_dict['loss']
    mean_absolute_error = history_dict['mean_absolute_error']
    pearson_r = history_dict['pearson_r']
    rmse = history_dict['rmse']
    val_loss = history_dict['val_loss']
    val_mean_absolute_error = history_dict['val_mean_absolute_error']
    val_pearson_r = history_dict['val_pearson_r']
    val_rmse = history_dict['val_rmse']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss (mse)')
    plt.plot(epochs, val_loss, 'g', label='Validation loss (mse)')
    # plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, mean_absolute_error, 'r', label='Training mae')
    plt.plot(epochs, val_mean_absolute_error, 'g', label='Validation mae')
    # plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, rmse, 'r', label='Training rmse')
    plt.plot(epochs, val_rmse, 'g', label='Validation rmse')
    # plt.title('Training and validation rmse')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, pearson_r, 'r', label='Training pcc')
    plt.plot(epochs, val_pearson_r, 'g', label='Validation pcc')
    # plt.title('Training and validation pcc')
    plt.xlabel('Epochs')
    plt.ylabel('PCC')
    plt.grid(True)
    plt.legend(loc="lower right")

    # plt.show()
    plt.savefig(outpth)#pngfile
    plt.clf()

def save_train_cv(model, modeldir, history_dict, k_count):
    #
    # save model architecture
    #
    try:
        model_json = model.to_json()
        with open('%s/fold_%s_model.json' % (modeldir, k_count), 'w') as json_file:
            json_file.write(model_json)
    except:
        print('save model.json to json failed, fold_num: %s' % k_count)
    #
    # save model weights
    #
    try:
        model.save_weights(filepath='%s/fold_%s_weightsFinal.h5' % (modeldir, k_count))
    except:
        print('save final model weights failed, fold_num: %s' % k_count)
    #
    # save training history
    #
    try:
        with open('%s/fold_%s_history.dict' % (modeldir, k_count), 'w') as file:
            file.write(str(history_dict))
        # with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'r') as file:
        #     print(eval(file.read()))
    except:
        print('save history_dict failed, fold_num: %s' % k_count)
    #
    # save loss figure
    #
    try:
        figure_pth = '%s/fold_%s_lossFigure.png' % (modeldir, k_count)
        loss_plot(history_dict, outpth=figure_pth)
    except:
        print('save loss plot figure failed, fold_num: %s' % k_count)

def save_train(model,modeldir,history_dict):
    #
    # save model architecture
    #
    try:
        model_json = model.to_json()
        with open('%s/model.json' % modeldir, 'w') as json_file:
            json_file.write(model_json)
    except:
        print('save model.json to json failed')
    #
    # save model weights
    #
    try:
        model.save_weights(filepath='%s/weightsFinal.h5' % (modeldir))
    except:
        print('save final model weights failed')
    #
    # save training history
    #
    try:
        with open('%s/history.dict' % (modeldir), 'w') as file:
            file.write(str(history_dict))
        # with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'r') as file:
        #     print(eval(file.read()))
    except:
        print('save history_dict failed, fold_num: %s')
    #
    # save loss figure
    #
    try:
        figure_pth = '%s/lossFigure.png' % (modeldir)
        loss_plot(history_dict, outpth=figure_pth)
    except:
        print('save loss plot figure failed')

