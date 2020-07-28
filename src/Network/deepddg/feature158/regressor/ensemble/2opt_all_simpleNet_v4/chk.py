from keras import Input, models, layers, optimizers, callbacks,regularizers, initializers

fold_num=1
modelpth = '/dl/sry/projects/from_hp/mCNN/src/Network/deepddg/opt_all_simpleNet_v4/model/140_4_32_32_32_64_0.5-2020.06.05.06.18.54/fold_%s_model.json'%fold_num
weightspth = '/dl/sry/projects/from_hp/mCNN/src/Network/deepddg/opt_all_simpleNet_v4/model/140_4_32_32_32_64_0.5-2020.06.05.06.18.54/fold_%s_weights-best.h5'%fold_num

#
# Load model
#
with open(modelpth, 'r') as json_file:
    loaded_model_json = json_file.read()
pre_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
pre_model.load_weights(filepath=weightspth)
## frozen layers before reduce
idx = 0
pre_model.summary()
for layer in pre_model.layers:
    print(idx, layer.name,layer.trainable)
    idx+=1
    # if idx <= 5:
    #     layer.trainable = False
