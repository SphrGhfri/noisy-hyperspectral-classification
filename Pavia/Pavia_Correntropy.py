from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import time
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import  Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.applications.xception import Xception
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow as tf
import gc
import os

# Reset Keras Session
start = time.time()

def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del Xception,base_model,custom_Xception_model 
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))
    
if not os.path.exists('results'):
    os.makedirs('results')
            
for i in range(1,50) :
    
    print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%H:%M")))
    
    
    image_input = Input(shape=(224, 224, 3))
    
    base_model = Xception(input_tensor=image_input, include_top=False,weights='imagenet')
    
    
    
    last_layer = base_model.get_layer('block14_sepconv2_act').output
    x = Flatten(name='flatten')(last_layer) 
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(9 , activation='softmax', name='output')(x)
    custom_Xception_model = Model(image_input, out)
    
    # freeze all the layers except the dense layers
    for layer in custom_Xception_model.layers[:2]:
    	layer.trainable = False
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    
    
    
    train_generator = train_datagen.flow_from_directory(
        directory='Pavia_Split_data/train/noiseless',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,                                       
        seed=42
    )
    
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/noiseless',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

# -------------------------------------------------------------------------------------------------------------   
    valid_generator1 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/gaussian/noise5',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator2 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/gaussian/noise10',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator3 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/gaussian/noise15',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator4 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/gaussian/noise20',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator5 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/gaussian/noise25',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    
# -----------------------------------------------------------------------------------------   
    valid_generator6 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/saltPepper/noise5',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator7 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/saltPepper/noise10',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator8 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/saltPepper/noise15',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator9 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/saltPepper/noise20',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator10 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/saltPepper/noise25',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    
#  ---------------------------------------------------------------------------------------------  
    valid_generator11 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/stripe/noise5',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator12 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/stripe/noise10',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator13 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/stripe/noise15',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator14 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/stripe/noise20',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator15 = valid_datagen.flow_from_directory(
        directory='Pavia_Split_data/validation/stripe/noise25',
        target_size=(224, 224),
        batch_size=20,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
 
    
    def correntropy(y_true, y_pred,gamma=1):
        a=y_pred-y_true
        a=K.abs(a)
        return (1-K.exp(-(K.pow(a,2))))
    
    
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    
    
    print('\n')
    print('----------------------------------------------------------')
    print('\n')  
    
   
    custom_Xception_model.compile(loss= correntropy ,optimizer='adadelta',metrics=['accuracy'])
    
    
    hist = custom_Xception_model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=2)
    
    					
    print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%H:%M")))
    print("Noiseless")
    print("0")
    test_loss, test_acc =custom_Xception_model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID, verbose=1)
   
    
    print("Gaussian")
    print("5")
    test_loss1, test_acc1 =custom_Xception_model.evaluate_generator(generator=valid_generator1,steps=STEP_SIZE_VALID, verbose=1)
    print("10")
    test_loss2, test_acc2 =custom_Xception_model.evaluate_generator(generator=valid_generator2,steps=STEP_SIZE_VALID, verbose=1)
    print("15")
    test_loss3, test_acc3 =custom_Xception_model.evaluate_generator(generator=valid_generator3,steps=STEP_SIZE_VALID, verbose=1)
    print("20")
    test_loss4, test_acc4 =custom_Xception_model.evaluate_generator(generator=valid_generator4,steps=STEP_SIZE_VALID, verbose=1)
    print("25")
    test_loss5, test_acc5 =custom_Xception_model.evaluate_generator(generator=valid_generator5,steps=STEP_SIZE_VALID, verbose=1)
    
    
    print("Salt-Pepper")
    print("5")
    test_loss6, test_acc6 =custom_Xception_model.evaluate_generator(generator=valid_generator6,steps=STEP_SIZE_VALID, verbose=1)
    print("10")
    test_loss7, test_acc7 =custom_Xception_model.evaluate_generator(generator=valid_generator7,steps=STEP_SIZE_VALID, verbose=1)
    print("15")
    test_loss8, test_acc8 =custom_Xception_model.evaluate_generator(generator=valid_generator8,steps=STEP_SIZE_VALID, verbose=1)
    print("20")
    test_loss9, test_acc9 =custom_Xception_model.evaluate_generator(generator=valid_generator9,steps=STEP_SIZE_VALID, verbose=1)
    print("25")
    test_loss10, test_acc10 =custom_Xception_model.evaluate_generator(generator=valid_generator10,steps=STEP_SIZE_VALID, verbose=1)
    
    print("Strip")
    print("5")
    test_loss11, test_acc11 =custom_Xception_model.evaluate_generator(generator=valid_generator11,steps=STEP_SIZE_VALID, verbose=1)
    print("10")
    test_loss12, test_acc12 =custom_Xception_model.evaluate_generator(generator=valid_generator12,steps=STEP_SIZE_VALID, verbose=1)
    print("15")
    test_loss13, test_acc13 =custom_Xception_model.evaluate_generator(generator=valid_generator13,steps=STEP_SIZE_VALID, verbose=1)
    print("20")
    test_loss14, test_acc14 =custom_Xception_model.evaluate_generator(generator=valid_generator14,steps=STEP_SIZE_VALID, verbose=1)
    print("25")
    test_loss15, test_acc15 =custom_Xception_model.evaluate_generator(generator=valid_generator15,steps=STEP_SIZE_VALID, verbose=1)
    
    tmp1=[test_acc, test_loss]
    np.savetxt('results/Pavia_noiseless_Correntropy_'+repr(i)+'.csv', tmp1, delimiter=',')
    
    tmp2=[test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_loss1, test_loss2, test_loss3, test_loss4, test_loss5]
    np.savetxt('results/Pavia_Gaussian_Correntropy_'+repr(i)+'.csv', tmp2, delimiter=',')
    
    tmp3=[test_acc6, test_acc7, test_acc8, test_acc9, test_acc10, test_loss6, test_loss7, test_loss8, test_loss9, test_loss10]
    np.savetxt('results/Pavia_SaltPepper_Correntropy_'+repr(i)+'.csv', tmp3, delimiter=',')
    
    tmp4=[test_acc11, test_acc12, test_acc13, test_acc14, test_acc15, test_loss11, test_loss12, test_loss13, test_loss14, test_loss15]
    np.savetxt('results/Pavia_Strip_Correntropy_'+repr(i)+'.csv', tmp4, delimiter=',')
    
    print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%H:%M")))
    reset_keras()
    reset_keras()

