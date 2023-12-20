import os
os.environ['KERAS_BACKEND'] = 'tf'
import keras as k
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.datasets import cifar100
import densenet_without_bsc
import densenet_with_bsc
import sys
choice=int(sys.argv[1])


def model_exp(path="saved/exp/"): ###1
    #l1=open(path+"exp121_loss.txt","a")
    #a1=open(path+"exp121_acc.txt","a")
    (X_train,Y_train),(X_test,Y_test)=cifar100.load_data()
    model=densenet_with_bsc.DenseNet()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=256,
              shuffle=True,
              epochs=30,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

    # Evaluate the model
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    #l1.write(str(scores[0])+"\n")
    print('Accuracy: %.3f' % scores[1])
    #a1.write(str(scores[0])+"\n")
    print(model.summary())
    #model.save(path+"exp_bsc.h5")
    #l1.close()
    #a1.close()


def model_org(path="saved/org/"): ### 0
    #l1=open(path+"exp121_org_loss.txt","a")
    #a1=open(path+"exp121_org_acc.txt","a")
    (X_train,Y_train),(X_test,Y_test)=cifar100.load_data()
    model=densenet_without_bsc.DenseNet()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=256,
              shuffle=True,
              epochs=30,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

    # Evaluate the model
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    #l1.write(str(scores[0])+"\n")
    print('Accuracy: %.3f' % scores[1])
    #a1.write(str(scores[0])+"\n")
    print(model.summary())
    #model.save(path+"exp_org.h5")
    #l1.close()
    #a1.close()

if choice==1:
  model_exp()
if choice==0:
  model_org() 
