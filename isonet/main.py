from isonet import *
from helper import *
from tensorflow import convert_to_tensor,transpose

if __name__ == "__main__":
    ground = read_from_mat('ground.mat')['ground'].astype(np.float32)
    observed = read_from_mat('stack.mat')['stack'].astype(np.float32)
    psf = read_from_mat('psf.mat')['psf']

    ctt = lambda x : convert_to_tensor(x)

    sx,sy,sz = psf.shape

    d = generate_training_samples(observed , ground ,  psf)

    x_train , y_train = convert_to_tensor(d['x']) , convert_to_tensor(d['y'])
    #print(x_train.shape)
    #print(y_train.shape)

    print('Training...')
    model = isonet1((sx,sy,1))

    x_test,y_test = map(ctt , observed) , map(ctt , ground)

    model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=5e-3),metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=1,
              epochs=1000,
              verbose=1)

    print('Testing...')

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:'  , score[1])
