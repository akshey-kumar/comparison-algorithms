import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from sklearn.metrics import mean_squared_error
from functions import Database, preprocess_data, prep_data, timeseries_train_test_split, plot_latent_timeseries

# best hyperparameters found from tuning were:
config = {
    'lr': 0.0005096067531892613 ,
    'epochs': 100, #int(1614.8131428890506),
    'batch_size': int(52.136172155693295),
    'win': int(5.341348373587461),
    'hidden_dim': int(167.28320332656327),
}

class RnnAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RnnAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder GRU
        self.encoder_rnn = layers.GRU(hidden_dim, return_sequences=False, return_state=True)
        # projection to/from latent
        self.to_embedding = layers.Dense(latent_dim)
        self.from_embedding = layers.Dense(hidden_dim)
        # Generator GRU (decoder)
        self.decoder_rnn = layers.GRU(hidden_dim, return_sequences=True, return_state=True)
        # Output projection
        self.readout = layers.Dense(input_dim)



    def encode(self, x):
        _, h_enc = self.encoder_rnn(x)
        embedding = self.to_embedding(h_enc)
        return embedding

    def decode(self, embedding, seq_shape):
        h0 = self.from_embedding(embedding)
        h0 = tf.expand_dims(h0, axis=0)
        dummy_input = tf.zeros(seq_shape)  # shape: (batch, time, input_dim)
        decoded_seq, _ = self.decoder_rnn(dummy_input, initial_state=tf.squeeze(h0, axis=0))
        #decoded_seq, _ = self.decoder_rnn(dummy_input, initial_state=h0)

        return self.readout(decoded_seq)

    def call(self, x):
        embedding = self.encode(x)
        output = self.decode(embedding, tf.shape(x))
        return output


# Training and evaluation
for worm_num in range(5):
    print(worm_num)
    algorithm = 'rnn_autoencoder'
    b_neurons = [
        'AVAR',
        'AVAL',
        'SMDVR',
        'SMDVL',
        'SMDDR',
        'SMDDL',
        'RIBR',
        'RIBL'
    ]
    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_set_no=worm_num)
    data.exclude_neurons(b_neurons)
    x = data.neuron_traces.T
    b = data.states

    # prepare data (This autoencoder predicts the difference between present and future state)
    _, x = preprocess_data(x, float(data.fps))
    x_, b_ = prep_data(x, b, win=config['win'])

    # train test split
    x_train, x_test, b_train, b_test = timeseries_train_test_split(x_, b_)
    x0_tr = x_train[:, 0, :, :]
    x0_tst = x_test[:, 0, :, :]
    x1_tr = x_train[:, 1, :, :]
    x1_tst = x_test[:, 1, :, :]

    # Instantiate model
    model = RnnAutoencoder(
        input_dim=x0_tr.shape[-1],
        hidden_dim=config["hidden_dim"],
        latent_dim=3,
    )
    model.build(input_shape=x0_tr.shape)#(None, x0_tr.shape[1], x0_tr.shape[-1]))  # (batch, timesteps, features)
    model.compile(optimizer=optimizers.Adam(learning_rate=config["lr"]), loss='mse')

    # Train
    model.fit(x0_tr, x0_tr, batch_size=config["batch_size"], epochs=config["epochs"], verbose=True)

    # Evaluate
    reconstructed = model.predict(x0_tr)
    mse = mean_squared_error(x0_tr.reshape(len(x0_tr), -1), reconstructed.reshape(len(reconstructed), -1))
    print('Reconstructed shape:', reconstructed.shape)
    print('MSE:', round(mse, 8))

    # Latent projections
    y0_tr = model.encode(x0_tr)
    y0_tst = model.encode(x0_tst)
    y1_tr = model.encode(x1_tr)
    y1_tst = model.encode(x1_tst)

    # Convert to numpy
    y0_tr = y0_tr.numpy()
    y0_tst = y0_tst.numpy()
    y1_tr = y1_tr.numpy()
    y1_tst = y1_tst.numpy()


    plot_latent_timeseries(y0_tr, b_train, state_names=np.unique(b_train))

    # saving
    # model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/Y0_tr__' + algorithm + '_worm_' + str(worm_num), y0_tr)
    np.savetxt('data/generated/saved_Y/Y1_tr__' + algorithm + '_worm_' + str(worm_num), y1_tr)
    np.savetxt('data/generated/saved_Y/Y0_tst__' + algorithm + '_worm_' + str(worm_num), y0_tst)
    np.savetxt('data/generated/saved_Y/Y1_tst__' + algorithm + '_worm_' + str(worm_num), y1_tst)
    np.savetxt('data/generated/saved_Y/B_train_1__' + algorithm + '_worm_' + str(worm_num), b_train)
    np.savetxt('data/generated/saved_Y/B_test_1__' + algorithm + '_worm_' + str(worm_num), b_test)
