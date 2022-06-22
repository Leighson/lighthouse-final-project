from model import baseline_model, final_model
from dataset import get_data
from utils import save_plots
from config import LR, EPOCHS

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# save model at given checkpoints
model_checkpoint = ModelCheckpoint(
    filepath='../outputs/model',
    monitor='val_loss',
    mode='auto',
    save_best_only=True)

# load training and validation data
Xtrain, Xtest, ytrain, ytest = get_data()

# build and compile the model
model = baseline_model()
print(model.summary())

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=MeanSquaredError())

# train model
history = model.fit(
    (Xtrain, ytrain),
    validation_data=(Xtest, ytest),
    epochs=EPOCHS,
    callbacks=[model_checkpoint],
    workers=4, 
    use_multiprocessing=True)

# plot and save history
save_plots(history)