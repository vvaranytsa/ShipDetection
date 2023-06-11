from model_funcs.func import *
from model_funcs.unet import *
from model_funcs.augmentation import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import os

# Defining roots
train = os.listdir(train_root)
test = os.listdir(test_root)

# Import file with coordinates
mask = pd.read_csv(os.path.join(base, 'csv/train_ship_segmentations_v2.csv'))

# Calculate ship counts
mask['ships'] = mask['EncodedPixels'].apply(lambda c_row: int(isinstance(c_row, str)))
unique_img_ids = mask.groupby('ImageId')['ships'].sum().reset_index()
unique_img_ids['has_ship'] = (unique_img_ids['ships'] > 0).astype(float)
mask.drop('ships', axis=1, inplace=True)

# Balancing the dataset
balanced_train_df = unique_img_ids.groupby('ships').apply(
    lambda x: x.sample(samples_group) if len(x) > samples_group else x)
balanced_train_df['ships'].plot.hist(bins=balanced_train_df['ships'].max() + 1)

# Splitting data into training and validation sets
train_ids, valid_ids = train_test_split(balanced_train_df, test_size=0.2, stratify=balanced_train_df['ships'])
train_df = mask.merge(train_ids, on='ImageId')
valid_df = mask.merge(valid_ids, on='ImageId')

# Generating image data generators
train_gen = make_image_gen(train_df)
train_x, train_y = next(train_gen)

valid_x, valid_y = next(make_image_gen(valid_df, valid_img))

# Data augmentation configuration
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)

# Creating the UNet model
train_model = unet()

# Configuring checkpoints and callbacks
weight_path = "model_folder/{}_weights.best.hdf5".format('train_model')

checkpoint = ModelCheckpoint(
    weight_path,
    monitor='val_loss',
    verbose=1,
    mode='min',
    save_weights_only=True
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15
)

callbacks_list = [checkpoint, early_stopping]


def fit():
    train_model.compile(optimizer='adam', loss=IoU_loss, metrics=[dice_coefficient])

    step_count = min(max_train_steps, train_df.shape[0] // batch)
    aug_gen = create_aug_gen(make_image_gen(train_df))
    loss_history = [train_model.fit(aug_gen,
                                  steps_per_epoch=step_count,
                                  epochs=max_train_epochs,
                                  validation_data=(valid_x, valid_y),
                                  callbacks=callbacks_list,
                                  workers=1
                                  )]
    return loss_history


# Fitting the model
loss_history = fit()

# Saving the model
final_model = train_model
final_model.save('model_folder/final_model.h5')
