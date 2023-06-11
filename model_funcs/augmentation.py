from keras.preprocessing.image import ImageDataGenerator

dg_args = dict(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.5, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    data_format='channels_last'
)

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen):
    for in_x, in_y in in_gen:
        seed = 42
        g_x = image_gen.flow(in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x), next(g_y)
