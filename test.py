from tensorflow.keras.utils import to_categorical
from numpy.random import randint

# generate categorical codes
cat_codes = randint(0, 10, 20)
# one hot encode
cat_codes = to_categorical(cat_codes, num_classes=10)

print(len(cat_codes))