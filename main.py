from pathlib import Path
import os
from Model import *

dataset_path = str(Path(__file__).parents[1]) + '/Dataset/dataset_for_model.pickle'

if os.path.exists(dataset_path):
    with open(dataset_path, 'rb') as file:
        dataset = pickle.load(file, encoding='latin1')
else:
    script_path = str(Path(__file__).parents[1]) + '/Dataset.py'
    with open(script_path, 'r') as file:
        exec(file.read())

model = Model(
            CNN(input_dimension=(3, 32, 32),
                number_of_filters=16,
                size_of_filter=3,
                hidden_dimension=256,
                number_of_classes=5),
            dataset,
            optimization_config={'learning_rate':1e-3},
            learning_rate_decay=1.0,
            batch_size=32,
            number_of_epochs=30,
            verbose_mode=True)

model.fit()

model.evaluate()

model.save()
