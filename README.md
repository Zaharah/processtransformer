## Process Transformer

Transformer Neural Model for Business Process Monitoring Tasks 

#### Tasks
- Next Activity Prediction
- Time Prediction of Next Activity
- Remaining Time Prediction

#### Install 
```
pip install processtransformer
```

#### Usage
```
import argparse
import tensorflow as tf
from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

parser = argparse.ArgumentParser(description="Process Transformer - Next Activity Prediction.")
parser.add_argument("--dataset", required=True, type=str, help="dataset name")
parser.add_argument("--task", type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY,  help="task name")
parser.add_argument("--epochs", default=1, type=int, help="number of total epochs")
parser.add_argument("--batch_size", default=12, type=int, help="batch size")
parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate")

# Load data
data_loader = loader.LogsDataLoader(name = args.dataset)

(train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
    vocab_size, num_output) = data_loader.load_data(args.task)

# Prepare training examples for next activity prediction task
train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df, 
    x_word_dict, y_word_dict, max_case_length)

# Create and train a transformer model
transformer_model = transformer.get_next_activity_model(
    max_case_length=max_case_length, 
    vocab_size=vocab_size,
    output_dim=num_output)

transformer_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
transformer_model.fit(train_token_x, train_token_y, 
    epochs=args.epochs, batch_size=args.batch_size)
```

See complete code examples within the github repository for other tasks, including preparing raw process data for transformer model.

#### Tools
- <a href="http://tensorflow.org/">Tensorflow >=2.4</a>
