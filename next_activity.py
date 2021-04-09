import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics 

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

parser = argparse.ArgumentParser(description="Process Transformer - Next Activity Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--task", type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY,  help="task name")

parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=12, type=int, help="batch size")

parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=0, type=int, 
                    help="gpu id")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if __name__ == "__main__":
    # Create directories to save the results and models
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/next_activity_ckpt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

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

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor="sparse_categorical_accuracy",
        mode="max", save_best_only=True)


    transformer_model.fit(train_token_x, train_token_y, 
        epochs=args.epochs, batch_size=args.batch_size, 
        shuffle=True, verbose=2, callbacks=[model_checkpoint_callback])

    # Evaluate over all the prefixes (k) and save the results
    k, accuracies,fscores, precisions, recalls = [],[],[],[],[]
    for i in range(max_case_length):
        test_data_subset = test_df[test_df["k"]==i]
        if len(test_data_subset) > 0:
            test_token_x, test_token_y = data_loader.prepare_data_next_activity(test_data_subset, 
                x_word_dict, y_word_dict, max_case_length)   
            y_pred = np.argmax(transformer_model.predict(test_token_x), axis=1)
            accuracy = metrics.accuracy_score(test_token_y, y_pred)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                test_token_y, y_pred, average="weighted")
            k.append(i)
            accuracies.append(accuracy)
            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)

    k.append(i + 1)
    accuracies.append(np.mean(accuracy))
    fscores.append(np.mean(fscores))
    precisions.append(np.mean(precisions))
    recalls.append(np.mean(recalls))
    print('Average accuracy across all prefixes:', np.mean(accuracies))
    print('Average f-score across all prefixes:', np.mean(fscores))
    print('Average precision across all prefixes:', np.mean(precisions))
    print('Average recall across all prefixes:', np.mean(recalls))    
    results_df = pd.DataFrame({"k":k, "accuracy":accuracies, "fscore": fscores, 
        "precision":precisions, "recall":recalls})
    results_df.to_csv(result_path+"_next_activity.csv", index=False)