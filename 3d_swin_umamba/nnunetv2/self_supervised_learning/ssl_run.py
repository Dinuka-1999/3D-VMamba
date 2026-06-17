from nnunetv2.self_supervised_learning.SSL_trainer import MAE_Trainer
from nnunetv2.self_supervised_learning.SSL_preprocess import SSL_preprocessor

def preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True, help="This should be the full name. eg: Dataset100_Data")
    parser.add_argument('--num_processes', type=int, required=False, default=4, help="Number of processes to use for preprocessing."
        'More processes will speed up preprocessing but also increase RAM usage. If you run out of RAM, reduce this number.')
    args = parser.parse_args()
    preprocessor = SSL_preprocessor()
    preprocessor.run(args.dataset_id, args.num_processes)

def train_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True, help="This should be the full name. eg: Dataset100_Data")
    parser.add_argument('--num_epochs', type=int, required=False, default=100, help="Number of epochs to train for.")
    parser.add_argument('--num_threads', type=int, required=False, default=1, help="Number of threads to use for dataloader"
                        ". More threads will speed up training but also increase RAM usage. If you run out of RAM, reduce this number.")
    args = parser.parse_args()
    trainer = MAE_Trainer(args.dataset_id, args.num_threads)
    trainer.run_training(args.num_epochs)

if __name__ == "__main__":
    preprocess_entry()
    train_entry()