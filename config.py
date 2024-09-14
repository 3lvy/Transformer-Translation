from pathlib import Path

# Training tip:
#   Adjust the batch size, num of epochs, learning rate (lr), and d_model to speed training and also improve model quality
#   Of course, depending on the hardware available
def get_config():
    return {
            "batch_size" : 4,
            "num_epochs" : 10,
            "lr" : 10**-4,
            "seq_len": 350,
            "d_model": 128,
            "datasource": 'Helsinki-NLP/opus_books',
            "lang_src": "en",
            "lang_tgt": "it",
            "model_folder" : "weights",
            "model_basename" : "tmodel_",
            "preload" : None,
            "tokenizer_file" : "tokenizer_{0}.json",
            "experiment_name" : "runs/tmodel"
        }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])