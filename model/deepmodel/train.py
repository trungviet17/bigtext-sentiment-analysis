from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import wandb
from dotenv import load_dotenv
import os 
import torch 

load_dotenv()
WANDB_KEY = os.getenv('WANDB_KEY')


def train(model_name: str, data_module, model): 
    print("Training model...")

    wandb_logger = WandbLogger(
        project="Big Sentiment Analysis", name = model_name,         
        save_dir="model/deepmodel/logging",  
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",  
        patience=3,          
        mode="min",          
        verbose=True        
    )

    trainer = Trainer(
        max_epochs=15,
        logger=wandb_logger,        
        callbacks=[early_stopping], 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )

    trainer.fit(model, data_module)



def save_model_as_onnx(lightning_model, file_path="model.onnx", input_example=None):
    if input_example is None:
        input_example = torch.randint(0, lightning_model.model.embedding.num_embeddings, (1, 50))  # Batch size 1, sequence length 50

    lightning_model.eval()

    torch.onnx.export(
        lightning_model.model,                 
        input_example,                        
        file_path,                             
        export_params=True,                   
        opset_version=11,                     
        input_names=["input"],                
        output_names=["output"],               
        dynamic_axes={                        
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model saved to {file_path} in ONNX format.")



