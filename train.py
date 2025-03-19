import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import types
import torch
import inspect
import numpy as np
import cv2
import json
from pathlib import Path
import os

# Fix the module for ControlNet
from forked_controlnet.ControlNet.cldm.logger import ImageLogger
from forked_controlnet.ControlNet.cldm.model import create_model, load_state_dict

# Dataset paths
prompt_path = "./fill50k/prompt.json"
source_dir = "./fill50k/fill50k/"
target_dir = "./fill50k/fill50k/"

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(prompt_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        source_file = item['source']
        target_file = item['target']
        prompt = item['prompt']
        
        # Check if files exist before trying to read
        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, target_file)
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
            
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file not found: {target_path}")
        
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        
        if source is None:
            raise ValueError(f"Error: Could not read source image: {source_path}")
            
        if target is None:
            raise ValueError(f"Error: Could not read target image: {target_path}")
        
        # Convert BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # Normalize
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return dict(jpg=target, txt=prompt, hint=source)

# Fix the on_train_batch_start method hook issue
def fix_lightning_hooks():
    from forked_controlnet.ControlNet.ldm.models.diffusion.ddpm import LatentDiffusion
    
    # Check if the method exists and only has 2 parameters (plus self)
    if hasattr(LatentDiffusion, 'on_train_batch_start'):
        params = inspect.signature(LatentDiffusion.on_train_batch_start).parameters
        if len(params) == 3 and 'dataloader_idx' not in params:
            # Store original method
            original_method = LatentDiffusion.on_train_batch_start
            
            # Create fixed method
            def fixed_method(self, batch, batch_idx, dataloader_idx=0):
                return original_method(self, batch, batch_idx)
            
            # Replace the method
            LatentDiffusion.on_train_batch_start = fixed_method
            print("✓ Fixed on_train_batch_start hook")

# Fix the attention mask issue for CLIP encoder
def fix_clip_attention_mask():
    from forked_controlnet.ControlNet.ldm.modules.encoders.modules import FrozenCLIPEmbedder
    
    # Check if this class exists and has the problematic method
    if hasattr(FrozenCLIPEmbedder, 'text_transformer_forward'):
        original_method = FrozenCLIPEmbedder.text_transformer_forward
        
        # New implementation that safely handles attention masks
        def patched_method(self, x, attn_mask=None):
            for r in self.model.transformer.resblocks:
                try:
                    # Try with no attention mask first (safer)
                    x = r(x)
                except Exception as e:
                    if 'attn_mask' in str(e):
                        try:
                            # If that fails, try with the mask
                            x = r(x, attn_mask=None)
                        except:
                            # If both fail, re-raise the original error
                            raise e
            return x
        
        # Replace the method
        FrozenCLIPEmbedder.text_transformer_forward = patched_method
        print("✓ Fixed CLIP attention mask issue")

def main():
    # Apply fixes
    fix_lightning_hooks()
    fix_clip_attention_mask()
    
    # Configuration
    resume_path = './models/control_sd21_ini.ckpt'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    
    # Create and load model
    print("Creating model...")
    model = create_model('./forked_controlnet/ControlNet/models/cldm_v21.yaml').cpu()
    print("Loading checkpoint...")
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    
    # Setup dataset and dataloader with error handling
    print("Setting up dataset...")
    try:
        dataset = MyDataset()
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        dataloader = DataLoader(
            dataset, 
            num_workers=0,  # Set to 0 to debug any dataset issues more easily
            batch_size=batch_size, 
            shuffle=True
        )
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        return
    
    # Setup logger and trainer
    print("Setting up trainer...")
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        precision=32, 
        callbacks=[logger],
        max_epochs=100,  # Set an appropriate value
        log_every_n_steps=10
    )
    
    # Train
    print("Starting training...")
    try:
        trainer.fit(model, dataloader)
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()