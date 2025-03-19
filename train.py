import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_retrieval import MyDataset
from forked_controlnet.ControlNet.cldm.logger import ImageLogger
from forked_controlnet.ControlNet.cldm.model import create_model, load_state_dict


resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

model = create_model('./forked_controlnet/ControlNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)