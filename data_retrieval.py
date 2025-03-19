import json
import cv2
import numpy as np
from torch.utils.data import Dataset

prompt_path = "D:/Deep Learning Projects/fill50k-controlnet/fill50k/prompt.json"
source_dir = "D:/Deep Learning Projects/fill50k-controlnet/fill50k/fill50k/"
target_dir = "D:/Deep Learning Projects/fill50k-controlnet/fill50k/fill50k/"


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
        source = cv2.imread(source_dir+source_file)
        target = cv2.imread(target_dir+target_file)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32)/255.0
        target = (target.astype(np.float32)/127.5) - 1.0

        return dict(jpg = target, txt=prompt, hint = source) 