#import modules
import torch
import numpy as np
import os
import pandas as pd

class Tester():
    def __init__(self, test_loader, model, outputs_save_path, TEST_loss_save_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.outputs_save_path = outputs_save_path
        self.TEST_loss_save_path = TEST_loss_save_path

        self.outputs = {}

        self.batch_losses = []
        self.mean_batch_loss = 0.0

    def test(self):
        for i, items in enumerate(self.test_loader):
            imgs = items[0]
            labels = list(items[1])
            print(f"Testing batch {i+1} of {len(self.test_loader)}")
            self.batch_losses.append(self.test_batch(imgs, labels))
                    
        self.mean_batch_loss = np.sum(self.batch_losses) / len(self.test_loader)
        print(f"TESTING: Mean batch loss = {self.mean_batch_loss}")

        self.save_losses()
        

    def test_batch(self, imgs, labels):
            imgs = imgs.to(self.device)
            
            if self.model.flatten:
                batch_size = imgs.shape[0]
                imgs = imgs.view(batch_size, -1)
                output_data = self.model(imgs)
            else:
                output_data = self.model(imgs)
  
            for j, output_datum in enumerate(output_data):
                self.outputs[labels[j]] = output_datum

            batch_loss = self.model.loss_fn(output_data, imgs).item()

            return batch_loss


    def save_outputs(self):
        for key in self.outputs.keys():
            model_output = self.outputs[key]
            model_output = model_output.view(2064, -1).detach().numpy()

            save_path = os.path.join(self.outputs_save_path, f'from-{key}.npy')
            np.save(file=save_path, arr=model_output)

    def save_losses(self):
        
        df = pd.DataFrame(
            {
                "Mean Batch Loss": [self.mean_batch_loss],
                "Batch Losses" : self.batch_losses
            }
        )
        
        df.to_csv(path_or_buf=self.TEST_loss_save_path, index=False)
        




