
from torch.utils.data import DataLoader
from utils.npy_dataset import npyDataset

class DataHandler():
    def __init__(self, train_path, eval_path, test_path, batch_size):
        self.train_set = npyDataset(arrays_path=train_path)
        self.eval_set = npyDataset(arrays_path=eval_path)
        self.test_set = npyDataset(arrays_path=test_path)
        
        self.train_loader = DataLoader(dataset=self.train_set, shuffle=True, batch_size=batch_size)
        self.eval_loader = DataLoader(dataset=self.eval_set, shuffle=False, batch_size=batch_size)
        self.test_loader = DataLoader(dataset=self.test_set, shuffle=False, batch_size=batch_size)

    def get_sets(self):
        return self.train_set, self.eval_set, self.test_set
    
    def get_loaders(self):
        return self.train_loader, self.eval_loader, self.test_loader
    
# if __name__ == "__main__":
    
#     train_path = './data/processed/horizontal_edge_detector_sets/multiple_filters_method/untrimmed/training set'
#     eval_path = './data/processed/horizontal_edge_detector_sets/multiple_filters_method/untrimmed/validation set'
#     test_path = './data/processed/horizontal_edge_detector_sets/multiple_filters_method/untrimmed/test set' 

#     datahandler = DataHandler(
#             train_path=train_path,
#             eval_path=eval_path,
#             test_path=test_path,
#             batch_size=4
#         )
    
#     train_loader, eval_loader, test_loader = datahandler.get_loaders()

#     for i, _ in enumerate(train_loader):
#         print(i, _)