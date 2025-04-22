import datetime
import os

class LossHistory():
    def __init__(self, log_dir, is_evolve=False):
        
        if is_evolve:
            self.log_dir = log_dir
        else:
            curr_time = datetime.datetime.now()
            time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
            self.time_str = time_str
            self.log_dir = os.path.join(log_dir, time_str)
            
        self.save_path = self.log_dir
        os.makedirs(self.save_path)

        self.losses = []
        self.val_loss = []

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_train_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

    def get_str(self):
        return self.time_str

    def write(self, sth):
        with open(os.path.join(self.log_dir, "log.txt"), 'a') as f:
            f.write(sth)
            f.write("\n")
    
