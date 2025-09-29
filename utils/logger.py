import os


class DefaultLogger:
    def __init__(self, exp_name, log_dir):
        self.exp_name = exp_name
        self.file = open(os.path.join(log_dir, exp_name), 'w')

    def open_log_file(self, exp_name, log_dir):
        self.file = open(os.path.join(log_dir, exp_name), 'w')

    def close_log_file(self):
        self.file.close()

    def write_round(self, round):
        self.file.write("round :"+str(round)+'\n')

    def write_test_acc(self, acc):
        self.file.write("global_test_acc :"+format(acc, '.4f')+'\n')

    def write_test_loss(self, loss):
        self.file.write("test_loss :"+format(loss, '.4f')+'\n')

    def write_mean_val_acc(self, mean_val_acc):
        self.file.write("client_mean_val_acc :"+format(mean_val_acc, '.4f')+'\n')

    def write_std_val_loss(self, std_val_loss):
        self.file.write("client_std_val_loss :"+format(std_val_loss, '.4f')+'\n')

    def write_mean_val_loss(self, mean_val_loss):
        self.file.write("client_mean_val_loss :"+format(mean_val_loss, '.4f')+'\n')

    def write_std_val_acc(self, std_val_acc):
        self.file.write("client_std_val_acc :"+format(std_val_acc, '.4f')+'\n')

    def write_global_val_loss(self, loss):
        self.file.write("global_val_loss :"+format(loss, '.4f')+'\n')
        
    def write_global_val_acc(self, acc):
        self.file.write("global_val_acc :"+format(acc, '.4f')+'\n')
        
    def write_global_test_loss(self, loss):
        self.file.write("global_features_test_loss :"+format(loss, '.4f')+'\n')
        
    def write_global_test_acc(self, acc):
        self.file.write("global_features_test_acc :"+format(acc, '.4f')+'\n')
        
    def write_aa(self, aa):
        self.file.write("AA :"+format(aa, '.4f')+'\n')
        
    def write_af(self, af):
        self.file.write("AF :"+format(af, '.4f')+'\n')
