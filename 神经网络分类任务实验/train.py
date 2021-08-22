from netSet import *

data_path = 'cifar-10-batches-bin/train'
batch_size = 32
status = "train"
epoch = 1
learn_rate = 0.01
para = {
    'epoch': epoch,
    'lr': learn_rate
}

network = LeNet5(10)
train_data = get_data(data_path, batch_size=batch_size, status=status)
model,steps_loss,steps_eval = train_model(network, train_data, para)

test_path = './cifar-10-batches-bin/test'
batch_size = 1
status = "test"
test_data = get_data(datapath=test_path, batch_size=batch_size, status=status)
test_model(model,test_data)
pic(steps_loss,model, model_name='LeNet')
eval_show(steps_eval, model_name='LeNet')


batch_size = 32
status = "train"
epoch = 20
learn_rate = 0.001
para = {
    'epoch': epoch,
    'lr': learn_rate
}
network = my_LeNet5(10)
train_data = get_data(data_path, batch_size=batch_size, status=status)
model, steps_loss, steps_eval = train_model(network, train_data, para)

test_path = './cifar-10-batches-bin/test'
batch_size = 1
status = "test"
test_data = get_data(datapath=test_path, batch_size=batch_size, status=status)
test_model(model,test_data)
pic(steps_loss, model,model_name='my_LeNet')
eval_show(steps_eval, model_name='my_LeNet')
