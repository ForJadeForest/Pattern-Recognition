import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.common import dtype as mstype
import mindspore
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.train.callback import Callback
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
import os


def get_data(datapath, batch_size=32, status='train'):
    """
    获取数据
    :param datapath: 数据路径
    :param batch_size: batch大小
    :param status: train or test 对应不同的数据处理
    :return: 处理好的数据
    """
    data = ds.Cifar10Dataset(datapath)
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale = CV.Rescale(rescale, shift)  # 归一化与平移
    # 对于RGB三通道分别设定mean和std
    normalize = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # 通道变化
    channel_swap = CV.HWC2CHW()
    # 类型变化
    typecast = C.TypeCast(mstype.int32)
    data = data.map(input_columns="label", operations=typecast)
    if status == "train":
        random_horizontal = CV.RandomHorizontalFlip()  # 随机翻转
        random_crop = CV.RandomCrop([32, 32], [4, 4, 4, 4])  # 随机裁剪
        data = data.map(input_columns="image", operations=random_crop)
        data = data.map(input_columns="image", operations=random_horizontal)
    data = data.map(input_columns="image", operations=rescale)
    data = data.map(input_columns="image", operations=normalize)
    data = data.map(input_columns="image", operations=channel_swap)

    # shuffle
    data = data.shuffle(buffer_size=1000)
    # 切分数据集到batch_size
    data = data.batch(batch_size, drop_remainder=True)

    return data


def conv_with_initialize(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """
    返回初始化的卷积层
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核大小
    :param stride: stride大小
    :param padding: padding值
    :return: 初始化的卷积层
    """
    weight = TruncatedNormal(0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """
    返回初始化的全连接层
    :param input_channels: 输入个数
    :param out_channels: 输出个数
    :return: 初始化的全连接层
    """
    weight = TruncatedNormal(0.02)
    bias = TruncatedNormal(0.02)
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=3):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv_with_initialize(channel, 6, 5)
        self.conv2 = conv_with_initialize(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class my_LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=3):
        super(my_LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv_with_initialize(channel, 32, 3)
        self.conv2 = conv_with_initialize(32, 64, 3)
        self.conv3 = conv_with_initialize(64, 128, 3)
        self.fc1 = fc_with_initialize(512, 120)
        self.fc2 = fc_with_initialize(120, 80)
        self.fc3 = fc_with_initialize(80, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.max_pool2d(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch - 1) * 1875 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 125 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(acc["Accuracy"])


def pic(steps_loss, model,model_name):
    """
    绘制loss变化图和预测例子图
    :param steps_loss: loss的记录值
    :param model: 训练的模型
    :param model_name: 模型的名字
    :return: 无
    """
    steps = steps_loss["step"]
    loss_value = steps_loss["loss_value"]
    steps = list(map(int, steps))
    loss_value = list(map(float, loss_value))
    plt.plot(steps, loss_value, color="red")
    plt.xlabel("Steps")
    plt.ylabel("Loss_value")
    plt.title("Change chart of model loss value")
    plt.savefig('{}_loss.png'.format(model_name))
    plt.show()
    data_path = 'cifar-10-batches-bin/test'
    ds_test = get_data(datapath=data_path, batch_size=1, status='test').create_dict_iterator()
    for i, d in enumerate(zip(ds_test, ds.Cifar10Dataset(data_path).create_dict_iterator())):
        data = d[0]
        labels = data["label"].asnumpy()

        output = model.predict(Tensor(data['image']))
        pred = np.argmax(output.asnumpy(), axis=1)
        plt.subplot(4, 8, i + 1)
        color = 'blue' if pred == labels else 'red'
        plt.title("pre:{}".format(pred[0]), color=color, fontsize=10)
        pic = d[1]["image"].asnumpy()
        plt.imshow(pic)
        plt.axis("off")
        if i >= 31:
            break
    plt.savefig('{}_pre.png'.format(model_name))
    plt.show()


def train_model(network, data, para):
    """
    训练模型
    :param network: 网络架构
    :param data: 训练数据
    :param para: 训练参数，dict形式，包含learn rate 和 epoch
    :return: 训练好的模型与训练过程中loss和准确率的变化
    """
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target='CPU')
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = nn.Momentum(params=network.trainable_params(), learning_rate=para['lr'], momentum=0.9)
    time_cb = TimeMonitor(data_size=data.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=1562, keep_checkpoint_max=10)
    checkpoint = ModelCheckpoint(prefix="checkpoint_lenet_original", directory='./checkpoint', config=config_ck)
    model = Model(network=network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()})
    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}
    step_loss_acc_info = StepLossAccInfo(model, data, steps_loss, steps_eval)
    callback = [time_cb, checkpoint, LossMonitor(per_print_times=100), step_loss_acc_info]
    print("============== Starting Training ==============")
    model.train(epoch=para['epoch'], train_dataset=data, callbacks=callback, dataset_sink_mode=False)
    print("train acc: ", model.eval(data, dataset_sink_mode=False))
    return model,steps_loss,steps_eval


def test_model(model, data):
    """
    测试模型
    :param model:训练好的模型
    :param data: 测试数据
    :return: 无
    """
    res = model.eval(data, dataset_sink_mode=False)
    print("test acc", res)


def eval_show(steps_eval,model_name):
    """
    准确率变化图像绘制
    :param steps_eval:准确率的值
    :param model_name: 模型名字
    :return: 无
    """
    plt.xlabel("step number")
    plt.ylabel("Model accuracy")
    plt.title("Model accuracy variation chart")
    plt.plot(steps_eval["step"], steps_eval["acc"], "red")
    plt.savefig('{}_acc1.png'.format(model_name))
    plt.show()