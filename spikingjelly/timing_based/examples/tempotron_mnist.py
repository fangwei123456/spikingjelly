import argparse

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .. import encoding, neuron


parser = argparse.ArgumentParser(description="spikingjelly Tempotron MNIST Training")

parser.add_argument(
    "--device",
    default="cuda:0",
    help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"',
)

parser.add_argument(
    "--dataset-dir",
    default="./",
    help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"',
)
parser.add_argument(
    "--log-dir",
    default="./",
    help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"',
)
parser.add_argument(
    "--model-output-dir",
    default="./",
    help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"',
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    help='Batch 大小，例如“64”\n Batch size, e.g., "64"',
)
parser.add_argument(
    "--duration",
    default=100.0,
    type=float,
    help="连续观察窗终点 / Continuous observation-window endpoint",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-3,
    type=float,
    metavar="LR",
    help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ',
    dest="lr",
)
parser.add_argument(
    "-N",
    "--epoch",
    default=40,
    type=int,
    help="最大训练轮数，默认40；完整复现可显式设100 / Maximum epochs, default 40; use 100 explicitly for a full reproduction",
)
parser.add_argument(
    "-m",
    default=16,
    type=int,
    help='使用高斯调谐曲线编码每个像素点使用的神经元数量，例如“16”\n input neuron number for encoding a piexl in GaussianTuning encoder, e.g., "16"',
)


def _classification_loss(
    v_max: torch.Tensor,
    t_spike: torch.Tensor,
    label: torch.Tensor,
    v_threshold: float,
) -> torch.Tensor:
    target = F.one_hot(label, v_max.shape[1]).bool()
    wrong = target != torch.isfinite(t_spike)
    error = torch.where(target, v_threshold - v_max, v_max - v_threshold)
    firing_error = (wrong * error).sum() / (label.numel() * v_threshold)
    # Calibrate logits to threshold units instead of driving very large peaks.
    return F.cross_entropy(5 * v_max / v_threshold, label) + firing_error


def main() -> None:
    r"""
    **API Language** - :ref:`中文 <timing-mnist-main-cn>` | :ref:`English <timing-mnist-main-en>`

    ----

    .. _timing-mnist-main-cn:

    * **中文**

    根据命令行参数下载 MNIST，用连续高斯事件编码和单层 Tempotron 训练与评估。
    示例使用尺度校准交叉熵加发放误差线性罚项，预测仍取峰值 argmax。
    logits 为 5*v_max/v_threshold；温度对应阈值的 0.2，减少分类置信度对
    过大电压的需求。正确类别应发放、其他类别应沉默；仅对错误发放状态惩罚
    峰值与阈值的线性偏差，并按阈值归一化。抑制错误提前发放可以恢复后续证据。
    loss 在示例中定义，神经元内部不增加替代梯度或恢复规则。
    默认最多训练40轮；完整100轮复现需显式指定 ``--epoch 100``。本示例不按测试集早停。
    记录 TensorBoard 指标，打印准确率并保存模型。时间单位由 duration 与神经元默认时间常数共同约定。
    CPU 使用 PyTorch；安装 Triton 时 CUDA 自动选择融合算子，仍有块级同步。

    ----

    .. _timing-mnist-main-en:

    * **English**

    Download MNIST and train/evaluate a single Tempotron with continuous Gaussian
    events using command-line options. Prediction uses peak argmax. The external
    loss combines calibrated cross-entropy with a linear firing-error penalty.
    Logits are 5*v_max/v_threshold: temperature is 0.2 of the threshold, reducing
    the voltage amplitude needed for classification confidence. The correct class
    should fire and other classes remain silent; only misfiring outputs contribute
    their signed peak-to-threshold deviation, normalized by threshold. Suppressing
    false premature spikes can recover late evidence. No surrogate or recovery rule
    is added inside the neuron. Training defaults to at most 40 epochs; explicitly
    pass ``--epoch 100`` for a full reproduction. Test data does not drive early
    stopping. The example writes TensorBoard metrics, prints
    accuracy and saves the model. Duration and default neuron time constants share time units.
    CPU uses PyTorch; CUDA automatically selects fused operators when Triton is
    installed and still has block-level synchronization.
    """

    args = parser.parse_args()
    print("########## Configurations ##########")
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))
    print("####################################")

    device = args.device

    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir

    batch_size = args.batch_size
    duration = args.duration
    learning_rate = args.lr
    train_epoch = args.epoch
    m = args.m

    # 每个像素点用m个神经元来编码
    encoder = encoding.GaussianTuning(
        n=1,
        m=m,
        x_min=torch.zeros(size=[1]).to(device),
        x_max=torch.ones(size=[1]).to(device),
    ).to(device)

    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_data_loader = data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 初始化网络
    net = neuron.Tempotron(28 * 28 * m, 10, duration).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    train_times = 0
    max_test_accuracy = 0

    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()
        for img, label in tqdm(train_data_loader):
            img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]
            events = encoder(img.to(device), duration)

            optimizer.zero_grad()

            v_max, _, t_spike = net(*events)
            loss = _classification_loss(
                v_max, t_spike, label.to(device), net.v_threshold
            )
            loss.backward()
            optimizer.step()

            train_correct_sum += (
                (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
            )
            train_sum += label.numel()

            train_batch_acc = (
                (v_max.argmax(dim=1) == label.to(device)).float().mean().item()
            )
            writer.add_scalar("train_batch_acc", train_batch_acc, train_times)

            train_times += 1
        train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with torch.no_grad():
            correct_num = 0
            img_num = 0
            for img, label in tqdm(test_data_loader):
                img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]

                events = encoder(img.to(device), duration)
                v_max, _, _ = net(*events)
                correct_num += (
                    (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
                )
                img_num += img.shape[0]
            test_accuracy = correct_num / img_num
            writer.add_scalar("test_accuracy", test_accuracy, epoch)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print(
            "Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(
                epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times
            )
        )
        print()

    # 保存模型
    torch.save(net, model_output_dir + "/tempotron_snn_mnist.ckpt")
    writer.close()


if __name__ == "__main__":
    main()
