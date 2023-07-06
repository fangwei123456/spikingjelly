import torch
import torch.nn as nn
import numpy as np
from ...activation_based import functional, layer, surrogate, neuron
import argparse


### Components ###
class ScaleLayer(nn.Module):
   def __init__(self):
       super().__init__()
       self.scale = torch.tensor(0.)

   def forward(self, input):
       return input * self.scale

class Neuronal_Cell(nn.Module):
    def __init__(self,args,  in_channel, out_channel, con_mat):
        """
        :param args: additional arguments
        :param in_channel: input channel
        :type in_channel: int
        :param out_channel: output channel
        :type out_channel: int
        :param con_mat: connection matrix
        :type con_mat: torch.Tensor

        Neuronal forward cell.

        """        
        super(Neuronal_Cell, self).__init__()
        self.cell_architecture = nn.ModuleList([])
        self.con_mat = con_mat
        for col in range(1,4):
            for row in range(col):
                op = con_mat[row,col]
                if op==0:
                    self.cell_architecture.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                                  stride=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture.append(nn.Sequential(
                                neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                               surrogate_function=surrogate.ATan(),
                                               detach_reset=True),
                                nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1,1), bias=False),
                                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                               affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))


    def forward(self, x_in):
        x_1 = self.cell_architecture[0](x_in)
        x_2 = self.cell_architecture[1](x_in) + self.cell_architecture[2](x_1)
        x_3 = self.cell_architecture[3](x_in) + self.cell_architecture[4](x_1) + self.cell_architecture[5](x_2)

        return x_3

class Neuronal_Cell_backward(nn.Module):
    def __init__(self,args,  in_channel, out_channel, con_mat):
        """
        :param args: additional arguments
        :param in_channel: input channel
        :type in_channel: int
        :param out_channel: output channel
        :type out_channel: int
        :param con_mat: connection matrix
        :type con_mat: torch.Tensor

        Neuronal backward cell.

        """     
        super(Neuronal_Cell_backward, self).__init__()

        self.cell_architecture = nn.ModuleList([])
        self.con_mat = con_mat
        self.cell_architecture_back = nn.ModuleList([])

        self.last_xin = 0.
        self.last_x1 = 0.
        self.last_x2 = 0.

        for col in range(1,4):
            for row in range(col):
                op = con_mat[row,col]
                if op==0:
                    self.cell_architecture.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                                  stride=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                    # l_idx +=1
                elif op == 3:
                    self.cell_architecture.append(nn.Sequential(
                                neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                               surrogate_function=surrogate.ATan(),
                                               detach_reset=True),
                                nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1,1), bias=False),
                                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                               affine=True, track_running_stats=True)))

                elif op == 4:
                    self.cell_architecture.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

        for col in range(0, 3):
            for row in range(col+1, 4):
                op = con_mat[row, col]
                if op == 0:
                    self.cell_architecture_back.append(ScaleLayer())
                elif op == 1:
                    self.cell_architecture_back.append(nn.Identity())
                elif op == 2:
                    self.cell_architecture_back.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                                  stride=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                elif op == 3:
                    self.cell_architecture_back.append(nn.Sequential(
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True),
                        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                                  stride=(1, 1), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1,
                                       affine=True, track_running_stats=True)))
                elif op == 4:
                    self.cell_architecture_back.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))

    def forward(self, x_in):
        x_1 = self.cell_architecture[0](x_in + self.last_xin)
        x_2 = self.cell_architecture[1](x_in+ self.last_xin ) + self.cell_architecture[2](x_1 + self.last_x1)
        x_3 = self.cell_architecture[3](x_in+ self.last_xin) + self.cell_architecture[4](x_1+ self.last_x1) + self.cell_architecture[5](x_2+ self.last_x2)

        self.last_xin = self.cell_architecture_back[0](x_1+ self.last_x1)+ self.cell_architecture_back[1](x_2+ self.last_x2)+ self.cell_architecture_back[2](x_3)
        self.last_x1 = self.cell_architecture_back[3](x_2+ self.last_x2)+ self.cell_architecture_back[4](x_3)
        self.last_x2 =  self.cell_architecture_back[5](x_3)

        return x_3

### Model ###
class SNASNet(nn.Module):
    def __init__(self, args, con_mat):
        """
        :param args: additional arguments
        :param con_mat: connection matrix
        :type con_mat: torch.Tensor

        The SNASNet `Neural Architecture Search for Spiking Neural Networks <https://arxiv.org/abs/2201.10355>`_ implementation by Spikingjelly.

        """     
        super(SNASNet, self).__init__()

        self.con_mat = con_mat
        self.total_timestep = args.timestep
        self.second_avgpooling = args.second_avgpooling

        if args.dataset == 'cifar10':
            self.num_class = 10
            self.num_final_neuron = 100
            self.num_cluster = 10
            self.in_channel = 3
            self.img_size = 32
            self.first_out_channel = 128
            self.channel_ratio = 2
            self.spatial_decay = 2 *self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif args.dataset == 'cifar100':
            self.num_class = 100
            self.num_final_neuron = 500
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 32
            self.channel_ratio = 1
            self.first_out_channel = 128
            self.spatial_decay = 2 *self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif args.dataset == 'tinyimagenet':
            self.num_class = 200
            self.num_final_neuron = 1000
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 64
            self.first_out_channel = 128
            self.channel_ratio = 1
            self.spatial_decay = 4 * self.second_avgpooling
            self.classifier_inter_ch = 4096
            self.stem_stride = 2

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channel, self.first_out_channel*self.channel_ratio, kernel_size=3, stride=self.stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channel*self.channel_ratio, affine=True),
        )

        if args.celltype == "forward":
            self.cell1 = Neuronal_Cell(args, self.first_out_channel*self.channel_ratio, self.first_out_channel*self.channel_ratio, self.con_mat)
        elif args.celltype == "backward":
            self.cell1 = Neuronal_Cell_backward(args, self.first_out_channel*self.channel_ratio, self.first_out_channel*self.channel_ratio, self.con_mat)
        else:
            print ("not implemented")
            exit()

        self.downconv1 = nn.Sequential(
            nn.BatchNorm2d(128*self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau= args.tau,
                                                      surrogate_function=surrogate.ATan(),
                                                      detach_reset=True),
                                        nn.Conv2d(128*self.channel_ratio, 256*self.channel_ratio, kernel_size=(3, 3),
                                                  stride=(1, 1), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(256*self.channel_ratio, eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True)
                                        )
        self.resdownsample1 = nn.AvgPool2d(2,2)

        if args.celltype == "forward":
            self.cell2 = Neuronal_Cell(args, 256*self.channel_ratio, 256*self.channel_ratio, self.con_mat)
        elif args.celltype == "backward":
            self.cell2 = Neuronal_Cell_backward(args, 256*self.channel_ratio, 256*self.channel_ratio, self.con_mat)
        else:
            print ("not implemented")
            exit()

        self.last_act = nn.Sequential(
                        nn.BatchNorm2d(256*self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True)
        )
        self.resdownsample2 = nn.AvgPool2d(self.second_avgpooling,self.second_avgpooling)

        self.classifier = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(256*self.channel_ratio*(self.img_size//self.spatial_decay)*(self.img_size//self.spatial_decay), self.classifier_inter_ch, bias=False),
            neuron.LIFNode(v_threshold=args.threshold, v_reset=0.0, tau=args.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
        nn.Linear(self.classifier_inter_ch, self.num_final_neuron, bias=True))
        self.boost = nn.AvgPool1d(self.num_cluster, self.num_cluster)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a =2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        self.neuron_init()

        acc_voltage = 0
        batch_size = input.size(0)
        static_x = self.stem(input)

        for t in range(self.total_timestep):
            x = self.cell1(static_x)
            x = self.downconv1(x)
            x = self.resdownsample1(x)
            x = self.cell2(x)
            x = self.last_act(x)
            x = self.resdownsample2(x)
            x = x.view(batch_size, -1)
            x = self.classifier(x)
            acc_voltage = acc_voltage + self.boost(x.unsqueeze(1)).squeeze(1)
        acc_voltage = acc_voltage / self.total_timestep
        return acc_voltage

    def neuron_init(self):
        self.cell1.last_xin =0.
        self.cell1.last_x1 =0.
        self.cell1.last_x2 =0.
        self.cell2.last_xin = 0.
        self.cell2.last_x1 = 0.
        self.cell2.last_x2 = 0.


if __name__ == "__main__":
    ### Example ###

    parser = argparse.ArgumentParser("SNASNet")
    parser.add_argument('--dataset', type=str, default='cifar100', help='[cifar10, cifar100]')
    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--tau', type=float, default=4/3, help='neuron decay time factor')
    parser.add_argument('--threshold', type=float, default=1.0, help='neuron firing threshold')
    parser.add_argument('--celltype', type=str, default='backward', help='[forward, backward]')
    parser.add_argument('--second_avgpooling', type=int, default=2, help='momentum')
    args = parser.parse_args()
    
    int_list = [[0, 0, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]
    best_neuroncell = torch.Tensor(int_list)

    print ('-'*7, "best_neuroncell",'-'*7)
    print (best_neuroncell)
    print('-' * 30)
    
    snasnet = SNASNet(args, best_neuroncell)