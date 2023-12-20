from collections import OrderedDict
from typing import Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor



def fn_binary_search_connections(start,end,list_keys):  ### 384- 20/12     512- 22/14   640- 23.9/15 
  if end-start>0:
    mid=int((start+end)/2)
    if start!=end:
      if mid-start>2:
        list_keys[mid].append(start+1)
    if end-mid>2:
      list_keys[end].append(mid+1)
    fn_binary_search_connections(start+1,mid-1,list_keys)
    fn_binary_search_connections(mid+1,end-1,list_keys)


def gen_list(gen_num_dense=12):
  list_keys=[]
  for i in range(gen_num_dense): # note: [[]]*gen_num_dense will not work correctly 
    list_keys.append([])
  return list_keys


class _DenseLayer(nn.Module):
    def __init__(
        self, block_number: int, layer_number: int, process_node: bool, initiai_feature_size: int, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False, binary_search_connections: bool = False,
    ) -> None:
        super().__init__()
        self.block_number = block_number
        self.layer_number = layer_number
        self.process_node = process_node
        self.binary_search_connections = binary_search_connections
        self.initial_feature_size = initiai_feature_size
        self.num_input_features = num_input_features
        if self.process_node:
            self.norm1 = nn.BatchNorm2d(self.initial_feature_size)
            self.conv1 = nn.Conv2d(self.initial_feature_size, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        else:
            self.norm1 = nn.BatchNorm2d(self.num_input_features)
            self.conv1 = nn.Conv2d(self.num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        norm_features = self.norm1(concated_features)
        bottleneck_output = self.conv1(self.relu1(norm_features))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        block_number: int,
        num_layers: int,
        initial_feature_size_list: list,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
        binary_search_connections: bool = False,
    ) -> None:
        super().__init__()
        self.binary_search_connections = binary_search_connections
        self.block_number = block_number
        self.num_layers = num_layers
        process_node = False
        if self.binary_search_connections:
            self.list_keys = gen_list(self.num_layers)
            fn_binary_search_connections(0,self.num_layers-1,self.list_keys)
        concat_index = -1
        for i in range(num_layers):
            if self.binary_search_connections:
                if self.list_keys[i]!=[]:
                    process_node = True
                    concat_index += 1
                    initial_feature_size = initial_feature_size_list[concat_index]
                elif concat_index >=0:
                    process_node = True
                    concat_index += 1
                    initial_feature_size = initial_feature_size_list[concat_index]
                else:
                    process_node = False
                    initial_feature_size = -1
            else:
                process_node = False
                initial_feature_size = -1
            layer = _DenseLayer(
                block_number = block_number,
                layer_number = i,
                process_node = process_node,
                initiai_feature_size= initial_feature_size,
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                binary_search_connections=self.binary_search_connections
            )
            self.add_module("denselayer%d" % (i + 1), layer)


    def forward(self, init_features: Tensor) -> Tensor: # 256, 512, 1024, 1024
        features = [init_features]
        stored_features = [init_features]
        if self.binary_search_connections:
            counter = -1 # for destination
            for name, layer in self.items():
                counter = counter + 1
                new_features = layer(features)
                features.append(new_features)
                stored_features.append(new_features) # to maintain indexed storage of outputs
                if counter < self.num_layers-1 and self.list_keys[counter+1]!=[]: # excluding last output
                    merge_idx = self.list_keys[counter+1][0]
                    merge_idx = merge_idx + 1 # we have input at index 0; outputs are stored from index 1.  
                    features.append(stored_features[merge_idx])
                    # print("combining in: ", dest_idx, " index: ",merge_idx, "for last index: ", self.num_layers-1, len(features))
        else:
            for name, layer in self.items():
                new_features = layer(features)
                features.append(new_features)
        combined_tensor = torch.cat(features, 1)
        return combined_tensor


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
        binary_search_connections = False,
    ) -> None:
        super().__init__()
        map_block_norm = [288, 576, 1216]
        map_initial_feature_size_list = [[256],
                                         [320,352,384,416,448,480,544],
                                         [448,480,512,544,576,640,704,736,768,800,832,864,928,960,992,1024,1056,1120,1184,2146],
                                         [736,800,832,864,896,960,992,1024,1088,1152]]
        self.binary_search_connections = binary_search_connections
        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block_number = i
            initial_feature_size_list = map_initial_feature_size_list[block_number]
            block = _DenseBlock(
                block_number=block_number,
                num_layers=num_layers,
                initial_feature_size_list = initial_feature_size_list,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                binary_search_connections = self.binary_search_connections,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                if self.binary_search_connections:
                    transition_channel = map_block_norm[i]
                else:
                    transition_channel = num_features
                trans = _Transition(num_input_features=transition_channel, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        if self.binary_search_connections:
            num_features = 1184
        self.final_norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        features = self.final_norm(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def initialise_model(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    binary_search_connections: bool,
    **kwargs: Any,
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features,binary_search_connections=binary_search_connections, **kwargs)
    print("Total parameters in Densenet 121 when Binary Search Connections is set "+str(binary_search_connections)+": ",get_n_params(model))
    return model

def get_BSC_Densenet_121_model(num_class):
    BSC_DenseNet_model = initialise_model(32, (6, 12, 24, 16), 64, num_classes=num_class, binary_search_connections=True)
    return BSC_DenseNet_model

def get_Densenet_121_model(num_class):
    DenseNet_model = initialise_model(32, (6, 12, 24, 16), 64, num_classes=num_class, binary_search_connections=False)
    return DenseNet_model

def get_densenet_models(num_class):
    # note: Here we are comparing only a single dense block with 7 layers. To establish the effectiveness of BSC-Densenet
    # Inorder to use Densenet 121, call: 
    DenseNet = get_Densenet_121_model(num_class)
    BSC_DenseNet = get_BSC_Densenet_121_model(num_class)
    return DenseNet, BSC_DenseNet