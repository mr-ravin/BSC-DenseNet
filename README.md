## Implementation of Binary Search Connections in DenseNet (i.e. BSC-DenseNet) using Pytorch
This repository includes the implementation of **BSC-DenseNet-121** from research paper `"Adding Binary Search Connections to Improve DenseNet Performance"`, published in **Elsevier-SSRN conference proceedings of NGCT 2019**. The base code of openly available DenseNet is also present in this repository for comparing our BSC-DenseNet on the CIFAR-100 dataset.

**Paper Title**: Adding Binary Search Connections to Improve DenseNet Performance

**Author**: [Ravin Kumar](https://mr-ravin.github.io/)

**Publication**: 27th February 2020

**Published Paper**: [click here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3545071)  <!-- Elsevier-SSRN conference proceedings of NGCT 2019 -->

**Doi**: [DOI Link of Paper](http://dx.doi.org/10.2139/ssrn.3545071)

**Other Sources**:
- [Research Gate](https://www.researchgate.net/publication/339673672_Adding_Binary_Search_Connections_to_Improve_DenseNet_Performance), [Research Gate - Preprint](https://www.researchgate.net/publication/382385286_Adding_binary_search_connections_to_improve_DenseNet_performance)
- [Osf.io](https://osf.io/preprints/osf/8z42s_v2)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3545071)
- [Internet Archive](https://archive.org/details/bsc-densenet)
- [Medium.com](https://medium.com/@ch.ravinkumar/boosting-densenet-model-with-binary-search-connections-a-smarter-way-to-improve-performance-5a009ea1b693)

**GitHub Repository**: [https://github.com/mr-ravin/BSC-DenseNet](https://github.com/mr-ravin/BSC-DenseNet)

🔍 **Note on BSC Connections**:
Binary Search Connections (BSC) introduced in this paper are implemented via **concatenation**, not addition. BSC DenseNet has extra connections within the densely connected block, preserving DenseNet’s feature aggregation mechanism and enhancing feature representation at each layer. 

##### Cite as:
```
Kumar, Ravin, Adding Binary Search Connections to Improve DenseNet Performance (February 27, 2020). 5th International Conference on Next Generation Computing Technologies (NGCT-2019). Available at SSRN: https://ssrn.com/abstract=3545071 or http://dx.doi.org/10.2139/ssrn.3545071 
```
---
### 🔍 Why DenseNet Needed an Upgrade?
DenseNet’s brilliance lies in its fully connected architecture, where each layer is linked to every preceding layer within a block. This design enhances gradient flow, allows better feature reuse, and requires fewer parameters. Yet, this “all-to-all” connection approach often treats every layer as equally valuable. Not every previous layer holds the same relevance at every stage of learning, leading to suboptimal performance in some cases.

This is where Binary Search Connections (BSC) come in. BSC introduces a more refined approach by selectively reinforcing connections based on a binary search-inspired logic. Rather than linking all layers indiscriminately, BSC targets important features and ensures that they are reintroduced at critical stages of the network. This refined connectivity emphasizes the most relevant information, allowing the model to focus on key features without cluttering the network with unnecessary connections. It’s like skipping directly to the most relevant dictionary pages instead of reading each one sequentially.

### 🧠 The Idea Behind Binary Search Connections
Binary Search Connections (BSC) draw inspiration from the binary search algorithm. Just as binary search efficiently narrows down possibilities by halving the search space, BSC-DenseNet introduces strategic connections between layers, focusing on the most critical features. Unlike the exhaustive “all-to-all” connectivity in DenseNet, BSC adds selective connections that emphasize key features as the model deepens.

BSC-DenseNet enhances DenseNet’s architecture by incorporating a hierarchical structure that reinforces important features. Early-layer features are selectively reintroduced at deeper stages, amplifying their influence. This intelligent reinforcement prioritizes the propagation of vital features, improving overall performance while maintaining DenseNet’s core connectivity.

---
### ⚙️ Architecture Overview of BSC-DenseNet
BSC-DenseNet builds on the standard DenseNet architecture, with dense blocks, transitions, and final classification. However, each dense block now includes optional Binary Search Connections, implemented through recursive logic. These connections are incorporated via **concatenation** (not addition), which maintains DenseNet's core identity while enhancing its representational power.

Here’s a look at the recursive implementation of Binary Search Connections in code:

![image](https://github.com/mr-ravin/BSC-DenseNet/blob/main/bsc_densenet.jpg?raw=true)

```
Binary_Search_Connection(start, end, list_keys)
if end-start>0
then
    mid=(start+end)/2
    if start!=end
    then
        if mid-start>2
        then
              list_keys[mid].append(start+1)    // send output of “start+1” index layer to “mid” index layer as input
        end if
    end if
    if end-mid>2
    then
          list_keys[end].append(mid+1)         // send output of “mid+1” index layer to “end” index layer as input
    end if
    Binary_Search_Connection(start+1, mid-1, list_keys)
    Binary_Search_Connection(mid+1, end-1, list_keys)
end if

# Function call:
# list_keys = [[] for _ in range(number_of_layers)]
# Binary_Search_Connection(0, number_of_layers - 1, list_keys)
```

```
+-----------------------------------------------+--------------------------+
|        Layers     |  Inputs from other layers | Inputs from other layers |
|                   |      (in DenseNet)        |    (in BSC-DenseNet)     |
+-------------------+---------------------------+--------------------------+
|        Layer 0    |      [ ]                  |   [ ]                    |
|        Layer 1    |      [0]                  |   [0]                    |
|        Layer 2    |      [0, 1]               |   [0, 1]                 |
|        Layer 3    |      [0, 1, 2]            |   [0, 1, 2, 1]           |
|        Layer 4    |      [0, 1, 2, 3]         |   [0, 1, 2, 3]           |
|        Layer 5    |      [0, 1, 2, 3, 4]      |   [0, 1, 2, 3, 4]        |
|        Layer 6    |      [0, 1, 2, 3, 4, 5]   |   [0, 1, 2, 3, 4, 5, 4]  |
+-----------------------------------------------+--------------------------+
Note: In BSC-DenseNet, Layer 3 has two inputs from Layer 1, and the Layer 6 has two inputs from Layer 4.
```

**Binary Search Connections (BSC)** are implemented via **concatenation**, preserving DenseNet’s feature aggregation design. In BSC-DenseNet, additional binary search-inspired connections introduce **repeated inputs from specific earlier layers**, effectively **reinforcing important feature paths**. This selective duplication **promotes stronger gradient flow** through key connections, enhancing representation learning.

--------
### Implementation in Pytorch
##### Important: Densenet-121 code is taken from `torchvision`library, from its [github repository](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py); Code of DenseNet-121 is modified to create BSC-Densenet-121 presenet in our `densenet.py` file.

###### Deep Learning Framework: Pytorch

###### Tested with python version: >=3.7 and <= 3.13.2 on Ubuntu.

- ##### Densenet-121 Model with BSC [Binary Search Connection]
```python
from densenet import get_BSC_Densenet_121_model
BSC_DenseNet_121_Model = get_BSC_Densenet_121_model(num_class=100)
```

- ##### Densenet-121 Model without BSC [Binary Search Connection]
```python
from densenet import get_Densenet_121_model
DenseNet_121_Model = get_Densenet_121_model(num_class=100)
```
---
### 📊 Performance Breakdown
Run the below terminal command to train and compare performance of BSC-DenseNet-121 with DenseNet-121 on CIFAR-1OO Dataset.

```python
python3 run.py --device cuda
```

#### 🧪 Experiment 1 : DenseNet vs BSC-DenseNet on CIFAR 100 Dataset
We compared the performance of DenseNet and BSC-DenseNet on Cifar 100 dataset for classification task. Following default values were used for both the models: **Growth rate = 32**, Block Config = (6, 12, 24, 16), and Number of Initial Features = 64.

⚠️ **Important**: In this experiment, DenseNet has a **lesser number of trainable parameters** compared to BSC-DenseNet.

```
After Epoch = 20
+------------------------------------+------------------------+-------------------------+
|                Model               |  Trainable Parameters  | Accuracy (on CIFAR-100) |
+------------------------------------+------------------------+-------------------------+
| DenseNet-121      (growth rate=32) |        7,056,356       |          50.52          |
| BSC-DenseNet-121  (growth rate=32) |        7,574,756       |          51.23          |
+------------------------------------+------------------------+-------------------------+
```
Overall Analysis is stored in visual graphs inside `ExperimentResults/EXP1_overall_analysis.png`.
![image](https://github.com/mr-ravin/BSC-DenseNet/blob/main/ExperimentResults/EXP1_overall_analysis.png?raw=true)


#### 🧪 Experiment 2: DenseNet vs BSC-DenseNet on CIFAR 100 Dataset
To assess whether the improved performance of BSC-DenseNet-121 stems from its architectural design or simply from having more trainable parameters, we conducted a controlled comparison.

⚠️ **Important**: In this experiment, DenseNet has a **greater number of trainable parameters** compared to BSC-DenseNet.

Specifically, we compared `BSC-DenseNet-121` with a `growth rate of 32` (totaling `7,574,756 trainable parameters`) against a vanilla `DenseNet-121 with a higher growth rate of 34` (resulting in `7,936,319 trainable parameters`). Despite having significantly fewer parameters, **BSC-DenseNet-121 outperformed the larger DenseNet-121**, suggesting that the binary search connections (BSC) contribute meaningfully to the model's effectiveness rather than mere parameter count.

```
After Epoch = 20
+------------------------------------+------------------------+-------------------------+
|               Model                |  Trainable Parameters  | Accuracy (on CIFAR-100) |
+------------------------------------+------------------------+-------------------------+
| DenseNet-121     (growth rate=34)  |        7,936,319       |          52.63          |
| BSC-DenseNet-121 (growth rate=32)  |        7,574,756       |          53.22          |
+------------------------------------+------------------------+-------------------------+
```
Overall Analysis is stored in visual graphs inside `ExperimentResults/EXP2_overall_analysis.png`.
![image](https://github.com/mr-ravin/BSC-DenseNet/blob/main/ExperimentResults/EXP2_overall_analysis.png?raw=true)

**In short**: BSC-DenseNet learns faster, generalizes better, and resists overfitting than DenseNet.

---

#### Conclusion

This paper proposed Binary Search Connections (BSC) as a novel architectural paradigm for deep convolutional networks, leveraging recursive logic inspired by binary search to guide feature propagation. Unlike DenseNet’s uniform all-to-all connectivity, BSC introduces structured and selective recurrence—intentionally repeating key inputs to reinforce critical features at deeper layers. This approach encourages efficient gradient flow and focused feature learning while avoiding indiscriminate connections. Experimental results on the CIFAR-100 dataset demonstrate that BSC-DenseNet achieves superior performance even when compared to larger parameter-count DenseNet variants, validating that the architectural innovation, not mere capacity, underlies the improvement. BSC-DenseNet thus offers a principled mechanism for hierarchical feature emphasis, marking a conceptual step forward in network design.

---

```
Copyright (c) 2023 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
