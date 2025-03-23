## Implementation of Binary Search Connections in DenseNet (BSC-DenseNet-121) using Pytorch
This repository includes the implementaiton of **BSC-Densenet-121** of research paper `"Adding Binary Search Connections to Improve DenseNet Performance"`, published in **Elsevier-SSRN conference proceedings of NGCT 2019**. The base code of openly available DenseNet is also present in this repository for comparing our BSC-DenseNet on the CIFAR100 dataset.

**Paper Title**: Adding Binary Search Connections to Improve DenseNet Performance

**Author**: [Ravin Kumar](https://mr-ravin.github.io/)

**Publication**: 27th February 2020

**Published Paper**: [click here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3545071)  <!-- Elsevier-SSRN conference proceedings of NGCT 2019 -->

**Doi**: [DOI Link of Paper](http://dx.doi.org/10.2139/ssrn.3545071)

**Other Sources**:
- [Research Gate](https://www.researchgate.net/publication/339673672_Adding_Binary_Search_Connections_to_Improve_DenseNet_Performance), [Research Gate - Preprint](https://www.researchgate.net/publication/382385286_Adding_binary_search_connections_to_improve_DenseNet_performance)
- [Osf.io](https://osf.io/preprints/osf/8z42s_v1)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3545071)
- [Internet Archive](https://archive.org/details/bsc-densenet), [Internet Archive - Preprint](https://archive.org/details/bsc-densenet--preprint)

**GitHub Repository**: [https://github.com/mr-ravin/BSC-DenseNet](https://github.com/mr-ravin/BSC-DenseNet)

##### Cite as:
```
Kumar, Ravin, Adding Binary Search Connections to Improve DenseNet Performance (February 27, 2020). 5th International Conference on Next Generation Computing Technologies (NGCT-2019). Available at SSRN: https://ssrn.com/abstract=3545071 or http://dx.doi.org/10.2139/ssrn.3545071 
```
---
###### Deep Learning Framework: Pytorch

#### Comparing Densenet-121 and BSC-Densenet-121 on CIFAR 100 Dataset
- Trainable Paramaters in `Densenet-121`: `7,056,356`
- Trainable Paramaters in `BSC-Densenet-121`: `7,574,756`
- `Densenet-121` accuracy on test set: `30.48`
- `BSC-Densenet-121` accuracy on test set: `32.33`

```python
python3 run.py
```
Overall Analysis is stored in visual graphs inside `overall_analysis.png`.
![image](https://github.com/mr-ravin/BSC-DenseNet/blob/main/overall_analysis.png?raw=true)

#### Where are the Densenet-121 and BSC-Densenet-121 Models?
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
