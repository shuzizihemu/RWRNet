# 配置环境

为了让您更好地配置环境，我们整理了以下表格，详细列出了您需要的硬件和软件环境。请参考以下内容：

| **配置项**       | **要求**                            |
| :--------------- | :---------------------------------- |
| 操作系统         | Ubuntu 20.04                        |
| CPU              | Intel Core i5 12400F                |
| GPU              | NVIDIA GeForce RTX 3080（12GB显存） |
| 内存             | 32GB                                |
| Python 版本      | 3.7                            |
| 深度学习框架版本 | PaddlePaddle 2.3.0                  |
| CUDA 版本        | 11.6                           |
| cuDNN 版本       | cuDNN 8.4.0                    |



# 安装必要的第三方包

在继续之前，请确保您已安装了以下必要的第三方包。您可以运行以下命令来安装它们：

```
pip install -r requirements.txt 
```

# 下载数据集

您可以从[百度网盘](https://pan.baidu.com/s/1Vquf1WdyW28-e_thgjYsXA?pwd=2022)下载所需的数据集。

# 运行脚本

完成环境配置后，对以下文件`train_config.yml`修改配置，您可以运行以下脚本进行训练：

```
python train.py 
```

运行以下脚本进行测试：

```
python test.py 
```

