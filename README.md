# TDCS
A simple Task-Dependent EEG Compression method via Subspace projection
## 数据集
数据集可以在 http://bci.med.tsinghua.edu.cn/download.html 免费获取。下载完成后把35个.mat文件放入benchmark_dataset文件夹中。
## 环境配置
### python版本
作者使用的python版本为3.11.8，建议使用3.10及以上的版本，避免出现关于f-string和match case相关的错误。
### requirements.txt
最基本的要求是安装scipy和numpy，把涉及到其他包的代码注释后，也可正常运行。其他包主要用来加速和为其他与TDCS算法无关的其他算法服务。
## 运行
### step 1 下载本项目源码
```shell
git clone https://github.com/IIN-EC-Lab-of-BUPT/TDCS
```
### step 2 将数据放入benchmark_dataset文件夹中
### step 3 切换工作目录或用pycharm以项目打开
```shell
cd ./TDCS
```
### step 4 运行
```shell
python3 demo.py
```
正常运行后，可以在控制台看到如下输出\
`[0.35 0.875 0.875 0.075 0.825 0.7 0.225 0.85 0.925 0.375 0.925 0.825 0.2 0.95 0.85 0.375 0.975 0.925]`

## 联系作者
复现过程中若有问题请发送邮件至lanmao.w@qq.com