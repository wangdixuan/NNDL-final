# NNDL-final

## 文件架构
```
Self-supervised Learning
├── moco                            # 实现moco算法所需工具
│   └── builder.py
│   └── loader.py
├── main_moco.py                    # 使用moco算法进行预训练
├── main_lincls.py                  # 使用moco算法进行分类头分类
├── supervised.py                   # 使用监督学习进行训练
├── resent8.py                      # 直接训练resnet18模型
└── README.md                       # 本文件
```

## 模型训练
### 自监督MoCo
1. 预训练
   
   ```
   python main_moco.py \
   -a resnet18 \
   --lr 0.03 \
   --batch-size 256 \
   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
   [folder with train and val folders]

2. 分类头
   
   ```
   python main_lincls.py \
   -a resnet18 \
   --lr 30.0 \
   --batch-size 256 \
   --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
   [folder with train and val folders]
   ```

上述参数可根据需要进行修改。

### 监督学习
1. 预训练
   
   修改train_dataset及val_dataset变量路径为数据集所在路径。
   ```
   python ./supervised.py
   ```
2. Tensorboard可视化
   ```
   tensorboard --logdir='./logs/runs_cls'
   ```


### 直接训练ResNet18
1. 预训练
   
   修改train_dataset及val_dataset变量路径为数据集所在路径。
   ```
   python ./resnet18.py
   ```
2. Tensorboard可视化
   ```
   tensorboard --logdir='./logs/runs_runs_cifar100_resnet18'
   ```



