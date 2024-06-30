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
