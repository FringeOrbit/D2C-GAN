"""
优化后的配置文件
"""

import torch

class OptimizedConfig:
    # 1. 更新为预处理后的数据集路径
    data_path = "./data/train_processed.csv"
    test_path = "./data/test_processed.csv"
    val_path = "./data/val_processed.csv"

    # 2. 新增全局 Scaler 的路径
    scaler_path = "./data/scaler.pkl"
    target_plot_well = '15/9-15'

    # 3. ⚠️ 极其重要：分隔符改为逗号
    raw_delimiter = ';'  # 你的原始测井数据使用的分隔符
    processed_delimiter = ','  # Pandas 处理后生成的标准 CSV 分隔符
    well_column = 'WELL'
    depth_column = 'DEPTH_MD'

    # 特征
    feature_names = ['GR', 'RHOB', 'NPHI', 'DTC']
    num_features = 4

    # 模型参数
    latent_dim = 128  # 增大隐空间

    generator_dims  = [64, 128, 256, 512]
    # 生成器参数（稍微增大）
    gen_hidden_dims = [256, 512, 256]

    # 判别器参数
    disc_hidden_dims = [128, 64, 32]

    # 训练参数 - 重要调整
    batch_size = 512
    epochs = 200  # audited server default; not a complete historical run manifest
    # ⚠️ 降低学习率
    learning_rate_g = 0.00005  # 从0.001降低到0.0002
    learning_rate_d = 0.000005  # 从实验 19 的 0.00001 进一步压低

    beta1 = 0.5
    beta2 = 0.999


    # 训练策略
    missing_rate_range = (0.1, 0.4)  # 缺失率范围

    # 学习率调度
    use_scheduler = True
    scheduler_patience = 10
    scheduler_factor = 0.5

    # 早停
    use_early_stopping = True
    early_stopping_patience = 30

    # 梯度裁剪
    grad_clip = 1.0

    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 保存路径
    checkpoint_dir = './optimized_checkpoints'
    result_dir = './optimized_results'

    # 训练策略 - 新增序列长度
    seq_len = 80  # 80-point windows
    missing_len_range = (25,75) # ⚠️ 新增参数：模拟真实的长段缺失

    # 损失权重 - 提取到 config 以便动态修改
    l1_loss_weight = 200.0  # 整体重构
    weight_adv_struct = 0.05  # 结构对抗
    weight_adv_texture = 0.002  # 纹理对抗
    weight_recon_missing = 500.0  # 缺失部分重构
    weight_petro = 10.0  # 物理约束
    weight_boundary = 5.0  # 边界感知
    weight_tv = 1.0  # 平滑度

    # === 网络架构控制开关 (用于消融实验) ===
    use_dilation = True  # 是否使用空洞卷积扩大感受野
    use_cbam = True  # 是否使用 CBAM 空间-通道注意力
    use_fft = True  # 是否使用频域特征提取模块
    use_learned_branch_weights = True  # five parallel blocks with two-way Softmax

    loss_type = 'huber'  # 可选值: 'huber', 'mse', 'l1'
