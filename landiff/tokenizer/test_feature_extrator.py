import torch

from landiff.tokenizer.models.feature_extractor.theia_extractor import TheiaExtractor


def test_theia_extractor_with_random_data():
    """测试使用随机数据的TheiaExtractor"""
    print("测试使用随机数据的TheiaExtractor")

    # 创建随机图像数据，模拟单张图像 - [1, 3, 256, 256]形状的uint8张量
    random_image = torch.randint(0, 256, (4, 3, 720, 720), dtype=torch.uint8)
    print(f"输入图像形状: {random_image.shape}, 数据类型: {random_image.dtype}")
    print(f"图像数值范围: [{random_image.min().item()}, {random_image.max().item()}]")

    # 初始化特征提取器，使用小型模型以加快测试速度
    extractor = TheiaExtractor(
        pretrained_model_name_or_path="theaiinstitute/theia-base-patch16-224-cddsv",
        image_size=(720, 720),
        interpolate=True,  # 启用插值以测试不同尺寸的处理
        bfp16=True,
    )

    # 设置为评估模式并转移到可用设备
    extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    extractor = extractor.to(device)
    random_image = random_image.to(device)

    # 提取特征
    with torch.no_grad():
        features = extractor(random_image)

    print(f"提取的特征形状: {features.shape}")
    print(f"特征数值范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
    return features


if __name__ == "__main__":
    # 运行所有测试
    features = test_theia_extractor_with_random_data()

    print("\n所有测试完成!")
