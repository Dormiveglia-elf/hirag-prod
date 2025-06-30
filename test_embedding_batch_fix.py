#!/usr/bin/env python3
"""
测试脚本：验证embedding批处理修复
解决大量chunk时OpenAI API "input invalid" 错误

使用方法：
1. 确保设置了OPENAI_EMBEDDING_API_KEY和OPENAI_EMBEDDING_BASE_URL环境变量
2. 运行: python test_embedding_batch_fix.py
"""

import asyncio
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)


async def test_embedding_batch_processing():
    """测试embedding服务的批处理功能"""
    try:
        from hirag_prod._llm import EmbeddingService

        print("🧪 测试embedding批处理功能...")

        # 创建测试数据 - 模拟大量chunks的情况
        test_texts = []

        # 生成不同长度的测试文本，模拟真实的chunk内容
        base_texts = [
            "This is a test document chunk about artificial intelligence and machine learning.",
            "Natural language processing is a fascinating field that combines linguistics and computer science.",
            "Large language models have revolutionized the way we process and understand text.",
            "Vector databases are essential for storing and retrieving high-dimensional embeddings.",
            "Knowledge graphs provide a structured way to represent relationships between entities.",
        ]

        # 扩展到足够大的数量以测试批处理（模拟3000+的情况）
        for i in range(600):  # 这会创建3000个文本
            for base_text in base_texts:
                test_texts.append(
                    f"Document {i+1}: {base_text} Additional content for variety {i}."
                )

        print(f"📊 生成了 {len(test_texts)} 个测试文本")

        # 测试不同的批处理大小
        batch_sizes_to_test = [1000, 500, 100]

        for batch_size in batch_sizes_to_test:
            print(f"\n🔄 测试批处理大小: {batch_size}")

            # 创建embedding服务实例
            embedding_service = EmbeddingService(default_batch_size=batch_size)

            start_time = time.perf_counter()

            try:
                # 执行embedding生成
                embeddings = await embedding_service.create_embeddings(
                    texts=test_texts, model="text-embedding-3-small"
                )

                end_time = time.perf_counter()
                elapsed = end_time - start_time

                print(f"✅ 成功生成 {len(embeddings)} 个embeddings")
                print(f"⏱️ 耗时: {elapsed:.2f} 秒")
                print(f"📈 速度: {len(test_texts) / elapsed:.1f} texts/秒")
                print(f"🎯 Embedding维度: {embeddings[0].shape}")

                # 验证结果完整性
                assert len(embeddings) == len(
                    test_texts
                ), f"Embedding数量不匹配: {len(embeddings)} vs {len(test_texts)}"
                print(f"✅ 数据完整性验证通过")

                break  # 成功了就不用测试更小的batch size

            except Exception as e:
                print(f"❌ 批处理大小 {batch_size} 失败: {e}")
                if "batch" in str(e).lower() or "limit" in str(e).lower():
                    print(f"⚠️ 可能需要更小的批处理大小")
                    continue
                else:
                    raise e

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保hirag_prod包在Python路径中")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


async def test_hirag_with_batch_config():
    """测试HiRAG主类的批处理配置"""
    try:
        from hirag_prod.hirag import HiRAG

        print("\n🧪 测试HiRAG批处理配置...")

        # 使用不同的embedding_batch_size创建HiRAG实例
        hirag = await HiRAG.create(embedding_batch_size=500)

        print(f"✅ HiRAG实例创建成功")
        print(f"📊 配置的embedding批处理大小: {hirag.embedding_batch_size}")
        print(
            f"📊 实际embedding服务批处理大小: {hirag.embedding_service.default_batch_size}"
        )

        await hirag.clean_up()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保hirag_prod包在Python路径中")
    except Exception as e:
        print(f"❌ HiRAG配置测试失败: {e}")


async def main():
    """主测试函数"""
    print("🚀 开始embedding批处理修复测试\n")

    # 测试1：基础embedding服务批处理
    await test_embedding_batch_processing()

    # 测试2：HiRAG配置
    await test_hirag_with_batch_config()

    print("\n🎉 所有测试完成!")
    print("\n📋 使用建议:")
    print("1. 对于大量chunks (3000+)，建议设置 embedding_batch_size=500 或更小")
    print("2. 如果仍遇到API限制，系统会自动减小批处理大小重试")
    print("3. 通过HiRAG.create(embedding_batch_size=500)设置批处理大小")
    print("4. 监控日志输出以了解批处理进度和自动调整")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())
