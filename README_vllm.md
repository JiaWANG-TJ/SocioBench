# vLLM连接指南

## 1. 概述

我们对`LLMAPIClient`类进行了增强，使其现在能够更灵活地连接到vLLM服务器。以下是主要变更：

1. 添加了`vllm_host`和`vllm_port`参数，允许用户指定vLLM服务器的地址和端口
2. 将vLLM模式下的模型名称固定为`default_model`，这样vLLM服务器会使用已加载的模型
3. 改进了错误处理，提供更明确的错误消息
4. 添加了测试脚本和文档，方便用户进行连接测试和故障排除

## 2. 使用方法

### 2.1 初始化客户端

```python
from evaluation.llm_api import LLMAPIClient

# 默认连接到localhost:8000
client = LLMAPIClient(api_type="vllm")

# 指定主机和端口
client = LLMAPIClient(
    api_type="vllm",
    vllm_host="your-server-ip",
    vllm_port=8000
)
```

### 2.2 调用API

```python
response = client.call([
    {"role": "user", "content": "你好，请做个简单的自我介绍"}
])
print(response)
```

## 3. 启动vLLM服务器

在使用vLLM模式之前，您需要先启动vLLM服务器。以下是启动步骤：

1. 安装vLLM:
   ```bash
   pip install vllm
   ```

2. 启动vLLM服务器(使用OpenAI兼容API):
   ```bash
   python -m vllm.entrypoints.openai.api_server --model [您的模型路径] --host localhost --port 8000
   ```

   示例:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model THUDM/chatglm3-6b --host localhost --port 8000
   ```

## 4. 测试连接

我们提供了一个测试脚本来检查vLLM连接是否正常：

```bash
python test_vllm.py --host localhost --port 8000
```

如果连接成功，您将看到模型的响应。如果连接失败，脚本会提供一些故障排除建议。

## 5. 常见问题

### 5.1 连接错误

如果遇到连接错误，请检查：

1. vLLM服务器是否已经启动
2. 主机和端口是否正确
3. 网络连接是否正常
4. 服务器和客户端之间是否有防火墙阻止

### 5.2 其他错误

如果遇到其他错误，请检查：

1. vLLM是否正确安装
2. 模型是否正确加载
3. 服务器日志中是否有错误信息

## 6. 参考资料

- [vLLM官方文档](https://github.com/vllm-project/vllm)
- [OpenAI API兼容性](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py) 