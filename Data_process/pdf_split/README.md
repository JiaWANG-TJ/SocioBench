# PDF 拆分工具

这是一个简单易用的 PDF 拆分工具，可以根据指定的页码范围将一个 PDF 文件拆分为多个子 PDF 文件。

## 功能特点

- 支持按页码范围拆分 PDF 文件
- 可以同时指定多个拆分范围
- 自动创建输出目录
- 提供友好的命令行界面
- 支持命令行参数和交互式输入两种方式

## 使用方法

### 方法一：直接运行脚本并指定页码范围（推荐）

```bash
python pdf_splitter.py <pdf文件路径> <页码范围1> <页码范围2> ...
```

例如：

```bash
python pdf_splitter.py example.pdf 1-50 51-100 200-300
```

### 方法二：交互式输入页码范围

```bash
python pdf_splitter.py <pdf文件路径>
```

然后按照提示输入页码范围，多个范围之间用空格分隔。

### 方法三：使用运行脚本

修改 `run_splitter.py` 中的 PDF 文件路径和页码范围，然后运行：

```bash
python run_splitter.py
```

## 页码范围格式

页码范围的格式为 `起始页-结束页`，例如：

- `1-10`：从第 1 页到第 10 页
- `11-20`：从第 11 页到第 20 页

注意：页码从 1 开始计数。

## 输出文件

拆分后的 PDF 文件将保存在源 PDF 文件所在目录下的 `<pdf文件名>_split_output` 文件夹中。

输出文件的命名格式为：`<原文件名>_<页码范围>.pdf`

## 依赖库

- PyPDF2：用于 PDF 文件处理

可以使用以下命令安装：

```bash
pip install PyPDF2
```

## 示例

假设有一个名为 `example.pdf` 的 PDF 文件，包含 100 页，要将其拆分为 1-30 页、31-60 页和 61-100 页三个 PDF 文件：

```bash
python pdf_splitter.py example.pdf 1-30 31-60 61-100
```

执行后，将在 `example_split_output` 目录下生成以下三个文件：

- `example_1-30.pdf`
- `example_31-60.pdf`
- `example_61-100.pdf` 