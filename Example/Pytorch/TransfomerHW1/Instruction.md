# HW1 环境配置

在 PowerShell 中进入项目根目录，将 `src` 加入 Python 模块搜索路径：

```powershell
cd D:\Coding_Python\AI-Infra-guide-based-on-CS336
$env:PYTHONPATH = "$PWD\src"
```

之后即可在脚本中使用 `from minbpe import RegexTokenizer`。路径应指向 `src`，不是 `src/minbpe`。此设置仅在当前终端及其启动的进程中生效，新开终端需要重新执行。

完整准备步骤见 [HW1 初始准备](../../../note/TransfomerHW1/Hw1.md)。
