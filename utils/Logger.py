import logging
import os


def get_logger(file_name, data_name, data_rate, current_time):
    """创建/复用按（数据集, 扰动比例）区分的 logger。

    日志文件：1.logs/{data_name}/{data_name}.log
    （每个数据集一个子目录、一个固定 log 文件，多次实验追加/覆盖写入；
     时间与扰动比例 时长由日志内容记录——每行带 asctime，比例通过 logger 名区分）。
    """
    filename = os.path.join("1.logs", data_name, f"{data_name}.log")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # logger 名含数据集与比例：不同数据集互不干扰；同数据集重复调用时
    # 先清掉旧 handler 再重绑，避免同一行日志被重复打印/写入
    logger = logging.getLogger(f"{os.path.basename(file_name)}_{data_name}_{data_rate}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 不向 root logger 传播，防止重复输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
