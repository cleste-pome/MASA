import logging


def get_logger(file_name, data_name, data_rate):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    # 多数据集循环会多次调用本函数：先清掉旧 handler，避免同一行日志被重复打印/写入
    # （每次调用都会换用当前数据集对应的日志文件）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    filename = "./1.logs/" + data_name + data_rate + ".log"
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
