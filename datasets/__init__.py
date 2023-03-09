import importlib

#! 1. __init__.py的作用： https://www.jianshu.com/p/73f7fbf75183
#;   简单来说，就是加了__init__.py之后，from datasets import xx的时候就会自动执行这里面的代码，
#;   但是这个文件中只是定义了一个函数，所有没有代码可以执行
#! 2.importlib动态导入对象的作用：https://blog.csdn.net/xie_0723/article/details/78004649

# find the dataset definition by name, for example dtu_yao (dtu_yao.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)  # 模块名称为datasets.dtu_yao
    #; import_module动态导入对象，把datasets.dtu_yao这个包导入
    module = importlib.import_module(module_name)  

    #; getattr获取对象属性值，见：https://www.runoob.com/python/python-func-getattr.html
    #; 虽然说datasets.dtu_yao是一个包，但是也可以把它看成一个对象，然后后面传入"MVSDataset"就是获取这个
    #;   包中的对象，从而就返回了dataset.dtu_yao.MVSDataset这个对象
    return getattr(module, "MVSDataset")  # 
