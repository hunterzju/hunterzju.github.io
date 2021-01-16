# 在ubuntu中用sourceTrail追踪Linux源码
## 编译最小linux内核
下载linux源码后，采取最小编译，并借助bear生成编译文件列表，然后在sourcetrail追踪。
linux的最小编译配置
```
make tinyconfig
```
然后可以通过make menuconfig调整配置
```
bear make -j8
```
采用生成的json文件在sourceTrail中新建c工程。

** 注 **
* sourceTrail对大型工程索引能力有限，linux编译最好配置为最小编译

## 单独编译内核驱动模块
追踪某一个内核驱动源码的时候可以单独编译对应的驱动生成记录编译文件的json列表，通过json文件建立sourcetrail工程。
直接按照完整编译配置
```
make menuconfig
```
然后编译对应的驱动目录
```
bear make ./driver/pci
```
然后用json建立sourcetrail工程