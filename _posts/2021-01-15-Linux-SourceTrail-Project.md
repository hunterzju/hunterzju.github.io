# 在ubuntu中用sourceTrail追踪Linux源码
下载linux源码后，采取最小编译，并借助bear生成编译文件列表，然后在sourcetrail追踪。
linux的最小编译配置
make tinyconfig
然后可以通过make menuconfig调整配置
bear make -j8
采用生成的json文件在sourceTrail中新建c工程。

** 注意 **
* sourceTrail对大型工程索引能力有限，linux编译需要配置为最小编译