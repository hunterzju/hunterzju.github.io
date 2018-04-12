CTF题目中一道编程类题目，题目来自[浙江大学AAA战队](https://zjusec.com)。

## 题目描述：

>噢，这有个链表？  
nc 10.214.xxx.xxx xxxx  
首先要求计算一个随机字符串的md5，请输入小写的32个字符  

给出了一段提示：
>Hint:
或许有个memcache的python包会简单很多呢？

## 解题过程
按照提示用nc连接服务器端口，返回消息如下：
```shell
Please calculate md5(mexui)=
```
用md5模块计算后输入结果又得到了如下的返回消息：
```
==================================================
You must be good at linked-list,
but how about "memcache"?
So... (I won't tell you "nc" is enough)
Let's crack a linked-list on memcache! Ovo
==================================================
For example: 
    start -> 1 -> 2 -> flag
You should "get start", then "get 1", "get 2", 
finally you will get the flag~
==================================================
Please Connect this memcached sever:
               10.214.xxx.xxx:xxxxx
***Attention***:
This memcache server will be removed after 5 minutes
HURRY UP! LEARN MEMCACHE! 


I won't tell you "nc" is enough

```
提示说有一个链表，通过memcache从服务器端口获取里面的值最终得到flag。  
最后还说nc就够了，但是用nc是有点nc了啊（我不会说我还真试了下）  
这里下载一波php-memcache和python-memcache，然后就可以轻松解题了：
```python
mc = memcache.Client([memcache_addr])
value = mc.get("start")
while 1:
    value = mc.get(value)
    print(value)
    if value == None:
        sys.exit()
```
这里的memcache_addr是通过pwntools建立socket通信后用正则表达式获取的。  
flag后会有一个None值，检测以下加个sys.exit()，看着舒服点。