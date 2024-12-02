---
---

[Joplin](https://joplinapp.org/)是一款免费且开源的笔记和待办事项应用程序，旨在帮助用户高效地整理和管理笔记。它支持Markdown格式，使得笔记的编写、编辑和阅读变得更加便捷和直观。无论是学术研究、工作记录还是日常随笔，Joplin都能提供强大的支持。

## Joplin安装以及插件
Joplin安装直接在官网下载对应安装包即可，这里主要介绍一些比较方便的插件。

### 浏览器插件-JoplinWebCliper
用于剪辑网页资源插入到笔记中，在chrome浏览器和edge浏览器插件市场中可以搜到（国内需要科学上网）。

浏览器插件安装完成后，在Joplin中的`工具`-`选项`-`网页剪藏器`配置页，按照说明配置token。

在浏览器中想收藏的内容，点击插件标签即可保存到Joplin的指定路径下：
![8db58f3b3a37b2694aa46d0494088737.png](../_resources/8db58f3b3a37b2694aa46d0494088737.png)

## 自建多端同步服务器
Joplin支持多种云服务，如Nextcloud、Dropbox、OneDrive和Joplin Cloud，用户可以在多个设备之间无缝地同步笔记。同时，Joplin采用端到端加密技术，确保用户数据的安全性和隐私性。

此外，Joplin也支持自建同步服务器，如果刚好有一个自己的云服务器，可以使用Github上的开源自建方案[JoplinSync](https://github.com/silentz/joplinsync)，其主要功能如下：
* 自建的webdav方式同步笔记
* 自动将笔记同步到github仓库，防止丢失

在自己服务器上clone对应仓库：
```bash
git clone https://github.com/silentz/joplinsync.git
```

然后执行`make`命令，初始化webdav的用户名，nginx的证书和git仓库目录，生成之后需要做对应修改：
* 将`./secrets/webdav_username.txt`和`./secrets/webdav_password.txt`中内容修改为自己的用户名和密码;
* 在云服务商那里申请一个https的安全证书，替换`./secrets/server.key`和`./secrets/server.crt`
* 在自己的github上建立一个仓库用于同步笔记

将笔记目录和github远程仓库关联：
```bash
cd data/
git remote add upstream_01 $Your_Github_repo_addr
```
将$Your_Github_repo_addr替换为自己的github仓库地址

使用docker拉起服务对应的容器：
```
docker-compose up
```

成功后可以看到出现了如下docker容器：
![35d4472ae0265684ba5e57c927773e6d.png](../_resources/35d4472ae0265684ba5e57c927773e6d.png)

在Joplin中配置同步选项：`工具`-`选项`-`同步`标签中配置
![9eab23b805f42832d51c4954aaac12cc.png](../_resources/9eab23b805f42832d51c4954aaac12cc.png)

协议选择webdav，url地址填入对应的服务器地址和端口（云服务记得开放对应端口），用户名和密码即在服务器配置的内容。

## 知识发布
可以使用Joplin插件：[pages-publisher](https://github.com/ylc395/joplin-plugin-pages-publisher)，将想分享的内容发布到自己的github.io上.