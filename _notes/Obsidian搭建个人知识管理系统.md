---
share: true
---

## 自建同步服务 - livesync
开源自建同步服务方案：
https://github.com/vrtmrz/obsidian-livesync/blob/main/README_cn.md

### Couchdb
CouchDB是一种开源的NoSQL数据库，专门设计用于存储和管理JSON文档。它的主要特点是使用RESTful API进行操作，支持高并发访问和分布式数据存储。CouchDB非常适合用于需要高可用性和数据冗余的应用程序，例如移动应用和Web应用。

配置Couchdb参考文档：
https://github.com/vrtmrz/obsidian-livesync/blob/main/docs/setup_own_server_cn.md

使用docker配置Couchdb：
https://github.com/vrtmrz/self-hosted-livesync-server

### Caddy服务器添加SSL功能
移动端访问需要https协议，需要一个ssl证书; Caddy可以比较方便地维护ssl证书。
使用docker配置，需要主要云服务器开放80和443端口，Caddy需要使用。

## 域名解析
域名解析添加A记录或者AAAA记录到服务商的域名解析服务就可以。

## 代理插件
obsidian有时访问不了插件市场，需要挂代理，使用如下插件：
https://github.com/windingblack/obsidian-global-proxy/releases/tag/1.0.4


## github备份
将笔记备份到github，使用插件：
https://github.com/kevinmkchin/Obsidian-GitHub-Sync

## 文章发布
obsidian论坛相关讨论：
https://forum-zh.obsidian.md/t/topic/8852

obsidian发布插件：
https://github.com/Enveloppe/obsidian-enveloppe

 github pages模板：
 https://github.com/maximevaillancourt/digital-garden-jekyll-template

netlify部署示例：
https://maximevaillancourt.com/blog/setting-up-your-own-digital-garden-with-jekyll

