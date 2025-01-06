---
title: CodeCompass代码阅读工具
categories:
  - Tools
  - Code
tags:
  - CodeCompass
  - SourceCodeReading
share: true
---

## CodeCompass
CodeCompass 是一个开源的代码理解工具，旨在帮助开发者更好地理解和导航大型软件项目中的代码。以下是一些主要特点：

1. **用户友好的 Web 界面**：CodeCompass 提供了一个直观的 Web 界面，开发者可以通过浏览器轻松地浏览和导航代码元素。
2. **多语言支持**：目前支持 C、C++ 和 Java 语言，正在扩展更多语言的支持。
3. **深度分析**：利用深度语法分析，生成各种图表，如调用路径、继承关系、聚合关系等，帮助开发者更好地理解代码的结构和关系。
4. **高性能**：即使面对大型代码库，CodeCompass 也能保持快速响应，提高工作效率。
5. **可扩展性**：设计上具有高度的可插拔特性，开发者可以根据需要添加新的语言支持或功能模块。

CodeCompass 可以帮助新加入项目的开发者更快速地熟悉代码库，也可以帮助维护历史悠久的项目迅速定位问题根源。此外，它还可以在教育领域中作为辅助教学工具，帮助学生更直观地学习和理解复杂的程序逻辑。

### docker使用
拉取容器：
```bash
docker pull modelcpp/codecompass:runtime-pgsql
```

#### runtime-pgsql容器设置
启动容器镜像：
```bash
docker run -it \
  --volume /home/hunter/workspace:/workspace \
  -p 8010:8080 \
  modelcpp/codecompass:runtime-pgsql \
  /bin/bash
```

在容器中添加对应帐号：
```
apt update && apt install -y sudo
adduser $USERNAME
usermod -aG sudo $USERNAME
```

安装并初始化数据库：
```bash
# 安装postgresql
apt-get install postgresql
# 切换的用户，postgresql无法root权限启动
su - $USERNAME
# 启动数据库，docker里systemctl不可用，用service启动服务
sudo service postgresql start
# 切换到postgres用户创建数据库
su - postgres
psql
```
以下为数据库中操作：
```sql
// 创建compass用户
CREATE USER compass WITH CREATEDB LOGIN PASSWORD '<mypassword>';
// 为compass用户创建数据库
CREATE DATABASE codedb OWNER compass;
// 如果连接数据库失败，则需要通过$USER用户名连接默认数据库postgres，创建对应数据库
CREATE ROLE compass WITH LOGIN PASSWORD 'your_password';
```
回到命令行中初始化数据库目录：
```bash
# 新建数据库目录
/workspace/codecompass_db/codedb
# 初始化数据库，需要绝对路径调用initdb
/usr/lib/postgresql/14/bin/initdb -D /workspace/codecompass_db/codedb/ -E SQL_ASCII
# 端口号冲突，需要停止之前开启的postgresql，或者使用新端口号
sudo service postgresql stop
# 修改socket目录权限
sudo chmod 777 -R /var/run/postgresql/
# 启动容器
/usr/lib/postgresql/14/bin/pg_ctl -D /workspace/codecompass_db/codedb/ -l logfile start
```

#### runtime-pgsql解析项目
codecompass解析命令：
```bash
CodeCompass_parser \
  -d "pgsql:host=localhost;port=5432;user=compass;password=yourpassword;database=codedb" \
  -w ~/workspace \
  -n llvm-mlir \
  -i ~/workspace/llvm-project/build/compile_commands.json \
  -i ~/workspace/llvm-project \
  -j 8
```

#### 启动webserver服务

命令：
```bash
# 容器内端口为8080，对应-p参数值
CodeCompass_webserver -w <workdir> -p <port>
```

启动后可以在浏览器通过`ip:8010`访问codecompass服务。