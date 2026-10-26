# 博客维护说明

本网站使用 Jekyll + Academic Pages。`master` 保存文章、图片、主题和配置；由 GitHub Pages 构建发布，不需要 Hexo，也不需要手动提交编译产物。

## 本次导入

2026-10-26 导入了 verl 实践系列的五篇文章，原稿来自 `F:\learning\verl`，未修改原稿。文件名日期为发布整理日期，正文保留实验发生日期。

- 文章：`_posts/2026-10-26-verl-01.md` 至 `2026-10-26-verl-05.md`
- 配图：`images/verl/`
- 博客入口：`/year-archive/`

## 换设备继续写作

安装 Git，配置自己的 GitHub 登录凭据后：

```bash
git clone --branch master https://github.com/lighter00x/lighter00x.github.io.git
cd lighter00x.github.io
```

只编辑 Markdown 并交给 GitHub Pages 发布时，不需要安装 Ruby 或 Node.js。也可以直接通过 GitHub 网页编辑文章。

每次开始编辑前运行 `git pull --ff-only`，避免覆盖另一台设备上的更新。

## 新增文章

在 `_posts/` 新建 `YYYY-MM-DD-english-slug.md`，填写真实发布日期、唯一链接和标题：

```markdown
---
title: '文章标题'
date: YYYY-MM-DD
permalink: /posts/english-slug/
excerpt: '一两句话介绍这篇文章。'
tags:
  - 学习笔记
---

{% include toc title="文章目录" %}

正文从这里开始。

## 第一节

![图片说明](/images/your-image.png)
```

将配图一起复制到仓库的 `images/` 目录；不要引用本机盘符或仓库外的文件。文章日期不要设在未来。

## 提交发布

检查 `git status` 和 `git diff`，只暂存要发布的文件。例如：

```bash
git add _posts/YYYY-MM-DD-english-slug.md images/your-image.png
git commit -m "Publish new blog post"
git push origin master
```

推送后，在仓库的 Actions 页面检查 Pages 构建结果。发布成功后访问：

- 网站：https://lighter00x.github.io/
- 博客：https://lighter00x.github.io/year-archive/

如果尚未启用 Pages，在 Settings → Pages 选择 Deploy from a branch，使用 `master` 分支和 `/ (root)` 目录。如果已经使用可正常工作的 GitHub Actions 部署，沿用现有配置。

## 可选：本地预览

只有需要本地预览时才安装 Ruby 和 Bundler。依照仓库 README 中的环境说明安装后，在仓库根目录执行：

```bash
bundle install
bundle exec jekyll serve
```

浏览器打开 `http://localhost:4000`。当前依赖未锁定，本地预览不保证跨设备得到完全相同的依赖版本；写作与线上发布优先使用 GitHub Pages 环境。
