# リリース手順

## コミット

```bash
git commit -m "Release v0.0.1"  # バージョンは適宜変更
```

## タグ

```bash
git tag v0.0.1
```

タグは頭に'v'をつける以外は、原則 [Semantic Versioning](https://semver.org/lang/ja/) に従う。

e.g.) v1.0.0 v3.2.1-alpha.0 v4.1.2-beta.1 v0.2.1-rc.0


## リリース

PyPIとGithub上でのリリースは1つのワークフローで同時に行えるが、下記の通り初期設定が必要。

### PyPIとGithub

1. `release-pypi.yml` の `# FIXME: uncomment` の部分をアンコメントする。
2. パッケージ名を変える ( `python-template` のところ)

### conda

botを使って自動でレシピを作ってくれる。requirementsは変わらなかったりするので手で直さなきゃいけない場面も多そう。

```bash
git remote add regro-cf-autotick-bot git@github.com:regro-cf-autotick-bot/python-template-feedstock.git
git fetch regro-cf-autotick-bot

```

#### condaのbotによるリリース参考

- [Official Documents](https://conda-forge.org/docs/maintainer/updating_pkgs/)
- [日本語でのブログ記事](https://zenn.dev/pejpo/articles/9f767fa1bf031e)
