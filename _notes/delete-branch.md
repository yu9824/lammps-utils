## リモートリポジトリの削除

**リポジトリの設定にある「Automatically delete head branches」をオンにすれば勝手に消してくれるらしい。**

ローカルに同じ名前のリモートリポジトリを一括削除するコマンド。

```bash
git branch -r | grep -v 'main\|gh-page' | sed 's/origin\///' | xargs -I {} git push origin --delete {}
```

`main` と `gh-page` 以外のリモートリポジトリが削除される。対応するローカルのブランチは削除されない。

マージされていなくても削除されるので注意。

## ローカルリポジトリの削除

```bash
git branch | grep -v "main" | xargs git branch -d
```

`main` 以外のリポジトリが削除できる。

この場合マージしてないブランチは保護される。削除するには `-d` を `-D` にする。
