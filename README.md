# visiters_counter

 来場者の属性を判別し，記録します．python3.11以降の環境で動作します．

## インストール

1. Python3.11以降の仮想環境を用意する．
1. Ver.1.13以上でcuda対応のpytorchを導入する．
1. [MiVOLO](https://github.com/WildChlamydia/MiVOLO)の[Install](https://github.com/WildChlamydia/MiVOLO/blob/main/README.md#install)に従ってMiVOLOを導入する．
1. MiVOLOリポジトリの[Demo](https://github.com/WildChlamydia/MiVOLO/blob/main/README.md#demo)からYOLOとMiVOLOのチェックポイントを`visitor-counter/models`へダウンロードする．
1. `visitor-counter`ディレクトリに移動し，以下のコードを実行
```console
pip install -r requirements.txt
```

## demoの実行方法

1. `detector_weights`, `checkpoint`にそれぞれYOLO, MiVOLOのチェックポイントを指定する．
1. visiters_counterディレクトリに移動し，以下のコードで`demo.py`を実行する．

```console
python3 demo.py
```

## References

> [**MiVOLO: Multi-input Transformer for Age and Gender Estimation**](https://arxiv.org/abs/2307.04616),
> Maksim Kuprashevich, Irina Tolstykh,
> *2023 [arXiv 2307.04616](https://arxiv.org/abs/2307.04616)*
