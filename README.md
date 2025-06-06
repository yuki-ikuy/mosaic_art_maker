# mosaic_art_maker
今後はwindows版をアップデートしていく予定です。
基本的にはwindows版でもmacで実行できるように、
OS依存のライブラリは極力使わないように設計はしていますが推奨はしません。  

実際の活用事例として  
大学の文化祭の展示としてモザイクアートの作成に使用しました。
XなどのURLを引用していいかは確認中  

開発スペック  
mac  
CPU:M1  
メモリ:16GB  
python 3.11.5  

windows  
CPU:Core i7-13700KF  
メモリ:32GB
GPU:GeForce RTX 4070 今回は使用しない  
python 3.11.5  

mac版では
複数ファイルによってモザイクアートを作成する。  
main.pyを実行し、GUIによってモザイクアートの作成を行う。  
作成したモザイクアートはoutputフォルダーに保存される。  
一枚までしか保存されないので、前に作成したものは上書きされる。  
※写真の分割数や画質を大きくすると、端が黒く切れるバグが発生するが、修正はまだ行えてない。

windows版では  
pyinstallerによってexe化し、ディスクトップPCに入れ使用することを想定し  　
一つのファイルに統合を行った。  
そのため、mac版とは一部使用の変更を行ったが、作成自体には問題ない。 
また、mac版で発生しているバグに関してはwindows版では修正している。 
作成したモザイクアートはoutputフォルダーに保存される。 
作成したアート名は日時と時間によって命名されるため、複数枚保存することが可能。  

操作の流れ  
1,設定項目を全て決め  
2,設定を保存する  
3,RGBリスト生成を押す  
4,モザイク生成開始を押す  

基本的事項  
※数字は全て半角、数字が大きいほど作成に時間がかかる。  
・設定項目  
元画像：作成したいモザイクアートのデザインとなる画像。バージョンによっては「デザイン画像」になっている。  
タイルフォルダー：モザイクアートを作成するために必要となる写真  
分割数：何分割で作成したいのかを決める。  
例：100 ：100＊100となり10000枚になる。多ければ多いほど細かいモザイクアートになる。  
リサイズ基準：長辺を基準にしており、数字が大きいほど高画質になる。  
同一画像最大使用数：同じ写真を何枚まで使うようにするかを決める  
連続使用制限：同じ写真を何枚連続で使うのかを決める。  
縦方向制限：縦に何枚空けて同じ写真を使用するか決める。  
横方向制限：横に何枚空けて同じ写真を使用するか決める。  
明るさ係数：モザイクアートの明るさを決める。  
色合い係数；色の強調を決める。  
Blur ksize：ぼかしを決める。奇数入力のみ  
DPI：メタデータとして入力 ※バージョンによっては削除  

・チェックボックス　※バージョンによっては削除予定  
縦横連携：　縦横で同じ間隔で写真を空けるか決める。  
明るさ補正　ON：元画像の色を明るさ補正を有効にする。  
色合い補正　ON：元画像の色合い補正を有効にする。  
ガウシアンぼかし　ON：元画像のぼかしを有効にする。  

