# helipad-pose

Os ficheiros `dataset/data.yaml` e `dataset_v2/data.yaml` são portáteis: os
caminhos de treino e validação são relativos à pasta do respetivo YAML. Os dados
(imagens e labels) não são versionados; cada pessoa deve colocá-los nas pastas
indicadas pelo YAML que pretende usar. Se houver um conjunto de teste, adiciona
`test: images/test` ao YAML.

As anotações mantêm sempre a mesma numeração para cada ponto, incluindo os
pontos à esquerda e à direita. Como o treino aplica inversões horizontais
(`fliplr=0.5`), `flip_idx` em `dataset/data.yaml` indica ao YOLO que índices
trocar durante essa transformação: `0↔5`, `1↔4`, `2↔3`, `6↔11`, `7↔10` e
`8↔9`. Assim, os índices continuam a representar os mesmos pontos após a
inversão.