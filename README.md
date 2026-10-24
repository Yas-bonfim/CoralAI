# CoralAI

Detecção de corais saudáveis e branqueados com YOLOv8, treinada a partir de datasets de classificação do Kaggle. O modelo recebe uma imagem, um vídeo ou a webcam e devolve, para cada detecção, um JSON no formato:

json
{"timestamp": 1729432800.125, "label": "bleached_coral", "confidence": 0.94, "bbox": [120, 85, 220, 185]}

O campo bbox está em pixels, no formato [x1, y1, x2, y2].

Aviso importante sobre os dados

Os datasets usados são de classificação: possuem apenas imagens separadas em pastas por classe, sem bounding boxes. Para treinar um detector, o projeto gera pseudo-boxes automaticamente com o YOLO-World (prompt "coral"). Quando nada é detectado, a box passa a ser a imagem inteira, com uma pequena margem.

Consequência prática: o modelo aprende a classificar bem, mas a localização é apenas aproximada. Para um detector de verdade, revise as boxes em uma ferramenta de anotação (Roboflow, CVAT, Label Studio) e treine novamente.

Datasets
Healthy and Bleached Corals Image Classification (já traz as pastas Training, Validation e Testing)
Corals Classification (opcional, usado em conjunto com o primeiro)

Baixe e extraia os datasets na pasta do projeto. As classes são unificadas pelo nome da pasta (bleached_corals vira bleached_coral, healthy_corals vira healthy_coral) e imagens duplicadas entre os datasets são removidas por hash.

Estrutura
CoralAI/
├── Bleached Corals and Healthy Corals Classification/   # dataset 1
│   ├── Training/   (bleached_corals, healthy_corals)
│   ├── Validation/ (bleached_corals, healthy_corals)
│   └── Testing/    (bleached_corals, healthy_corals)
├── config.py             # caminhos e funções compartilhadas
├── prepare_dataset.py    # gera o dataset YOLO e as pseudo-boxes
├── train.py              # fine-tuning do YOLOv8
├── predict.py            # inferência e saída em JSON
├── run_pipeline.py       # executa prepare -> train -> predict
├── environment.yml       # ambiente conda
├── coral_yolo_data/      # gerado pelo prepare_dataset.py
└── runs_coral/           # gerado pelo train.py
Instalação
bash
conda env create -f environment.yml
conda activate coralai

O ambiente instala Python 3.11, numpy, pillow, opencv-python e ultralytics (que traz o PyTorch). Para treinar com GPU NVIDIA, instale o PyTorch com suporte a CUDA seguindo o seletor em pytorch.org e confira com:

bash
python -c "import torch; print(torch.cuda.is_available())"
Uso

Execute sempre a partir da pasta do projeto, nesta ordem.

1. Preparar o dataset
bash
python prepare_dataset.py

Lê as imagens, respeita os splits Training, Validation e Testing, gera as pseudo-boxes e cria coral_yolo_data/. Se o dataset já existir, a etapa é pulada.

Opção	Efeito
--extra-root "pasta"	adiciona outro dataset (pode repetir); sem pastas de split, divide 80/10/10
--full-image-boxes	não usa o YOLO-World; a box é a imagem inteira
--force	recria o dataset mesmo que ele já exista
2. Treinar
bash
python train.py --weights yolov8n.pt --epochs 80 --batch 16 --device 0

O melhor modelo fica em runs_coral/yolov8_coral/weights/best.pt. Se ele já existir, o treino é pulado; use --force para retreinar. Sem GPU, omita --device (o treino fica bem mais lento).

3. Detectar
bash
python predict.py --source foto.jpg --draw
python predict.py --source video.mp4 --draw
python predict.py --source 0 --draw            # webcam, "q" para sair

Cada detecção é impressa como uma linha de JSON. Com --draw, as imagens e os vídeos anotados são salvos em saida_anotada/ e a webcam abre uma janela ao vivo. O limiar de confiança é ajustável com --conf (padrão 0.25).

Rotina completa
bash
python run_pipeline.py                              # prepara e treina o que faltar
python run_pipeline.py --source foto.jpg --draw     # inclui a inferência
python run_pipeline.py --force                      # refaz tudo do zero
Automação no Linux (cron)

O cron não carrega o ambiente do conda, então use o Python do ambiente com o caminho completo. Exemplo, retreinando todo domingo às 3h:

0 3 * * 0 cd /home/usuario/CoralAI && /home/usuario/anaconda3/envs/coralai/bin/python train.py --force --epochs 80 --device 0 >> train.log 2>&1

Cada execução com --force sobrescreve o best.pt; copie o arquivo antes se quiser guardar versões anteriores.

Limitações
As boxes de treino são automáticas, então a precisão da localização é limitada.
Se os datasets contiverem imagens muito parecidas (por exemplo, frames do mesmo recife), pode haver vazamento entre treino e validação, e as métricas saem otimistas.
Se você corrigir as boxes manualmente em coral_yolo_data/, não use --force no prepare_dataset.py, pois isso recria o dataset e apaga as correções.
O mapeamento de classes depende dos nomes das pastas; confira a lista de classes impressa ao final do prepare_dataset.py.
