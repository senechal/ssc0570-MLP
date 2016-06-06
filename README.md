# Trabalhos de Redes Neurais
Trabalhos realizados utilizando ambiente LINUX, de preferencia utilize o mesmo ambiente para executar o programa.      


## Requisitos:

    Python 2.7
        pip

### Instalação:
Antes de rodar o programa é necessario instalar seus requisitos, como em qualquer projeto python para isso é utilizado um ambiente virtual para que as configurações originais do sistema não seja modificadas, para isso utilizaremos o virtualenv:

    sudo pip install virtualenv
        
Apos a instalação do mesmo, estraia o programa ou clone o repositorio:

    https://github.com/senechal/ssc0570-Redes-Neurais.git

rode:

    virtualenv env
                    
Isso ira criar um ambiente virtual para instalarmos os requerimentos do programa.
Instalando requerimentos:

    pip install -r requerimentos.txt
                     
## Trabalho 1 - MLP

### Execução
Para executar o programa rode:

    python run.py mlp --config=mlp_config.json --test=mlp_test.json --train=mlp_train.json
                                    
O programa utiliza esse arquivos json para configurar e apontar a localização das bases de dados.

## Trabalho 2 - SOM

### Execução
Para executar o programa rode:

    python run.py som --config=som_config.json --test=som_test.json --train=som_train.json
                                    
O programa utiliza esse arquivos json para configurar e apontar a localização das bases de dados.

Para o SOM, existem duas maneiras de treinar a rede, uma utilizando os dados de teste existentes no som_train.json de maniera sequencial, ou utilizando entradas randomicas apartir dos dados de testes. Para mudar esse conportamento, modifique no arquivo de configuração o campo test para "random" e o campo "iterations" (Numero de iterações).


